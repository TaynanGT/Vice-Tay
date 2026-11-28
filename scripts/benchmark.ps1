param([Parameter(Mandatory)][string]$Recipe, [switch]$ValidateOnly)
$ErrorActionPreference = 'Stop'
$culture = [Globalization.CultureInfo]::InvariantCulture
function Assert($ok, $message) { if (-not $ok) { throw $message } }
function UnixUs([datetime]$value) {
    $utc = $value.ToUniversalTime()
    return [int64](([DateTimeOffset]$utc).ToUnixTimeMilliseconds() * 1000 + [math]::Floor(($utc.Ticks % 10000) / 10))
}
function FullPath([string]$path, [bool]$mustExist = $true) {
    Assert ([IO.Path]::IsPathFullyQualified($path)) "Path must be absolute: $path"
    $full = [IO.Path]::GetFullPath($path)
    if ($mustExist) { Assert (Test-Path -LiteralPath $full) "Path missing: $full" }
    if ($full -eq [IO.Path]::GetPathRoot($full)) { return $full }
    return $full.TrimEnd('\')
}
function ContainsPath([string]$parent, [string]$child) {
    return $child.Equals($parent, [StringComparison]::OrdinalIgnoreCase) -or
        $child.StartsWith("$parent\", [StringComparison]::OrdinalIgnoreCase)
}
function AssertNoReparse([string]$path) {
    $walk = $path
    while ($walk) {
        if (Test-Path -LiteralPath $walk) {
            Assert (-not ((Get-Item -LiteralPath $walk -Force).Attributes -band [IO.FileAttributes]::ReparsePoint)) "Reparse path forbidden: $walk"
        }
        $next = Split-Path -Path $walk -Parent
        if ($next -eq $walk) { break }
        $walk = $next
    }
}
function AssertOutsideVictoryRoot([string]$path) {
    $walk = $path
    while ($walk) {
        Assert (-not (Test-Path -LiteralPath (Join-Path $walk '.victory-root'))) "Path belongs to a .victory-root checkout: $path"
        $next = Split-Path -Path $walk -Parent
        if ($next -eq $walk) { break }
        $walk = $next
    }
}
function SeedInventory([string]$path) {
    $hasher = [Security.Cryptography.IncrementalHash]::CreateHash([Security.Cryptography.HashAlgorithmName]::SHA256)
    $fileCount = 0L; $directoryCount = 0L; $totalBytes = 0L
    try {
        foreach ($item in @(Get-ChildItem -LiteralPath $path -Force -Recurse | Sort-Object FullName)) {
            Assert (-not ($item.Attributes -band [IO.FileAttributes]::ReparsePoint)) "Seed reparse entry forbidden: $($item.FullName)"
            $relative = [IO.Path]::GetRelativePath($path, $item.FullName)
            if ($item.PSIsContainer) { $line = "D`t$relative`n"; $directoryCount++ }
            else {
                $digest = (Get-FileHash -LiteralPath $item.FullName -Algorithm SHA256).Hash
                $line = "F`t$relative`t$($item.Length)`t$digest`n"
                $fileCount++; $totalBytes += $item.Length
            }
            $hasher.AppendData([Text.Encoding]::UTF8.GetBytes($line))
        }
        return [ordered]@{ files = $fileCount; directories = $directoryCount; bytes = $totalBytes; tree_sha256 = [Convert]::ToHexString($hasher.GetHashAndReset()) }
    } finally { $hasher.Dispose() }
}
function WriteJsonNew([string]$path, $value) {
    $json = $value | ConvertTo-Json -Depth 20
    $stream = [IO.File]::Open($path, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write)
    try { $bytes = [Text.UTF8Encoding]::new($false).GetBytes($json); $stream.Write($bytes, 0, $bytes.Length); $stream.Flush($true) } finally { $stream.Dispose() }
}
Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class BenchmarkWindowClose {
    public delegate bool EnumWindowsProc(IntPtr hwnd, IntPtr lparam);
    [DllImport("user32.dll")] public static extern bool EnumWindows(EnumWindowsProc callback, IntPtr lparam);
    [DllImport("user32.dll")] public static extern uint GetWindowThreadProcessId(IntPtr hwnd, out uint processId);
    [DllImport("user32.dll")] public static extern bool IsWindowVisible(IntPtr hwnd);
    [DllImport("user32.dll")] public static extern bool PostMessage(IntPtr hwnd, uint msg, IntPtr wparam, IntPtr lparam);
    public static int CloseVisibleWindows(uint target) {
        int count = 0;
        EnumWindows((hwnd, unused) => {
            uint owner; GetWindowThreadProcessId(hwnd, out owner);
            if (owner == target && IsWindowVisible(hwnd)) {
                if (PostMessage(hwnd, 0x0010, IntPtr.Zero, IntPtr.Zero)) count++;
            }
            return true;
        }, IntPtr.Zero);
        return count;
    }
}
'@
Add-Type -Path (Join-Path $PSScriptRoot 'BenchmarkProcess.cs')
function SendBatch($batch, [string]$mailbox, [long]$anchorTick, [datetime]$anchorUtc, $process, [string]$phase, [Collections.Generic.List[object]]$log) {
    $dueTick = $anchorTick + [long]([double]$batch.at_seconds * [Diagnostics.Stopwatch]::Frequency)
    $dueUtc = $anchorUtc.AddSeconds([double]$batch.at_seconds)
    while ([Diagnostics.Stopwatch]::GetTimestamp() -lt $dueTick) {
        Assert (-not $process.HasExited) 'Process exited before scheduled keys.'
        $remainingMs = ($dueTick - [Diagnostics.Stopwatch]::GetTimestamp()) * 1000.0 / [Diagnostics.Stopwatch]::Frequency
        Start-Sleep -Milliseconds ([math]::Min(50, [math]::Max(1, $remainingMs)))
    }
    Assert (-not (Test-Path -LiteralPath $mailbox)) 'Prior key batch not consumed.'
    $temp = "$mailbox.$([guid]::NewGuid().ToString('N')).tmp"
    try {
        [IO.File]::WriteAllText($temp, ($batch.keys -join "`n") + "`n", [Text.UTF8Encoding]::new($false))
        [IO.File]::Move($temp, $mailbox)
        $sentTick = [Diagnostics.Stopwatch]::GetTimestamp()
        $sent = [datetime]::UtcNow
        Assert (($sentTick - $dueTick) * 1000.0 / [Diagnostics.Stopwatch]::Frequency -le 150) 'Key batch missed scheduled send deadline.'
        $deadlineTick = $sentTick + 5 * [Diagnostics.Stopwatch]::Frequency
        while (Test-Path -LiteralPath $mailbox) {
            Assert (-not $process.HasExited) 'Process exited before key consumption.'
            Assert ([Diagnostics.Stopwatch]::GetTimestamp() -lt $deadlineTick) 'Key batch consumption timed out.'
            Start-Sleep -Milliseconds 50
        }
        $log.Add([ordered]@{ phase = $phase; due_utc = $dueUtc.ToString('o'); sent_utc = $sent.ToString('o'); consumed_observed_utc = ([datetime]::UtcNow).ToString('o'); keys = @($batch.keys) })
    } finally { if (Test-Path -LiteralPath $temp) { Remove-Item -LiteralPath $temp } }
}
$recipePath = FullPath $Recipe
$plan = Get-Content -LiteralPath $recipePath -Raw | ConvertFrom-Json
Assert ($plan.schema_version -eq 1) 'Recipe schema_version must be 1.'
$launchPath = FullPath $plan.launch_environment
$launch = Get-Content -LiteralPath $launchPath -Raw | ConvertFrom-Json
Assert ($launch.schema_version -eq 1) 'Launch environment schema_version must be 1.'
$runtime = FullPath $launch.runtime; $exe = FullPath $launch.exe
$workdir = FullPath $launch.working_directory
$dataSeed = FullPath $plan.data_seed; $cacheSeed = FullPath $plan.cache_seed
$outputRoot = FullPath $plan.output_root $false
foreach ($path in @($dataSeed,$cacheSeed,$outputRoot)) { Assert ($path -ne [IO.Path]::GetPathRoot($path)) 'Seed/output cannot be filesystem root.' }
Assert (-not $dataSeed.Equals($cacheSeed, [StringComparison]::OrdinalIgnoreCase)) 'Data and cache seeds must differ.'
foreach ($path in @($runtime,$exe,$workdir,$dataSeed,$cacheSeed,$outputRoot,$recipePath,$launchPath)) { AssertNoReparse $path }
foreach ($path in @($dataSeed,$cacheSeed,$outputRoot)) { AssertOutsideVictoryRoot $path }
Assert (Test-Path -LiteralPath $dataSeed -PathType Container) 'Data seed must be directory.'
Assert (Test-Path -LiteralPath $cacheSeed -PathType Container) 'Cache seed must be directory.'
Assert ($launch.console -in 'series-x','series-s') 'Invalid console.'
Assert ($outputRoot -notmatch '(?i)(^|\\)\.victory-root(\\|$)' -and $dataSeed -notmatch '(?i)(^|\\)\.victory-root(\\|$)' -and $cacheSeed -notmatch '(?i)(^|\\)\.victory-root(\\|$)') 'Seeds/output cannot be in .victory-root.'
foreach ($a in @($dataSeed,$cacheSeed,$outputRoot)) {
    foreach ($b in @($workdir, (Split-Path -Path $exe -Parent), (Split-Path -Path $runtime -Parent))) {
        Assert (-not (ContainsPath $a $b) -and -not (ContainsPath $b $a)) "Overlapping benchmark path: $a and $b"
    }
}
Assert (-not (ContainsPath $dataSeed $outputRoot) -and -not (ContainsPath $outputRoot $dataSeed)) 'Output overlaps data seed.'
Assert (-not (ContainsPath $cacheSeed $outputRoot) -and -not (ContainsPath $outputRoot $cacheSeed)) 'Output overlaps cache seed.'
Assert (-not (ContainsPath $dataSeed $cacheSeed) -and -not (ContainsPath $cacheSeed $dataSeed)) 'Data and cache seeds overlap.'
Assert (-not (Test-Path -LiteralPath $outputRoot)) 'Output root must be new; existing player data never overwritten.'
Assert ([int]$plan.repeats -ge 1 -and [int]$plan.repeats -le 100) 'repeats must be 1..100.'
foreach ($field in 'ready_timeout_seconds','warmup_seconds','measure_seconds','close_timeout_seconds') {
    Assert ([double]$plan.$field -gt 0 -and [double]$plan.$field -le 3600) "Invalid $field"
}
if ($null -ne $plan.frame_budget_ms) { Assert ([double]$plan.frame_budget_ms -gt 0) 'frame_budget_ms must be positive.' }
foreach ($list in @(@($plan.startup_keys),@($plan.route_keys))) {
    $previousDue = -1.0
    foreach ($batch in $list) {
        $due = [double]$batch.at_seconds
        Assert ([double]::IsFinite($due) -and $due -ge 0 -and $due -gt $previousDue) 'Key schedule times must rise strictly.'
        Assert (@($batch.keys).Count -gt 0) 'Empty key batch.'
        foreach ($line in $batch.keys) { Assert ($line -match '^(?:[A-Za-z0-9]|Enter|Esc|Tab|Back|Up|Down|Left|Right)\s+(?:\d+(?:\.\d+)?|\.\d+)$') "Invalid key line: $line" }
        $previousDue = $due
    }
}
Assert (@($plan.route_keys | Where-Object { $_.at_seconds -ge $plan.measure_seconds }).Count -eq 0) 'Route key after measurement window.'
Assert (@($plan.startup_keys | Where-Object { $_.at_seconds -ge $plan.ready_timeout_seconds }).Count -eq 0) 'Startup key after readiness timeout.'
$inventory = [ordered]@{ data = @(SeedInventory $dataSeed); cache = @(SeedInventory $cacheSeed) }
$gitHead = (& git @('-C',$workdir,'rev-parse','HEAD') 2>$null)
$gitDiff = (& git @('-C',$workdir,'diff','--binary','HEAD') 2>$null | Out-String)
$untrackedFiles = @(& git @('-C',$workdir,'ls-files','--others','--exclude-standard'))
$untrackedIdentity = foreach ($relative in $untrackedFiles) {
    $candidate = [IO.Path]::GetFullPath((Join-Path $workdir $relative))
    if (Test-Path -LiteralPath $candidate -PathType Leaf) { [ordered]@{ path = $relative; sha256 = (Get-FileHash -LiteralPath $candidate -Algorithm SHA256).Hash } }
}
$configPath = Join-Path $workdir 'victory.toml'
$configuredPath = if ($launch.config_path) { FullPath $launch.config_path $false } else { $configPath }
$configuredHash = if (Test-Path -LiteralPath $configuredPath -PathType Leaf) { (Get-FileHash -LiteralPath $configuredPath -Algorithm SHA256).Hash } else { $null }
Assert ($configuredHash -eq $launch.config_sha256) 'Frozen configuration changed since export.'
$gameConfigPath = Join-Path (Split-Path -Path $exe -Parent) 'MicrosoftGame.config'
$identity = [ordered]@{
    recipe_sha256 = (Get-FileHash -LiteralPath $recipePath -Algorithm SHA256).Hash
    launch_sha256 = (Get-FileHash -LiteralPath $launchPath -Algorithm SHA256).Hash
    runtime_sha256 = (Get-FileHash -LiteralPath $runtime -Algorithm SHA256).Hash
    exe_sha256 = (Get-FileHash -LiteralPath $exe -Algorithm SHA256).Hash
    git_head = $gitHead
    git_dirty_diff_sha256 = [Convert]::ToHexString([Security.Cryptography.SHA256]::HashData([Text.Encoding]::UTF8.GetBytes($gitDiff)))
    git_untracked = @($untrackedIdentity)
    config_sha256 = if (Test-Path -LiteralPath $configPath -PathType Leaf) { (Get-FileHash -LiteralPath $configPath -Algorithm SHA256).Hash } else { $null }
    effective_config_path = $configuredPath
    effective_config_sha256 = $configuredHash
    microsoft_game_config_sha256 = if (Test-Path -LiteralPath $gameConfigPath -PathType Leaf) { (Get-FileHash -LiteralPath $gameConfigPath -Algorithm SHA256).Hash } else { $null }
    seeds = $inventory
    gpu = @(Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue | Select-Object Name,DriverVersion,PNPDeviceID)
    cpu = @(Get-CimInstance Win32_Processor -ErrorAction SilentlyContinue | Select-Object Name,NumberOfCores)
    os = [Environment]::OSVersion.VersionString
}
if ($ValidateOnly) { [pscustomobject]@{ valid = $true; output_root = $outputRoot; repeats = $plan.repeats; identity = $identity }; return }
New-Item -ItemType Directory -Path $outputRoot -ErrorAction Stop | Out-Null
for ($runNumber = 1; $runNumber -le [int]$plan.repeats; $runNumber++) {
    $runDir = Join-Path $outputRoot ('run-{0:D3}' -f $runNumber)
    New-Item -ItemType Directory -Path $runDir -ErrorAction Stop | Out-Null
    $manifest = [ordered]@{ schema_version = 1; status = 'incomplete'; identity = $identity; run = $runNumber; command = @($runtime,'run',$exe,'--console',$launch.console,'--execute'); launch_environment = $launch.environment; events = [Collections.Generic.List[object]]::new(); started_utc = ([datetime]::UtcNow).ToString('o') }
    $process = $null; $owned = $null
    try {
        $dataDir = Join-Path $runDir 'data'; $cacheDir = Join-Path $runDir 'cache'
        Copy-Item -LiteralPath $dataSeed -Destination $dataDir -Recurse -ErrorAction Stop
        Copy-Item -LiteralPath $cacheSeed -Destination $cacheDir -Recurse -ErrorAction Stop
        $dataCopyIdentity = SeedInventory $dataDir
        $cacheCopyIdentity = SeedInventory $cacheDir
        Assert ($dataCopyIdentity.tree_sha256 -eq $identity.seeds.data.tree_sha256 -and
            $cacheCopyIdentity.tree_sha256 -eq $identity.seeds.cache.tree_sha256) 'Seed copy differs from frozen preflight identity.'
        $currentConfigHash = if (Test-Path -LiteralPath $configuredPath -PathType Leaf) { (Get-FileHash -LiteralPath $configuredPath -Algorithm SHA256).Hash } else { $null }
        Assert ($currentConfigHash -eq $launch.config_sha256) 'Frozen configuration changed before launch.'
        $mailbox = Join-Path $runDir 'keys.txt'; $csv = Join-Path $runDir 'frames.csv'
        $ready = Join-Path $runDir 'ready.marker'
        $launchEnvironment = [Environment]::GetEnvironmentVariables()
        foreach ($key in @($launchEnvironment.Keys | Where-Object { $_ -like 'VICTORY_*' -or $_ -like '=*' })) { $launchEnvironment.Remove($key) }
        foreach ($property in $launch.environment.PSObject.Properties) {
            Assert ($property.Name -match '^VICTORY_[A-Z0-9_]+$') "Invalid environment key: $($property.Name)"
            $launchEnvironment[$property.Name] = [string]$property.Value
        }
        $launchEnvironment['VICTORY_DATA_DIR'] = $dataDir
        $launchEnvironment['VICTORY_CACHE_DIR'] = $cacheDir
        $launchEnvironment['VICTORY_KEYS_FILE'] = $mailbox
        $launchEnvironment['VICTORY_FRAME_CSV'] = $csv
        $manifest.effective_environment = @{}; foreach ($entry in $launchEnvironment.GetEnumerator()) { if ($entry.Key -like 'VICTORY_*') { $manifest.effective_environment[$entry.Key] = $entry.Value } }
        $owned = [BenchmarkProcess]::Start($runtime, [string[]]@('run',$exe,'--console',$launch.console,'--execute'), $workdir, $launchEnvironment, (Join-Path $runDir 'stdout.log'), (Join-Path $runDir 'stderr.log'))
        $process = $owned.Process
        $manifest.process_id = $process.Id
        $anchorTick = [Diagnostics.Stopwatch]::GetTimestamp(); $anchor = [datetime]::UtcNow
        foreach ($batch in $plan.startup_keys) { SendBatch $batch $mailbox $anchorTick $anchor $process 'startup' $manifest.events }
        $readyDeadlineTick = $anchorTick + [long]([double]$plan.ready_timeout_seconds * [Diagnostics.Stopwatch]::Frequency)
        while (-not (Test-Path -LiteralPath $ready)) {
            Assert (-not $process.HasExited) 'Process exited before ready marker.'
            Assert ([Diagnostics.Stopwatch]::GetTimestamp() -lt $readyDeadlineTick) 'Ready marker timed out.'
            Start-Sleep -Milliseconds 50
        }
        $manifest.ready_observed_utc = ([datetime]::UtcNow).ToString('o')
        $warmUntilTick = [Diagnostics.Stopwatch]::GetTimestamp() + [long]([double]$plan.warmup_seconds * [Diagnostics.Stopwatch]::Frequency)
        while ([Diagnostics.Stopwatch]::GetTimestamp() -lt $warmUntilTick) { Assert (-not $process.HasExited) 'Process exited during warmup.'; Start-Sleep -Milliseconds 50 }
        $windowStartTick = [Diagnostics.Stopwatch]::GetTimestamp(); $windowStart = [datetime]::UtcNow; $manifest.window_start_utc = $windowStart.ToString('o')
        $windowEndTick = $windowStartTick + [long]([double]$plan.measure_seconds * [Diagnostics.Stopwatch]::Frequency)
        $windowEnd = $windowStart.AddSeconds([double]$plan.measure_seconds)
        foreach ($batch in $plan.route_keys) {
            SendBatch $batch $mailbox $windowStartTick $windowStart $process 'route' $manifest.events
            Assert ([Diagnostics.Stopwatch]::GetTimestamp() -le $windowEndTick) 'Route key consumption exceeded measurement window.'
        }
        while ([Diagnostics.Stopwatch]::GetTimestamp() -lt $windowEndTick) { Assert (-not $process.HasExited) 'Process exited during measurement.'; Start-Sleep -Milliseconds 50 }
        $manifest.window_end_utc = $windowEnd.ToString('o')
        $tailDeadlineTick = [Diagnostics.Stopwatch]::GetTimestamp() + [long]([double]$plan.close_timeout_seconds * [Diagnostics.Stopwatch]::Frequency)
        $tailCovered = $false
        $endUs = UnixUs $windowEnd
        while ([Diagnostics.Stopwatch]::GetTimestamp() -lt $tailDeadlineTick) {
            Assert (-not $process.HasExited) 'Runtime exited before post-window coverage frame.'
            if (Test-Path -LiteralPath $csv -PathType Leaf) {
                $tailLine = Get-Content -LiteralPath $csv -Tail 1 -ErrorAction SilentlyContinue
                $tailCells = [string]$tailLine -split ','
                if ($tailCells.Count -eq 21 -and $tailCells[1] -eq [string]$process.Id -and $tailCells[4] -match '^\d+$' -and [int64]$tailCells[4] -ge $endUs) {
                    $tailCovered = $true
                    $manifest.tail_coverage_observed_utc = ([datetime]::UtcNow).ToString('o')
                    $manifest.tail_frame_index = $tailCells[3]
                    break
                }
            }
            Start-Sleep -Milliseconds 100
        }
        Assert $tailCovered 'Post-window capture coverage timed out.'
        $process.Refresh()
        $manifest.close_windows_signaled = [BenchmarkWindowClose]::CloseVisibleWindows([uint32]$process.Id)
        if ($manifest.close_windows_signaled -eq 0) { [void]$process.CloseMainWindow() }
        if (-not $process.WaitForExit([int]([double]$plan.close_timeout_seconds * 1000))) { throw 'Graceful close timed out.' }
        $jobSettleStartTick = [Diagnostics.Stopwatch]::GetTimestamp()
        $jobSettleDeadlineTick = $jobSettleStartTick + [long]([math]::Min(1.0, [double]$plan.close_timeout_seconds) * [Diagnostics.Stopwatch]::Frequency)
        while ($owned.ActiveProcesses -gt 0 -and [Diagnostics.Stopwatch]::GetTimestamp() -lt $jobSettleDeadlineTick) { Start-Sleep -Milliseconds 10 }
        $manifest.job_settle_ms = ([Diagnostics.Stopwatch]::GetTimestamp() - $jobSettleStartTick) * 1000.0 / [Diagnostics.Stopwatch]::Frequency
        Assert ($owned.ActiveProcesses -eq 0) 'Owned relaunch or helper process remains after root exit.'
        $manifest.exit_code = $process.ExitCode
        Assert ($process.ExitCode -eq 0) 'Runtime exited with error.'
        $summaryPath = Join-Path $runDir 'summary.json'
        & (Join-Path $PSScriptRoot 'measure-frame-capture.ps1') -Csv $csv -WindowStartUtc $windowStart -WindowEndUtc $windowEnd -FrameBudgetMs $(if ($plan.frame_budget_ms) { [double]$plan.frame_budget_ms } else { 33.333333 }) -OutputJson $summaryPath
        $summary = Get-Content -LiteralPath $summaryPath -Raw | ConvertFrom-Json
        Assert ($summary.process_id -eq $process.Id) 'Capture PID differs from owned runtime PID.'
        $manifest.status = 'complete'
    } catch {
        $manifest.failure = $_.Exception.Message
        if ($owned -and $owned.ActiveProcesses -gt 0) {
            $manifest.shutdown = 'forced_after_failure'
        }
    } finally {
        if ($owned) { $owned.Dispose() }
        $manifest.ended_utc = ([datetime]::UtcNow).ToString('o')
        if (-not $manifest.Contains('shutdown')) { $manifest.shutdown = 'graceful' }
        WriteJsonNew (Join-Path $runDir 'manifest.json') $manifest
    }
    if ($manifest.status -ne 'complete') { throw "Run $runNumber invalid: $($manifest.failure). Evidence retained in $runDir" }
}
