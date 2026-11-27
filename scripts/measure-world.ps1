# Called by play.ps1 after the hardware preset is applied.
param([string]$Victory, [string]$Exe, [string]$Console, [int]$Seconds)
$run = New-Item -ItemType Directory -Path (Join-Path $env:TEMP ("victory-world-" + [guid]::NewGuid().ToString('N')))
$log = Join-Path $run.FullName 'frames.log'
$start = [Diagnostics.ProcessStartInfo]::new()
$start.FileName = $Victory
$start.Arguments = 'run "' + $Exe + '" --console ' + $Console + ' --execute'
$start.WorkingDirectory = (Get-Location).Path
$start.UseShellExecute = $false
$start.EnvironmentVariables['VICTORY_LOG_FILE'] = $log
$start.EnvironmentVariables['VICTORY_LOG'] = 'warn,victory_gpu::host=info,victory_hle::modules::fidelityfx=info,victory_gpu::draw::rtv_trace=info'
# The input clock starts when the title begins polling GameInput.
$start.EnvironmentVariables['VICTORY_KEYS'] = '3-3.4:Enter,8-8.4:Enter'
Write-Host "Benchmark log $log"
$game = [Diagnostics.Process]::Start($start)
$timer = [Diagnostics.Stopwatch]::StartNew()
$reader = $null
$worldStart = $null
$samples = @()
$consecutive = 0
try {
    while (!$game.HasExited) {
        if (!$reader -and (Test-Path -LiteralPath $log)) {
            $reader = [IO.StreamReader]::new([IO.File]::Open($log, 'Open', 'Read', 'ReadWrite'))
        }
        while ($reader -and ($line = $reader.ReadLine()) -ne $null) {
            if ($line -match ' ([\d.]+) fps, ([\d.]+) ms/frame .*?, (\d+) draws/frame') {
                $fps = [double]::Parse($Matches[1], [cultureinfo]::InvariantCulture)
                if ([int]$Matches[3] -lt 1000) { $consecutive = 0; continue }
                $consecutive++
                if ($consecutive -ge 2 -and $null -eq $worldStart) {
                    $worldStart = $timer.Elapsed.TotalSeconds
                    Write-Host 'World rendering detected. Measurement started.'
                }
                if ($null -ne $worldStart) { $samples += $fps }
            }
        }
        if ($null -ne $worldStart -and $timer.Elapsed.TotalSeconds - $worldStart -ge $Seconds) {
            if ($samples.Count -lt 2) { throw 'Insufficient world-frame samples.' }
            $ordered = @($samples | Sort-Object)
            $middle = $ordered.Count -shr 1
            $median = ($ordered[$middle] + $ordered[($ordered.Count - 1) -shr 1]) / 2
            Write-Host "Median world FPS $median ($($ordered.Count) reporting intervals)."
            return
        }
        if ($null -eq $worldStart -and $timer.Elapsed.TotalMinutes -ge 5) { throw 'Timed out waiting for world rendering.' }
        Start-Sleep -Milliseconds 250
    }
    throw "Game exited before benchmark completion (exit code $($game.ExitCode))."
} finally {
    if ($reader) { $reader.Dispose() }
    if (!$game.HasExited) {
        [void]$game.CloseMainWindow()
        if (!$game.WaitForExit(4000)) { $game.Kill(); $game.WaitForExit() }
    }
    $game.Dispose()
}
