$ErrorActionPreference = 'Stop'
$scratchRoot = [IO.Path]::GetFullPath([IO.Path]::GetTempPath()).TrimEnd('\')
$scratch = Join-Path $scratchRoot ('vice-benchmark-process-' + [guid]::NewGuid().ToString('N'))
$owner = $null
$grandchild = $null
New-Item -ItemType Directory -Path $scratch | Out-Null
try {
    Add-Type -Path (Join-Path $PSScriptRoot 'BenchmarkProcess.cs') -ErrorAction Stop
    $childScript = Join-Path $scratch 'child script.ps1'
    @'
$ErrorActionPreference = 'Stop'
$payload = [ordered]@{ arguments = @($args); special = $env:BENCHMARK_TEST_VALUE }
$payload | ConvertTo-Json -Compress -Depth 4 | Set-Content -LiteralPath $env:BENCHMARK_TEST_ARGS -Encoding utf8
$sleepScript = Join-Path $env:BENCHMARK_TEST_ROOT 'sleep child.ps1'
$spawned = Start-Process -FilePath (Get-Process -Id $PID).Path -ArgumentList @('-NoProfile','-File',('"' + $sleepScript + '"')) -PassThru -WindowStyle Hidden
[string]$spawned.Id | Set-Content -LiteralPath $env:BENCHMARK_TEST_PID
Write-Output 'root stdout'
[Console]::Error.WriteLine('root stderr')
'@ | Set-Content -LiteralPath $childScript -Encoding utf8
    $sleepScript = Join-Path $scratch 'sleep child.ps1'
    'Start-Sleep -Seconds 60' | Set-Content -LiteralPath $sleepScript -Encoding utf8
    $argsPath = Join-Path $scratch 'arguments.json'
    $pidPath = Join-Path $scratch 'grandchild.pid'
    $stdoutPath = Join-Path $scratch 'stdout.log'
    $stderrPath = Join-Path $scratch 'stderr.log'
    $environment = [Collections.Hashtable]::new([StringComparer]::OrdinalIgnoreCase)
    foreach ($name in [Environment]::GetEnvironmentVariables().Keys) {
        if (-not ([string]$name).StartsWith('=')) { $environment[[string]$name] = [string][Environment]::GetEnvironmentVariable([string]$name) }
    }
    $environment['BENCHMARK_TEST_ROOT'] = $scratch
    $environment['BENCHMARK_TEST_ARGS'] = $argsPath
    $environment['BENCHMARK_TEST_PID'] = $pidPath
    $environment['BENCHMARK_TEST_VALUE'] = 'value with spaces "quotes" \ slash'
    $expectedArguments = @('space value', 'a"b', 'ends-with-slash\', 'double\\slash', '')
    $pwshPath = (Get-Process -Id $PID).Path
    $owner = [BenchmarkProcess]::Start($pwshPath, (@('-NoProfile','-File',$childScript) + $expectedArguments),
        $scratch, $environment, $stdoutPath, $stderrPath)
    if (-not $owner.Process.WaitForExit(10000)) { throw 'Root process did not exit.' }
    $deadline = [Diagnostics.Stopwatch]::StartNew()
    while (-not (Test-Path -LiteralPath $pidPath) -and $deadline.Elapsed.TotalSeconds -lt 5) { Start-Sleep -Milliseconds 50 }
    if (-not (Test-Path -LiteralPath $pidPath)) { throw 'Grandchild PID not written.' }
    $grandchildId = [int](Get-Content -LiteralPath $pidPath -Raw)
    $grandchild = [Diagnostics.Process]::GetProcessById($grandchildId)
    if ($owner.ActiveProcesses -lt 1) { throw 'Job lost grandchild after root exit.' }
    $observed = Get-Content -LiteralPath $argsPath -Raw | ConvertFrom-Json
    if (@($observed.arguments).Count -ne $expectedArguments.Count) { throw 'Argument count changed.' }
    for ($index = 0; $index -lt $expectedArguments.Count; $index++) {
        if ($observed.arguments[$index] -cne $expectedArguments[$index]) { throw "Argument $index changed." }
    }
    if ($observed.special -cne $environment['BENCHMARK_TEST_VALUE']) { throw 'Environment changed.' }
    if ((Get-Content -LiteralPath $stdoutPath -Raw) -notmatch 'root stdout') { throw 'stdout missing.' }
    if ((Get-Content -LiteralPath $stderrPath -Raw) -notmatch 'root stderr') { throw 'stderr missing.' }
    $owner.Dispose(); $owner = $null
    if (-not $grandchild.WaitForExit(5000)) { throw 'Job disposal did not kill grandchild.' }
    [IO.File]::WriteAllText($stdoutPath, 'sentinel')
    $rejected = $false
    try {
        $unexpected = [BenchmarkProcess]::Start($pwshPath, @('-NoProfile','-File',$childScript),
            $scratch, $environment, $stdoutPath, (Join-Path $scratch 'second-stderr.log'))
        $unexpected.Dispose()
    } catch [ComponentModel.Win32Exception] { $rejected = $true }
    if (-not $rejected -or [IO.File]::ReadAllText($stdoutPath) -cne 'sentinel') { throw 'Existing log was overwritten or accepted.' }
    'PASS: argv, Unicode environment, job descendant ownership, kill on dispose, logs, CreateNew'
} finally {
    if ($null -ne $owner) { $owner.Dispose() }
    if ($null -ne $grandchild) { $grandchild.Dispose() }
    $resolvedScratch = [IO.Path]::GetFullPath($scratch)
    $resolvedParent = [IO.Path]::GetDirectoryName($resolvedScratch).TrimEnd('\')
    $scratchName = [IO.Path]::GetFileName($resolvedScratch)
    if (-not [string]::Equals($resolvedParent, $scratchRoot, [StringComparison]::OrdinalIgnoreCase) -or
        -not $scratchName.StartsWith('vice-benchmark-process-', [StringComparison]::OrdinalIgnoreCase)) {
        throw 'Unsafe scratch path; cleanup skipped.'
    }
    Remove-Item -LiteralPath $resolvedScratch -Recurse -Force -ErrorAction SilentlyContinue
}
