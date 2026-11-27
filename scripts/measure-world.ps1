# Called by play.ps1 after the hardware preset is applied.
param([string]$Victory, [string]$Exe, [string]$Console, [int]$Seconds, [string]$Ffmpeg)
$ErrorActionPreference = 'Stop'
$run = New-Item -ItemType Directory -Path (Join-Path $env:TEMP ("victory-world-" + [guid]::NewGuid().ToString('N')))
$log = Join-Path $run.FullName 'frames.log'
$start = [Diagnostics.ProcessStartInfo]::new()
$start.FileName = $Victory
$start.Arguments = 'run "' + $Exe + '" --console ' + $Console + ' --execute'
$start.WorkingDirectory = (Get-Location).Path
$start.UseShellExecute = $false
$start.EnvironmentVariables['VICTORY_LOG_FILE'] = $log
$start.EnvironmentVariables['VICTORY_LOG'] = 'warn,victory_gpu::host=info,victory_hle::modules::fidelityfx=info'
if ($env:VICTORY_LOG) {
    $start.EnvironmentVariables['VICTORY_LOG'] += ",$env:VICTORY_LOG"
}
# The input clock starts when the title begins polling GameInput.
$start.EnvironmentVariables['VICTORY_KEYS'] = '3-3.4:Enter,8-8.4:Enter'
$keys = Join-Path $run.FullName 'input'
$start.EnvironmentVariables['VICTORY_KEYS_FILE'] = $keys
Write-Host "Benchmark log $log"
$game = [Diagnostics.Process]::Start($start)
$timer = [Diagnostics.Stopwatch]::StartNew()
$reader = $null
$worldStart = $null
$samples = @()
$consecutive = 0
$nextPromptRetry = 20
$lastSample = 0
$recorder = $null
try {
    while (!$game.HasExited) {
        if ($Ffmpeg -and !$recorder) {
            $game.Refresh()
            if ($game.MainWindowHandle -ne 0) {
                $capture = [Diagnostics.ProcessStartInfo]::new()
                $capture.FileName = $Ffmpeg
                $capture.WorkingDirectory = $run.FullName
                $capture.UseShellExecute = $false
                $capture.RedirectStandardInput = $true
                $handle = $game.MainWindowHandle.ToInt64()
                $capture.Arguments = "-hide_banner -loglevel error -f lavfi -i gfxcapture=hwnd=${handle}:capture_cursor=0:max_framerate=8,hwdownload,format=bgra -vf scale=960:-2 -c:v libx264 -preset ultrafast -pix_fmt yuv420p -fps_mode vfr -flush_packets 1 -cluster_time_limit 1000 window.mkv"
                $recorder = [Diagnostics.Process]::Start($capture)
                Write-Host "Window recording $($run.FullName)\window.mkv"
            }
        }
        if ($recorder -and $recorder.HasExited) { throw "Window recording stopped early (exit code $($recorder.ExitCode))." }
        if (!$reader -and (Test-Path -LiteralPath $log)) {
            $reader = [IO.StreamReader]::new([IO.File]::Open($log, 'Open', 'Read', 'ReadWrite'))
        }
        while ($reader -and ($line = $reader.ReadLine()) -ne $null) {
            if ($line -match ' ([\d.]+) fps, ([\d.]+) ms/frame .*?, (\d+) draws/frame') {
                $fps = [double]::Parse($Matches[1], [cultureinfo]::InvariantCulture)
                $draws = [int]$Matches[3]
                if ($draws -lt 1000) {
                    $consecutive = 0
                    # The initial account prompt can appear after shader warm-up.
                    if ($null -eq $worldStart -and $draws -eq 11 -and $timer.Elapsed.TotalSeconds -ge $nextPromptRetry) {
                        if (!(Test-Path -LiteralPath $keys)) {
                            [IO.File]::WriteAllText("$keys.tmp", "Enter 0.4")
                            [IO.File]::Move("$keys.tmp", $keys)
                        }
                        $nextPromptRetry = $timer.Elapsed.TotalSeconds + 12
                    }
                    continue
                }
                $consecutive++
                if ($consecutive -ge 2 -and $null -eq $worldStart) {
                    $worldStart = $timer.Elapsed.TotalSeconds
                    Write-Host 'World rendering detected. Measurement started.'
                }
                if ($null -ne $worldStart) {
                    $samples += $fps
                    $lastSample = $timer.Elapsed.TotalSeconds
                }
            }
        }
        if ($null -ne $worldStart -and $timer.Elapsed.TotalSeconds - $lastSample -gt 30) {
            throw 'World-frame reporting stopped before the measurement finished.'
        }
        if ($null -ne $worldStart -and $samples.Count -ge 2 -and $lastSample - $worldStart -ge $Seconds) {
            if ($Ffmpeg -and !$recorder) { throw 'No game window was available to record.' }
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
    try {
        if ($recorder -and !$recorder.HasExited) {
            $recorder.StandardInput.WriteLine('q')
            if (!$recorder.WaitForExit(5000)) { $recorder.Kill(); $recorder.WaitForExit(); throw 'Window recording did not finalize.' }
            if ($recorder.ExitCode -ne 0) { throw "Window recording failed (exit code $($recorder.ExitCode))." }
        }
    } finally {
        if ($recorder) { $recorder.Dispose() }
        if (!$game.HasExited) {
            [void]$game.CloseMainWindow()
            if (!$game.WaitForExit(4000)) { $game.Kill(); $game.WaitForExit() }
        }
        $game.Dispose()
    }
}
