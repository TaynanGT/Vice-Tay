$ErrorActionPreference = 'Stop'
$scratch = Join-Path ([IO.Path]::GetTempPath()) ("vice-tay-frame-tests-" + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $scratch | Out-Null
$summarizer = Join-Path $PSScriptRoot 'measure-frame-capture.ps1'
$runner = Join-Path $PSScriptRoot 'benchmark.ps1'
$start = [DateTimeOffset]::Parse('2024-01-01T00:00:00Z')
$baseUs = $start.ToUnixTimeMilliseconds() * 1000
$columns = 'schema_version','process_id','stream_id','frame_index','timestamp_unix_us','elapsed_ns','interval_valid','frame_ms','draws','dispatches','prepare_ms','gpu_wait_ms','readback_ms','readback_bytes','pipelines_pending','shader_cache_misses','textures_uploaded','texture_upload_bytes','generated_frames_total','swapchain_present_count','frame_generation'
function MakeCapture([string]$name, [int[]]$times, [int]$drop = 0, [bool]$marker = $true, [int]$badIndex = -1, [bool]$fg = $false, [bool]$interleaved = $false) {
    $csv = Join-Path $scratch "$name.csv"
    $lines = [Collections.Generic.List[string]]::new()
    $lines.Add(($columns -join ','))
    for ($i = 0; $i -lt $times.Count; $i++) {
        $interval = if ($interleaved) { $i -ge 2 } else { $i -ge 1 }
        $ms = if ($interval) { [string]($times[$i] - $times[$i - $(if ($interleaved) { 2 } else { 1 })]) } else { '' }
        $index = if ($i -eq $badIndex) { $i + 2 } else { $i + 1 }
        $streamId = if ($interleaved) { [string](1 + ($i % 2)) } else { '1' }
        $values = @('1','123',$streamId,[string]$index,[string]($baseUs + $times[$i] * 1000),[string]($times[$i] * 1000000),$(if ($interval) { '1' } else { '0' }),$ms,$(if ($interval) { '1' } else { '' }),$(if ($interval) { '0' } else { '' }),$(if ($interval) { '0' } else { '' }),$(if ($interval) { '0' } else { '' }),$(if ($interval) { '0' } else { '' }),$(if ($interval) { '0' } else { '' }),'0','0','0','0',$(if ($fg) { [string]$i } else { '0' }),$(if ($fg) { [string]($i * 2) } else { '' }),$(if ($fg) { '1' } else { '0' }))
        $lines.Add(($values -join ','))
    }
    [IO.File]::WriteAllText($csv, ($lines -join "`n") + "`n", [Text.UTF8Encoding]::new($false))
    if ($marker) {
        @{schema_version=1;status='complete';process_id=123;written_frames=$times.Count;dropped_frames=$drop;csv_bytes=(Get-Item -LiteralPath $csv).Length} |
            ConvertTo-Json | Set-Content -LiteralPath "$csv.complete.json" -Encoding utf8NoBOM
    }
    return $csv
}
function ExpectFailure([scriptblock]$action, [string]$name) {
    $failed = $false
    try { & $action | Out-Null } catch { $failed = $true }
    if (-not $failed) { throw "Expected failure: $name" }
}
$windowStart = $start.UtcDateTime.AddMilliseconds(5)
$windowEnd = $start.UtcDateTime.AddMilliseconds(115)
$csv = MakeCapture 'valid' @(0,10,30,60,100,150)
$result = & $summarizer -Csv $csv -WindowStartUtc $windowStart -WindowEndUtc $windowEnd -FrameBudgetMs 33 | ConvertFrom-Json
if ($result.measured_intervals -ne 3 -or $result.mean_frame_ms -ne 30 -or $result.median_frame_ms -ne 30 -or
    $result.p95_frame_ms -ne 40 -or $result.p99_frame_ms -ne 40 -or $result.one_percent_low_fps -ne 25 -or
    $result.over_budget -ne 1 -or [math]::Abs($result.rendered_fps - 33.3333333333333) -gt 0.0001) { throw 'Known statistics or boundary exclusion failed.' }
ExpectFailure { & $summarizer -Csv (MakeCapture 'missing' @(0,10,30,60,100,150) -marker $false) -WindowStartUtc $windowStart -WindowEndUtc $windowEnd } 'missing completion'
ExpectFailure { & $summarizer -Csv (MakeCapture 'dropped' @(0,10,30,60,100,150) -drop 1) -WindowStartUtc $windowStart -WindowEndUtc $windowEnd } 'dropped frames'
$truncatedCsv = MakeCapture 'truncated' @(0,10,30,60,100,150)
$truncatedText = [IO.File]::ReadAllText($truncatedCsv)
[IO.File]::WriteAllText($truncatedCsv, $truncatedText.Substring(0, $truncatedText.Length - 10))
ExpectFailure { & $summarizer -Csv $truncatedCsv -WindowStartUtc $windowStart -WindowEndUtc $windowEnd } 'truncated CSV'
ExpectFailure { & $summarizer -Csv (MakeCapture 'index' @(0,10,30,60,100,150) -badIndex 3) -WindowStartUtc $windowStart -WindowEndUtc $windowEnd } 'missing index'
ExpectFailure { & $summarizer -Csv (MakeCapture 'short' @(0,10,30,60,100) ) -WindowStartUtc $windowStart -WindowEndUtc $windowEnd } 'window coverage'
$fgResult = & $summarizer -Csv (MakeCapture 'fg' @(0,10,30,60,100,150) -fg $true) -WindowStartUtc $windowStart -WindowEndUtc $windowEnd | ConvertFrom-Json
if ($fgResult.generated_frame_delta -ne 3 -or $fgResult.swapchain_present_delta -ne 6) { throw 'FG deltas missed first selected interval.' }
ExpectFailure { & $summarizer -Csv (MakeCapture 'interleaved' @(0,10,20,30,40,50,60,70,80,90,100,110,150,160) -interleaved $true) -WindowStartUtc $windowStart -WindowEndUtc $windowEnd } 'overlapping streams'
$seed = Join-Path $scratch 'seed'; New-Item -ItemType Directory -Path $seed | Out-Null
$binary = (Get-Command pwsh).Source
$launchPath = Join-Path $scratch 'launch.json'
@{schema_version=1;runtime=$binary;exe=$binary;console='series-x';working_directory=(Resolve-Path (Join-Path $PSScriptRoot '..')).Path;config_path=(Join-Path $scratch 'absent.toml');config_sha256=$null;environment=@{}} |
    ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $launchPath -Encoding utf8NoBOM
$recipePath = Join-Path $scratch 'recipe.json'
@{schema_version=1;launch_environment=$launchPath;data_seed=$seed;cache_seed=$seed;output_root=(Join-Path $scratch 'output');repeats=1;ready_timeout_seconds=1;warmup_seconds=1;measure_seconds=1;close_timeout_seconds=1;startup_keys=@();route_keys=@()} |
    ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $recipePath -Encoding utf8NoBOM
ExpectFailure { & $runner -Recipe $recipePath -ValidateOnly } 'overlapping seeds'
$otherSeed = Join-Path $scratch 'other-seed'; New-Item -ItemType Directory -Path $otherSeed | Out-Null
$validRecipe = Get-Content -LiteralPath $recipePath -Raw | ConvertFrom-Json
$validRecipe.cache_seed = $otherSeed
$validRecipe.output_root = Join-Path $seed 'unsafe-results'
$validRecipe | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $recipePath -Encoding utf8NoBOM
ExpectFailure { & $runner -Recipe $recipePath -ValidateOnly } 'output under seed'
$validRecipe.output_root = Join-Path $scratch 'output'
$validRecipe | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $recipePath -Encoding utf8NoBOM
$validated = & $runner -Recipe $recipePath -ValidateOnly
if (-not $validated.valid -or (Test-Path -LiteralPath $validRecipe.output_root)) { throw 'Valid preflight failed or created output.' }
Write-Output "Synthetic benchmark tests passed. Evidence: $scratch"
