param(
    [Parameter(Mandatory)][string]$Csv,
    [Parameter(Mandatory)][datetime]$WindowStartUtc,
    [Parameter(Mandatory)][datetime]$WindowEndUtc,
    [double]$FrameBudgetMs = 33.333333,
    [string]$OutputJson
)
$ErrorActionPreference = 'Stop'
$culture = [Globalization.CultureInfo]::InvariantCulture
function Assert($ok, $why) { if (-not $ok) { throw $why } }
function UnixUs([datetime]$value) {
    $utc = $value.ToUniversalTime()
    $offset = [DateTimeOffset]$utc
    return [int64]($offset.ToUnixTimeMilliseconds() * 1000 + [math]::Floor(($utc.Ticks % 10000) / 10))
}
function Percentile([double[]]$values, [double]$fraction) {
    $values = @($values | Sort-Object)
    $index = [math]::Max(0, [int][math]::Ceiling($fraction * $values.Count) - 1)
    return $values[$index]
}
Assert ($WindowEndUtc.ToUniversalTime() -gt $WindowStartUtc.ToUniversalTime()) 'Invalid measurement window.'
Assert ($FrameBudgetMs -gt 0) 'Frame budget must be positive.'
Assert (Test-Path -LiteralPath $Csv -PathType Leaf) 'Capture CSV missing.'
$completePath = "$Csv.complete.json"
Assert (Test-Path -LiteralPath $completePath -PathType Leaf) 'Completion marker missing.'
$complete = Get-Content -LiteralPath $completePath -Raw | ConvertFrom-Json
Assert ($complete.schema_version -eq 1 -and $complete.status -eq 'complete') 'Invalid completion marker.'
foreach ($field in 'process_id','written_frames','dropped_frames','csv_bytes') {
    Assert ([string]$complete.$field -match '^\d+$') "Invalid completion $field."
}
Assert ([uint64]$complete.process_id -gt 0) 'Invalid completion process_id.'
Assert ($complete.dropped_frames -eq 0) 'Capture dropped frames.'
$bytes = (Get-Item -LiteralPath $Csv).Length
Assert ($complete.csv_bytes -eq $bytes) 'CSV byte count differs from completion marker.'
$expected = 'schema_version','process_id','stream_id','frame_index','timestamp_unix_us','elapsed_ns','interval_valid','frame_ms','draws','dispatches','prepare_ms','gpu_wait_ms','readback_ms','readback_bytes','pipelines_pending','shader_cache_misses','textures_uploaded','texture_upload_bytes','generated_frames_total','swapchain_present_count','frame_generation'
$rawLines = [IO.File]::ReadAllLines($Csv)
Assert ($rawLines.Length -ge 2) 'CSV has no sample rows.'
$header = $rawLines[0]
Assert ($header -ceq ($expected -join ',')) 'Unexpected CSV schema.'
foreach ($line in $rawLines | Select-Object -Skip 1) { Assert ($line.Split(',').Length -eq $expected.Count -and $line -notmatch '"') 'Malformed CSV column count or quoting.' }
$rows = @(Import-Csv -LiteralPath $Csv)
Assert ($rows.Count -eq [int64]$complete.written_frames) 'CSV row count differs from completion marker.'
Assert ($rows.Count -gt 0) 'Capture empty.'
$windowStart = UnixUs $WindowStartUtc
$windowEnd = UnixUs $WindowEndUtc
$previous = @{}
$coverage = @{}
$lastIndex = 0L
$lastElapsed = -1L
$lastTimestamp = -1L
$originUs = $null
$selected = [Collections.Generic.List[double]]::new()
$fgFirstPrior = $null; $fgLast = $null; $fgPresentComplete = $true
foreach ($row in $rows) {
    foreach ($field in 'schema_version','process_id','stream_id','frame_index','timestamp_unix_us','elapsed_ns','interval_valid','draws','dispatches','readback_bytes','pipelines_pending','shader_cache_misses','textures_uploaded','texture_upload_bytes','generated_frames_total','frame_generation') {
        if ($row.interval_valid -eq '0' -and $field -in 'draws','dispatches','readback_bytes') { Assert ($row.$field -eq '') "Baseline $field must be empty."; continue }
        Assert ($row.$field -match '^\d+$') "Invalid numeric $field."
        $parsedInteger = [uint64]0
        Assert ([uint64]::TryParse($row.$field, [ref]$parsedInteger)) "Out-of-range numeric $field."
    }
    foreach ($field in 'prepare_ms','gpu_wait_ms','readback_ms') {
        if ($row.interval_valid -eq '0') { Assert ($row.$field -eq '') "Baseline $field must be empty."; continue }
        Assert ($row.$field -match '^(?:\d+(?:\.\d*)?|\.\d+)$' -and [double]::IsFinite([double]::Parse($row.$field, $culture))) "Invalid numeric $field."
    }
    Assert ($row.frame_generation -in '0','1') 'Invalid frame generation flag.'
    Assert ($row.swapchain_present_count -eq '' -or $row.swapchain_present_count -match '^\d+$') 'Invalid swapchain present count.'
    Assert ([uint64]$row.stream_id -gt 0 -and [uint64]$row.process_id -gt 0) 'Invalid PID or stream ID.'
    Assert ($row.schema_version -eq '1') 'Row schema version mismatch.'
    Assert ([int64]$row.process_id -eq [int64]$complete.process_id) 'Row PID mismatch.'
    $index = [int64]$row.frame_index
    $elapsed = [int64]$row.elapsed_ns
    $timestamp = [int64]$row.timestamp_unix_us
    Assert ($index -eq ($lastIndex + 1)) 'Missing or reordered frame index.'
    Assert ($elapsed -gt $lastElapsed -and $timestamp -gt $lastTimestamp) 'Nonmonotonic capture time.'
    $thisOrigin = $timestamp - [int64][math]::Floor($elapsed / 1000.0)
    if ($null -eq $originUs) { $originUs = $thisOrigin } else { Assert ($thisOrigin -eq $originUs) 'UTC and elapsed clocks disagree.' }
    Assert ($row.interval_valid -in '0','1') 'Invalid interval flag.'
    Assert (-not [string]::IsNullOrWhiteSpace($row.stream_id)) 'Empty stream ID.'
    if (-not $coverage.ContainsKey($row.stream_id)) { $coverage[$row.stream_id] = @{ first = $timestamp; last = $timestamp; selected = 0 } }
    $coverage[$row.stream_id].last = $timestamp
    if (-not $previous.ContainsKey($row.stream_id)) {
        Assert ($row.interval_valid -eq '0') 'First stream row must be baseline.'
        Assert ($row.frame_ms -eq '') 'Baseline frame_ms must be empty.'
    } else {
        Assert ($row.interval_valid -eq '1') 'Stream baseline repeated.'
        $prior = $previous[$row.stream_id]
        $frameMs = [double]::Parse($row.frame_ms, $culture)
        Assert ($row.frame_ms -match '^(?:\d+(?:\.\d*)?|\.\d+)$') 'Invalid frame_ms format.'
        Assert ([double]::IsFinite($frameMs) -and $frameMs -gt 0) 'Invalid frame_ms.'
        $elapsedMs = ($elapsed - [int64]$prior.elapsed_ns) / 1000000.0
        Assert ([math]::Abs($frameMs - $elapsedMs) -le 0.002) 'Frame interval disagrees with elapsed clock.'
        Assert ([uint64]$row.generated_frames_total -ge [uint64]$prior.generated_frames_total) 'Generated counter reset.'
        if ($row.swapchain_present_count -ne '' -and $prior.swapchain_present_count -ne '') {
            Assert ([uint64]$row.swapchain_present_count -ge [uint64]$prior.swapchain_present_count) 'Present counter reset.'
        }
        if ([int64]$prior.timestamp_unix_us -ge $windowStart -and $timestamp -le $windowEnd) {
            $selected.Add($frameMs)
            $coverage[$row.stream_id].selected++
            if ($row.frame_generation -eq '1') {
                if ($null -eq $fgFirstPrior) { $fgFirstPrior = $prior }
                $fgLast = $row
                if ($prior.swapchain_present_count -eq '' -or $row.swapchain_present_count -eq '') { $fgPresentComplete = $false }
            }
        }
    }
    $previous[$row.stream_id] = $row
    $lastIndex = $index; $lastElapsed = $elapsed; $lastTimestamp = $timestamp
}
Assert ($lastTimestamp -ge $windowEnd) 'Capture ends before window end.'
Assert ($rows[0].timestamp_unix_us -le $windowStart) 'Capture starts after window start.'
foreach ($entry in $coverage.Values) {
    if ($entry.first -le $windowEnd -and $entry.last -ge $windowStart) {
        Assert ($entry.first -le $windowStart -and $entry.last -ge $windowEnd -and $entry.selected -gt 0) 'Stream lacks baseline or coverage across measurement window.'
    }
}
Assert ($selected.Count -gt 0) 'No complete frame intervals inside window.'
$activeStreams = @($coverage.GetEnumerator() | Where-Object { $_.Value.first -le $windowEnd -and $_.Value.last -ge $windowStart })
Assert ($activeStreams.Count -eq 1) 'Multiple active streams overlap measurement window; rendered FPS cannot combine them.'
$sorted = @($selected | Sort-Object)
$sum = ($selected | Measure-Object -Sum).Sum
$middle = [int][math]::Floor($sorted.Count / 2)
$median = if ($sorted.Count % 2) { $sorted[$middle] } else { ($sorted[$middle - 1] + $sorted[$middle]) / 2 }
$slowCount = [math]::Max(1, [int][math]::Ceiling($sorted.Count * 0.01))
$slowSum = 0.0
for ($i = $sorted.Count - $slowCount; $i -lt $sorted.Count; $i++) { $slowSum += $sorted[$i] }
$out = [ordered]@{
    schema_version = 1; valid = $true; process_id = [int64]$complete.process_id
    window_start_utc = $WindowStartUtc.ToUniversalTime().ToString('o')
    window_end_utc = $WindowEndUtc.ToUniversalTime().ToString('o')
    captured_frames = $rows.Count; measured_intervals = $selected.Count
    mean_frame_ms = $sum / $selected.Count; median_frame_ms = $median
    p95_frame_ms = Percentile $sorted 0.95; p99_frame_ms = Percentile $sorted 0.99
    rendered_fps = 1000.0 * $selected.Count / $sum
    one_percent_low_fps = 1000.0 * $slowCount / $slowSum
    over_50_ms = @($selected | Where-Object { $_ -gt 50 }).Count
    over_100_ms = @($selected | Where-Object { $_ -gt 100 }).Count
    budget_ms = $FrameBudgetMs
    over_budget = @($selected | Where-Object { $_ -gt $FrameBudgetMs }).Count
}
if ($fgLast) {
    $out.generated_frame_delta = [int64]$fgLast.generated_frames_total - [int64]$fgFirstPrior.generated_frames_total
    if ($fgPresentComplete -and $fgFirstPrior.swapchain_present_count -ne '' -and $fgLast.swapchain_present_count -ne '') {
        $out.swapchain_present_delta = [int64]$fgLast.swapchain_present_count - [int64]$fgFirstPrior.swapchain_present_count
    }
}
$json = $out | ConvertTo-Json -Depth 6
if ($OutputJson) { $json | Set-Content -LiteralPath $OutputJson -Encoding utf8NoBOM }
else { $json }
