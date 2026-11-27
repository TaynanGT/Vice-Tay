# Summarize the rendered frame reports emitted by victory_gpu::host.
[CmdletBinding(DefaultParameterSetName = 'Summary')]
param(
    [Parameter(Mandatory, Position = 0)][string]$Path,
    [datetimeoffset]$From,
    [datetimeoffset]$Until,
    [ValidateRange(0, [int]::MaxValue)][int]$MinDraws = 0,
    [Parameter(Mandatory, ParameterSetName = 'TextureProgress')][switch]$TextureProgress,
    [Parameter(ParameterSetName = 'Summary')][switch]$TimingBreakdown
)

$hasFrom = $PSBoundParameters.ContainsKey('From')
$hasUntil = $PSBoundParameters.ContainsKey('Until')
if ($hasFrom -and $hasUntil -and $Until -le $From) {
    throw 'Until must be later than From.'
}

$file = (Get-Item -LiteralPath $Path -ErrorAction Stop).FullName
$culture = [Globalization.CultureInfo]::InvariantCulture
$framePattern = '^\s*(?<time>\d{4}-\d\d-\d\dT\S+)\s+INFO\s+victory_gpu::host:\s+(?<fps>\d+(?:\.\d+)?) fps, (?<ms>\d+(?:\.\d+)?) ms/frame \((?<timing>.*?)\), (?<draws>\d+) draws/frame'
$complete = ' draws/frame'
$limit = 512
if ($TextureProgress) {
    $framePattern += '.*?textures picked by index: (?<uploaded>\d+) uploaded so far \((?<mib>\d+) MB\), (?<waiting>\d+) waiting, (?<unmapped>\d+) passed over unmapped'
    $complete = ' passed over unmapped'
    $limit = 2048
}
$pattern = [regex]$framePattern
$recordStart = [regex]'^\s*\d{4}-\d\d-\d\dT\S+(?=\s|$)'
$rates = [Collections.Generic.List[double]]::new()
$times = [Collections.Generic.List[double]]::new()
$timingPattern = [regex]'^(?<PrepareMs>\d+(?:\.\d+)?) preparing:.*?;\s+(?<GpuWaitMs>\d+(?:\.\d+)?) on the GPU;\s+(?<CopyBackMs>\d+(?:\.\d+)?) copying back\b'
$timingSamples = [ordered]@{
    PrepareMs  = [Collections.Generic.List[double]]::new()
    GpuWaitMs  = [Collections.Generic.List[double]]::new()
    CopyBackMs = [Collections.Generic.List[double]]::new()
}
$report = ''
$progressSamples = 0
foreach ($line in [IO.File]::ReadLines($file)) {
    $part = $line -replace '\x1b\[[0-9;]*m', ''
    if ($recordStart.IsMatch($part)) {
        $report = $part
    } elseif ($report -and -not [string]::IsNullOrWhiteSpace($part)) {
        $report = $report.TrimEnd() + ' ' + $part.TrimStart()
    }
    if (!$report) { continue }

    if (-not $report.Contains($complete)) {
        # A copied console report can wrap. Bound the incomplete record.
        if ($report.Length -gt $limit) { $report = '' }
        continue
    }
    $sample = $pattern.Match($report)
    $report = ''
    if (-not $sample.Success -or [long]$sample.Groups['draws'].Value -lt $MinDraws) { continue }
    if ($hasFrom -or $hasUntil -or $TextureProgress) {
        $time = [datetimeoffset]::Parse($sample.Groups['time'].Value, $culture)
        if (($hasFrom -and $time -lt $From) -or ($hasUntil -and $time -ge $Until)) { continue }
    }
    if ($TextureProgress) {
        [pscustomobject]@{
            TimestampUtc     = $time.UtcDateTime.ToString('o', $culture)
            DrawsPerFrame    = [long]$sample.Groups['draws'].Value
            UploadedTextures = [long]$sample.Groups['uploaded'].Value
            UploadedMiB      = [long]$sample.Groups['mib'].Value
            WaitingTextures  = [long]$sample.Groups['waiting'].Value
            PassedUnmapped   = [long]$sample.Groups['unmapped'].Value
        }
        $progressSamples++
        continue
    }
    if ($TimingBreakdown) {
        $timing = $timingPattern.Match($sample.Groups['timing'].Value)
        if (-not $timing.Success) { continue }
        foreach ($field in $timingSamples.Keys) {
            $timingSamples[$field].Add([double]::Parse($timing.Groups[$field].Value, $culture))
        }
    }
    $rates.Add([double]::Parse($sample.Groups['fps'].Value, $culture))
    $times.Add([double]::Parse($sample.Groups['ms'].Value, $culture))
}
if ($TextureProgress) {
    if ($progressSamples -eq 0) { throw 'No texture progress reports matched the selected interval and draw count.' }
    return
}
if ($rates.Count -eq 0) {
    $description = if ($TimingBreakdown) { 'rendered frame reports with submission timings' } else { 'rendered frame reports' }
    throw "No $description matched the selected interval and draw count."
}

function Get-Median([Collections.Generic.List[double]]$sorted) {
    $upper = [int][math]::Floor($sorted.Count / 2)
    $lower = [int][math]::Floor(($sorted.Count - 1) / 2)
    ($sorted[$lower] + $sorted[$upper]) / 2
}
$rates.Sort()
$times.Sort()
$summary = [ordered]@{
    Samples           = $rates.Count
    MedianRenderedFps = Get-Median $rates
    P10RenderedFps    = $rates[[int][math]::Ceiling(0.1 * $rates.Count) - 1]
    MedianFrameMs     = Get-Median $times
}
if ($TimingBreakdown) {
    foreach ($field in $timingSamples.Keys) {
        $timingSamples[$field].Sort()
        $summary["Median$field"] = Get-Median $timingSamples[$field]
    }
}
[pscustomobject]$summary
