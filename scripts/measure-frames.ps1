# Summarize the rendered frame reports emitted by victory_gpu::host.
[CmdletBinding()]
param(
    [Parameter(Mandatory, Position = 0)][string]$Path,
    [datetimeoffset]$From,
    [datetimeoffset]$Until,
    [ValidateRange(0, [int]::MaxValue)][int]$MinDraws = 0
)

$hasFrom = $PSBoundParameters.ContainsKey('From')
$hasUntil = $PSBoundParameters.ContainsKey('Until')
if ($hasFrom -and $hasUntil -and $Until -le $From) {
    throw 'Until must be later than From.'
}

$file = (Get-Item -LiteralPath $Path -ErrorAction Stop).FullName
$culture = [Globalization.CultureInfo]::InvariantCulture
$pattern = [regex]'^\s*(?<time>\d{4}-\d\d-\d\dT\S+)\s+INFO\s+victory_gpu::host:\s+(?<fps>\d+(?:\.\d+)?) fps, (?<ms>\d+(?:\.\d+)?) ms/frame \(.*?\), (?<draws>\d+) draws/frame'
$recordStart = [regex]'^\s*\d{4}-\d\d-\d\dT\S+(?=\s|$)'
$rates = [Collections.Generic.List[double]]::new()
$times = [Collections.Generic.List[double]]::new()
$report = ''
foreach ($line in [IO.File]::ReadLines($file)) {
    $part = $line -replace '\x1b\[[0-9;]*m', ''
    if ($recordStart.IsMatch($part)) {
        $report = $part
    } elseif ($report -and -not [string]::IsNullOrWhiteSpace($part)) {
        $report = $report.TrimEnd() + ' ' + $part.TrimStart()
    }
    if (!$report) { continue }

    if (-not $report.Contains(' draws/frame')) {
        # A copied console report can wrap. Bound the incomplete record.
        if ($report.Length -gt 512) { $report = '' }
        continue
    }
    $sample = $pattern.Match($report)
    $report = ''
    if (-not $sample.Success -or [long]$sample.Groups['draws'].Value -lt $MinDraws) { continue }
    if ($hasFrom -or $hasUntil) {
        $time = [datetimeoffset]::Parse($sample.Groups['time'].Value, $culture)
        if (($hasFrom -and $time -lt $From) -or ($hasUntil -and $time -ge $Until)) { continue }
    }
    $rates.Add([double]::Parse($sample.Groups['fps'].Value, $culture))
    $times.Add([double]::Parse($sample.Groups['ms'].Value, $culture))
}
if ($rates.Count -eq 0) { throw 'No rendered frame reports matched the selected interval and draw count.' }

function Get-Median([Collections.Generic.List[double]]$sorted) {
    $upper = [int][math]::Floor($sorted.Count / 2)
    $lower = [int][math]::Floor(($sorted.Count - 1) / 2)
    ($sorted[$lower] + $sorted[$upper]) / 2
}
$rates.Sort()
$times.Sort()
[pscustomobject]@{
    Samples           = $rates.Count
    MedianRenderedFps = Get-Median $rates
    P10RenderedFps    = $rates[[int][math]::Ceiling(0.1 * $rates.Count) - 1]
    MedianFrameMs     = Get-Median $times
}
