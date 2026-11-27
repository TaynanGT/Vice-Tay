# Runs play.ps1, saves the diagnostics log, and prints a ready-to-paste summary.
# Example (Pair 1 - code isolation):
#   .\scripts\ab-bench.ps1 -Exe 'D:\victory-data\game\game.exe' -Label 'exp-laptop' -Machine laptop
# Example (Pair 2 - preset isolation):
#   .\scripts\ab-bench.ps1 -Exe 'D:\victory-data\game\game.exe' -Label 'exp-4060' -Machine 4060
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$Exe,
    [string]$Label = 'run',
    [ValidateSet('series-x', 'series-s')][string]$Console = 'series-s',
    [ValidateSet('auto', '7900xtx', '4070', '4090', '4060', 'laptop')][string]$Machine = 'laptop',
    [ValidateSet('diag', 'release')][string]$Build = 'diag',
    [int]$MinDraws = 1001,
    [string]$LogDir = '.\logs',
    [switch]$DlssQuality,
    [switch]$FrameGen
)

New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$logPath = Join-Path $LogDir "$Label-$Machine-$stamp.log"

$playArgs = @{
    Exe     = $Exe
    Console = $Console
    Machine = $Machine
    Build   = $Build
}
if ($DlssQuality) { $playArgs.Dlss = 'quality' }
if ($FrameGen) { $playArgs.FrameGen = $true }

Write-Host "Logging run to $logPath ..."
& (Join-Path $PSScriptRoot 'play.ps1') @playArgs 2>&1 | Tee-Object -FilePath $logPath

if ($Build -eq 'diag' -and (Test-Path -LiteralPath $logPath)) {
    Write-Host "`n--- Summary ($Label / $Machine, MinDraws=$MinDraws) ---"
    try {
        $summary = & (Join-Path $PSScriptRoot 'measure-frames.ps1') -Path $logPath -MinDraws $MinDraws
        $summary | Format-Table -AutoSize
        Write-Host ("Markdown row: | $Label | $Machine | {0:N1} | {1:N1} | {2:N1} | {3} |" -f
            $summary.MedianRenderedFps, $summary.P10RenderedFps, $summary.MedianFrameMs, $summary.Samples)
    } catch {
        Write-Warning "Could not summarize log: $($_.Exception.Message)"
    }
}
