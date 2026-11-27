# Launch-free checks for play.ps1; no title, runtime build, or CIM access is required.
$ErrorActionPreference = 'Stop'
$play = Join-Path $PSScriptRoot 'play.ps1'
$tokens = $null
$parseErrors = $null
[void][System.Management.Automation.Language.Parser]::ParseFile($play, [ref]$tokens, [ref]$parseErrors)
if ($parseErrors.Count) { throw ($parseErrors | Out-String) }

$saved = @{}
foreach ($entry in Get-ChildItem Env:VICTORY_*) { $saved[$entry.Name] = $entry.Value }
$messages = [System.Collections.Generic.List[string]]::new()
function Write-Host {
    param([object]$Object)
    [void]$messages.Add([string]$Object)
}
function Get-CimInstance {
    [CmdletBinding()]
    param([string]$ClassName)
    if ($queryMode -eq 'unavailable') { throw 'CIM unavailable for this check' }
    switch ($ClassName) {
        'Win32_ComputerSystem' { [pscustomobject]@{ TotalPhysicalMemory = 16GB } }
        'Win32_OperatingSystem' { [pscustomobject]@{ FreePhysicalMemory = 512 * 1024 } }
        'Win32_PerfFormattedData_PerfOS_Memory' {
            if ($queryMode -eq 'partial') { throw 'Commit query unavailable for this check' }
            [pscustomobject]@{ CommittedBytes = 2GB; CommitLimit = 8GB }
        }
        'Win32_PageFileUsage' {
            if ($queryMode -eq 'partial') { throw 'Page-file query unavailable for this check' }
            [pscustomobject]@{ AllocatedBaseSize = 1024; CurrentUsage = 10; PeakUsage = 20 }
            [pscustomobject]@{ AllocatedBaseSize = 2048; CurrentUsage = 30; PeakUsage = 40 }
        }
        default { throw "Unexpected CIM class $ClassName" }
    }
}
function Test-Path { param($LiteralPath, $PathType) $true }
# Even if the dry-run return regresses, no real executable can be invoked.
function Resolve-Path { param($LiteralPath) [pscustomobject]@{ Path = 'X:\no-launch-test\not-an-executable.exe' } }
function Get-FileHash { [CmdletBinding()] param($LiteralPath, $Algorithm) [pscustomobject]@{ Hash = 'test-hash' } }
function Assert-LaunchCheck {
    param([bool]$Condition, [string]$Message)
    if (-not $Condition) { throw $Message }
}
function Invoke-LaunchCheck {
    param([string]$Mode, [hashtable]$More = @{})
    $queryMode = $Mode
    $messages.Clear()
    & $play -Exe 'X:\no-launch-test\not-a-title.exe' -Machine laptop -More $More -DryRun
    Assert-LaunchCheck ($messages -contains 'Dry run: not launching the title.') 'Dry run did not return safely'
}

try {
    $env:VICTORY_GAME_DIR = 'X:\preserved-game-directory'
    $env:VICTORY_ASYNC_SUBMIT = 'inherited-value'
    $env:VICTORY_THIN = '0'
    Invoke-LaunchCheck 'available'
    Assert-LaunchCheck ($env:VICTORY_GAME_DIR -eq 'X:\preserved-game-directory') 'Inherited game directory was lost'
    Assert-LaunchCheck ($env:VICTORY_THIN -eq '1') 'Preset did not replace inherited non-path setting'
    Assert-LaunchCheck ($null -eq [Environment]::GetEnvironmentVariable('VICTORY_ASYNC_SUBMIT')) 'Inherited diagnostic was not discarded'
    $headroom = '{0:N0} MiB' -f 6144
    $allocated = '{0:N0} MiB' -f 3072
    Assert-LaunchCheck ([bool]($messages -like "*headroom $headroom*")) 'Commit summary is incorrect'
    Assert-LaunchCheck ([bool]($messages -like "*page files allocated/current/peak $allocated/40 MiB/60 MiB*")) 'Page-file totals are incorrect'
    Assert-LaunchCheck ([bool]($messages -like 'Discarded inherited values for *VICTORY_ASYNC_SUBMIT*')) 'Discarded diagnostic was not reported'

    Invoke-LaunchCheck 'available' @{
        VICTORY_GAME_DIR = 'X:\explicit-game-directory'
        VICTORY_THIN = '0'
        VICTORY_ASYNC_SUBMIT = '0'
    }
    Assert-LaunchCheck ($env:VICTORY_GAME_DIR -eq 'X:\explicit-game-directory') 'Explicit game directory did not win'
    Assert-LaunchCheck ($env:VICTORY_THIN -eq '0') 'Explicit setting did not override preset'
    Assert-LaunchCheck ($env:VICTORY_ASYNC_SUBMIT -eq '0') 'Explicit diagnostic was discarded'
    Assert-LaunchCheck ($messages -contains '  VICTORY_ASYNC_SUBMIT=0') 'Effective diagnostic was not displayed'

    Invoke-LaunchCheck 'partial'
    Assert-LaunchCheck ([bool]($messages -like '*physical free 512 MiB; system commit unavailable*')) 'Partial-query summary is incorrect'
    Invoke-LaunchCheck 'unavailable'
    Assert-LaunchCheck ($messages -contains 'Windows memory (read-only): physical, commit, and page-file queries are unavailable.') 'Unavailable memory was not reported'
    Write-Output 'PASS: parsing, memory totals, partial/failed queries, inherited paths, override precedence, and dry-run safety'
} finally {
    foreach ($entry in Get-ChildItem Env:VICTORY_*) { Remove-Item "Env:$($entry.Name)" }
    foreach ($key in $saved.Keys) { Set-Item "Env:$key" $saved[$key] }
}