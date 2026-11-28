$ErrorActionPreference = 'Stop'
function Assert($ok, $message) { if (-not $ok) { throw $message } }
$repo = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$scratch = Join-Path ([IO.Path]::GetTempPath()) ("vice-tay-runner-tests-" + [guid]::NewGuid().ToString('N'))
$bin = Join-Path $scratch 'bin'; $dataSeed = Join-Path $scratch 'data-seed'; $cacheSeed = Join-Path $scratch 'cache-seed'
foreach ($path in @($scratch,$bin,$dataSeed,$cacheSeed)) { New-Item -ItemType Directory -Path $path | Out-Null }
$dataFile = Join-Path $dataSeed 'save.txt'; $cacheFile = Join-Path $cacheSeed 'cache.txt'
[IO.File]::WriteAllText($dataFile, 'private synthetic save')
[IO.File]::WriteAllText($cacheFile, 'private synthetic cache')
$dataHash = (Get-FileHash -LiteralPath $dataFile -Algorithm SHA256).Hash
$cacheHash = (Get-FileHash -LiteralPath $cacheFile -Algorithm SHA256).Hash
$fake = Join-Path $bin 'fake-runtime.exe'
$rustcExe = (Get-Command rustc -ErrorAction Stop).Source
$rustArgs = @('--edition=2021','-O',(Join-Path $repo 'tests\fixtures\frame-benchmark\main.rs'),'-o',$fake)
& $rustcExe @rustArgs
Assert ($LASTEXITCODE -eq 0 -and (Test-Path -LiteralPath $fake -PathType Leaf)) 'Synthetic runtime build failed.'
$benchmark = Join-Path $PSScriptRoot 'benchmark.ps1'
function NewRecipe([string]$name, [bool]$drop) {
    $launchPath = Join-Path $scratch "$name-launch.json"
    $recipePath = Join-Path $scratch "$name-recipe.json"
    $outputRoot = Join-Path $scratch "$name-output"
    $environment = @{}
    if ($drop) { $environment['VICTORY_FAKE_DROP'] = '1' }
    [ordered]@{ schema_version = 1; runtime = $fake; exe = $fake; console = 'series-x'; working_directory = $repo; config_path = (Join-Path $scratch 'absent.toml'); config_sha256 = $null; environment = $environment } |
        ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $launchPath -Encoding utf8NoBOM
    [ordered]@{
        schema_version = 1; launch_environment = $launchPath; data_seed = $dataSeed; cache_seed = $cacheSeed
        output_root = $outputRoot; repeats = 1; ready_timeout_seconds = 3; warmup_seconds = 0.05
        measure_seconds = 0.2; close_timeout_seconds = 3; frame_budget_ms = 33
        startup_keys = @([ordered]@{ at_seconds = 0.1; keys = @('Enter 0.1') })
        route_keys = @([ordered]@{ at_seconds = 0.05; keys = @('W 0.1') })
    } | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $recipePath -Encoding utf8NoBOM
    return [pscustomobject]@{ Recipe = $recipePath; Output = $outputRoot }
}
$good = NewRecipe 'good' $false
& $benchmark -Recipe $good.Recipe
$goodDir = Join-Path $good.Output 'run-001'
$manifest = Get-Content -LiteralPath (Join-Path $goodDir 'manifest.json') -Raw | ConvertFrom-Json
$summary = Get-Content -LiteralPath (Join-Path $goodDir 'summary.json') -Raw | ConvertFrom-Json
Assert ($manifest.status -eq 'complete' -and $manifest.shutdown -eq 'graceful') 'Successful run manifest incomplete.'
Assert ($summary.valid -and $summary.process_id -eq $manifest.process_id -and $summary.measured_intervals -gt 0) 'Successful run summary invalid.'
Assert ($manifest.events.Count -eq 2 -and $manifest.events[0].phase -eq 'startup' -and $manifest.events[1].phase -eq 'route') 'Key batches not observed consumed.'
Assert ($manifest.tail_frame_index -gt 0 -and $manifest.window_start_utc -and $manifest.window_end_utc) 'Tail/window evidence missing.'
Assert ((Get-FileHash -LiteralPath (Join-Path $goodDir 'data\save.txt') -Algorithm SHA256).Hash -eq $dataHash) 'Data seed copy changed.'
Assert ((Get-FileHash -LiteralPath (Join-Path $goodDir 'cache\cache.txt') -Algorithm SHA256).Hash -eq $cacheHash) 'Cache seed copy changed.'
Assert ((Get-FileHash -LiteralPath $dataFile -Algorithm SHA256).Hash -eq $dataHash -and (Get-FileHash -LiteralPath $cacheFile -Algorithm SHA256).Hash -eq $cacheHash) 'Original seed changed.'
Assert ((Get-Content -LiteralPath (Join-Path $goodDir 'stdout.log') -Raw) -match 'synthetic runtime completed') 'stdout log missing.'
Assert ((Get-Content -LiteralPath (Join-Path $goodDir 'stderr.log') -Raw) -match 'synthetic stderr marker') 'stderr log missing.'
Assert (-not (Get-Process -Id $manifest.process_id -ErrorAction SilentlyContinue)) 'Successful runtime remains alive.'
$bad = NewRecipe 'dropped' $true
$badPlan = Get-Content -LiteralPath $bad.Recipe -Raw | ConvertFrom-Json
$badPlan.repeats = 2
$badPlan | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $bad.Recipe -Encoding utf8NoBOM
$failed = $false
try { & $benchmark -Recipe $bad.Recipe } catch { $failed = $true }
Assert $failed 'Dropped-frame run unexpectedly passed.'
$badDir = Join-Path $bad.Output 'run-001'
$badManifest = Get-Content -LiteralPath (Join-Path $badDir 'manifest.json') -Raw | ConvertFrom-Json
Assert ($badManifest.status -eq 'incomplete' -and $badManifest.failure -match 'dropped') 'Dropped-frame failure not retained.'
Assert (Test-Path -LiteralPath (Join-Path $badDir 'frames.csv') -PathType Leaf) 'Failed run CSV evidence missing.'
Assert (-not (Test-Path -LiteralPath (Join-Path $bad.Output 'run-002'))) 'Runner retried after invalid run.'
Assert (-not (Get-Process -Id $badManifest.process_id -ErrorAction SilentlyContinue)) 'Failed runtime remains alive.'
Assert ((Get-FileHash -LiteralPath $dataFile -Algorithm SHA256).Hash -eq $dataHash -and (Get-FileHash -LiteralPath $cacheFile -Algorithm SHA256).Hash -eq $cacheHash) 'Seed changed after failed run.'
Write-Output "Synthetic runner end-to-end tests passed. Retained evidence: $scratch"
