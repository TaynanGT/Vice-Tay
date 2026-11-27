# Optional NVIDIA runtime setup. The engine never downloads or updates DLLs.
# NVIDIA Streamline v2.14.16; DLLs and licences stay outside this repository.
param([Parameter(Mandatory=$true)][string]$Directory)
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$repo = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..')).TrimEnd('\')
$destination = [IO.Path]::GetFullPath($Directory)
if ($destination.TrimEnd('\') -eq $repo -or $destination.StartsWith($repo + '\', [StringComparison]::OrdinalIgnoreCase)) {
    throw 'NVIDIA runtime files must be outside the repository.'
}
$files = @('sl.interposer.dll', 'sl.common.dll', 'sl.dlss.dll', 'sl.dlss_g.dll',
    'sl.reflex.dll', 'sl.pcl.dll', 'nvngx_dlss.dll', 'nvngx_dlssg.dll')

function Remove-PrivateDirectory([string]$Path, [string]$Parent) {
    if (-not (Test-Path -LiteralPath $Path -PathType Container)) { return }
    $resolved = (Resolve-Path -LiteralPath $Path).ProviderPath
    $item = Get-Item -LiteralPath $resolved -Force
    if ($resolved -ne [IO.Path]::GetFullPath($Path) -or
        ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -or
        -not $resolved.StartsWith($Parent.TrimEnd('\') + '\', [StringComparison]::OrdinalIgnoreCase)) {
        throw 'The temporary runtime directory moved.'
    }
    foreach ($file in Get-ChildItem -LiteralPath $resolved -File -Force) {
        [IO.File]::Delete($file.FullName)
    }
    [IO.Directory]::Delete($resolved)
}

function Install-VerifiedRuntime([string]$Staging, [string]$Destination) {
    [void][IO.Directory]::CreateDirectory($Destination)
    $lockPath = Join-Path $Destination '.vice-tay-streamline-install.lock'
    $lock = [IO.FileStream]::new($lockPath, [IO.FileMode]::CreateNew,
        [IO.FileAccess]::ReadWrite, [IO.FileShare]::None, 4096, [IO.FileOptions]::DeleteOnClose)
    $transaction = Join-Path $Destination ('.streamline-' + [Guid]::NewGuid().ToString('N'))
    $changes = [Collections.Generic.List[object]]::new()
    $recovered = $true
    try {
        [void][IO.Directory]::CreateDirectory($transaction)
        foreach ($file in Get-ChildItem -LiteralPath $Staging -File | Sort-Object Name) {
            $target = Join-Path $Destination $file.Name
            $incoming = Join-Path $transaction ($file.Name + '.incoming')
            $previous = if ([IO.File]::Exists($target)) {
                Join-Path $transaction ($file.Name + '.previous')
            }
            [IO.File]::Copy($file.FullName, $incoming)
            $change = [pscustomobject]@{
                Target = $target
                Incoming = $incoming
                Previous = $previous
                Installed = $false
            }
            $changes.Add($change)
            if ($previous) {
                [IO.File]::Replace($incoming, $target, $previous)
            } else {
                [IO.File]::Move($incoming, $target)
            }
            $change.Installed = $true
        }
    } catch {
        $failure = $_
        for ($i = $changes.Count - 1; $i -ge 0; $i--) {
            $change = $changes[$i]
            try {
                if ($change.Previous -and [IO.File]::Exists($change.Previous)) {
                    if ([IO.File]::Exists($change.Target)) {
                        [IO.File]::Replace($change.Previous, $change.Target, $change.Incoming + '.discard')
                    } else {
                        [IO.File]::Move($change.Previous, $change.Target)
                    }
                } elseif ($change.Installed -and -not $change.Previous) {
                    [IO.File]::Delete($change.Target)
                }
            } catch {
                $recovered = $false
                Write-Warning "Could not restore $($change.Target). $_"
            }
        }
        if (-not $recovered) {
            throw "The install and rollback failed. Previous files are in $transaction. $failure"
        }
        throw $failure
    } finally {
        try {
            if ($recovered) { Remove-PrivateDirectory $transaction $Destination }
        } finally {
            $lock.Dispose()
        }
    }
}

Add-Type -AssemblyName System.Net.Http
Add-Type -AssemblyName System.IO.Compression
$client = [Net.Http.HttpClient]::new()
$client.Timeout = [TimeSpan]::FromMinutes(10)
$archive = $null
$memory = $null
$tempRoot = [IO.Path]::GetFullPath([IO.Path]::GetTempPath())
$staging = Join-Path $tempRoot ('vice-tay-streamline-' + [Guid]::NewGuid().ToString('N'))
try {
    if ($staging.StartsWith($repo + '\', [StringComparison]::OrdinalIgnoreCase)) {
        throw 'Use a temporary directory outside the repository for runtime setup.'
    }
    [void][IO.Directory]::CreateDirectory($staging)
    $url = 'https://github.com/NVIDIA-RTX/Streamline/releases/download/v2.14.16/streamline-sdk-v2.14.16.zip'
    Write-Host 'Downloading official NVIDIA Streamline v2.14.16 (263 MiB) ...'
    $bytes = $client.GetByteArrayAsync($url).GetAwaiter().GetResult()
    $memory = [IO.MemoryStream]::new($bytes, $false)
    $archive = [IO.Compression.ZipArchive]::new($memory, [IO.Compression.ZipArchiveMode]::Read)
    foreach ($file in $files) {
        $entry = @($archive.Entries | Where-Object {
            $_.FullName -match ('(^|/)bin/x64/' + [regex]::Escape($file) + '$') -and
            $_.FullName -notmatch '/development/'
        })
        if ($entry.Count -ne 1) { throw "Expected exactly one release DLL named $file." }
        $inputStream = $entry[0].Open()
        $outputStream = [IO.File]::Create((Join-Path $staging $file))
        try { $inputStream.CopyTo($outputStream) }
        finally { $outputStream.Dispose(); $inputStream.Dispose() }
    }
    # Preserve the upstream licences without extracting arbitrary archive paths.
    $licences = @($archive.Entries | Where-Object {
        $_.FullName -match '(^|/)(license\.txt|bin/x64/[^/]*license[^/]*\.txt)$' -and
        $_.FullName -notmatch '/development/'
    })
    foreach ($entry in $licences) {
        $leaf = [IO.Path]::GetFileName($entry.FullName)
        $inputStream = $entry.Open()
        $outputStream = [IO.File]::Create((Join-Path $staging $leaf))
        try { $inputStream.CopyTo($outputStream) }
        finally { $outputStream.Dispose(); $inputStream.Dispose() }
    }
    if (-not (Test-Path -LiteralPath (Join-Path $staging 'license.txt'))) {
        throw 'The official SDK licence was not found.'
    }
    foreach ($file in $files) {
        $path = Join-Path $staging $file
        $signature = Get-AuthenticodeSignature -LiteralPath $path
        if ($signature.Status -ne 'Valid' -or $signature.SignerCertificate.Subject -notmatch 'NVIDIA') {
            throw "NVIDIA signature validation failed for $file ($($signature.Status))."
        }
        Write-Host ("Verified NVIDIA signature: {0} ({1:N0} bytes)" -f $file, (Get-Item -LiteralPath $path).Length)
    }
    Install-VerifiedRuntime $staging $destination
} finally {
    if ($archive) { $archive.Dispose() }
    if ($memory) { $memory.Dispose() }
    $client.Dispose()
    Remove-PrivateDirectory $staging $tempRoot
}
Write-Host "Runtime ready: $destination"
