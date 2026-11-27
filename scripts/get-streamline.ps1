# Optional NVIDIA runtime setup. The engine never downloads or updates DLLs.
# NVIDIA Streamline v2.14.16; DLLs and licences stay outside this repository.
param([Parameter(Mandatory=$true)][string]$Directory)
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$repo = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..')).TrimEnd('\')
$destination = [IO.Path]::GetFullPath($Directory).TrimEnd('\')
if ($destination -eq $repo -or $destination.StartsWith($repo + '\', [StringComparison]::OrdinalIgnoreCase)) {
    throw 'NVIDIA runtime files must be outside the repository.'
}
$files = @('sl.interposer.dll', 'sl.common.dll', 'sl.dlss.dll', 'sl.dlss_g.dll',
    'sl.reflex.dll', 'sl.pcl.dll', 'nvngx_dlss.dll', 'nvngx_dlssg.dll')
New-Item -ItemType Directory -Force -Path $destination | Out-Null
Add-Type -AssemblyName System.Net.Http
Add-Type -AssemblyName System.IO.Compression
$client = [Net.Http.HttpClient]::new()
$client.Timeout = [TimeSpan]::FromMinutes(10)
$archive = $null
$memory = $null
try {
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
        $outputStream = [IO.File]::Create((Join-Path $destination $file))
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
        $outputStream = [IO.File]::Create((Join-Path $destination $leaf))
        try { $inputStream.CopyTo($outputStream) }
        finally { $outputStream.Dispose(); $inputStream.Dispose() }
    }
    if (-not (Test-Path -LiteralPath (Join-Path $destination 'license.txt'))) {
        throw 'The official SDK licence was not found.'
    }
    foreach ($file in $files) {
        $path = Join-Path $destination $file
        $signature = Get-AuthenticodeSignature -LiteralPath $path
        if ($signature.Status -ne 'Valid' -or $signature.SignerCertificate.Subject -notmatch 'NVIDIA') {
            throw "NVIDIA signature validation failed for $file ($($signature.Status))."
        }
        Write-Host ("Verified NVIDIA signature: {0} ({1:N0} bytes)" -f $file, (Get-Item -LiteralPath $path).Length)
    }
} finally {
    if ($archive) { $archive.Dispose() }
    if ($memory) { $memory.Dispose() }
    $client.Dispose()
}
Write-Host "Runtime ready: $destination"
