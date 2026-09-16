param(
    [string]$Tag = "release",
    [string]$Source = "",
    [string[]]$ExtraExclude = @()
)

. "$PSScriptRoot/Autodl.Common.ps1"

$root = Get-AutodlProjectRoot
$config = Get-AutodlConfig
$remoteBase = Get-AutodlRemoteBase
$releaseId = New-AutodlId -Suffix $Tag -Default "release"
$remoteRelease = "$remoteBase/releases/$releaseId"

if ([string]::IsNullOrWhiteSpace($Source)) {
    $sourcePath = $root
} else {
    $sourcePath = (Resolve-Path -LiteralPath $Source).Path
}

Write-Host "[deploy] release id: $releaseId"
Write-Host "[deploy] source: $sourcePath"
Write-Host "[deploy] remote target: $remoteRelease"

Invoke-AutodlSsh "mkdir -p $(Quote-Sh $remoteRelease) $(Quote-Sh "$remoteBase/runs") $(Quote-Sh "$remoteBase/shared")"

$archive = Join-Path ([System.IO.Path]::GetTempPath()) "autodl-deploy-$releaseId.tar.gz"
if (Test-Path -LiteralPath $archive) {
    Remove-Item -LiteralPath $archive -Force
}

$excludes = @(".git", ".autodl/config.json", ".autodl/last-release", ".autodl/last-run", "remote-runs", "analysis-packs", "__pycache__", "*.pyc", ".env")
if ($config.PSObject.Properties.Name -contains "deploy_excludes" -and $config.deploy_excludes) {
    $excludes = @($config.deploy_excludes)
}
$excludes += @($ExtraExclude | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })

$tarArgs = @("-czf", $archive, "-C", $sourcePath)
foreach ($exclude in $excludes) {
    $tarArgs += "--exclude=$exclude"
}
$tarArgs += "."

Write-Host "[deploy] creating archive..."
& tar @tarArgs
if ($LASTEXITCODE -ne 0) {
    throw "tar failed with exit code $LASTEXITCODE"
}

try {
    Write-Host "[deploy] uploading archive..."
    Copy-ToAutodl -LocalPath $archive -RemotePath "/tmp/autodl-deploy-$releaseId.tar.gz"

    Write-Host "[deploy] extracting archive..."
    $remoteArchive = "/tmp/autodl-deploy-$releaseId.tar.gz"
    Invoke-AutodlSsh "cd $(Quote-Sh $remoteRelease) && tar xzf $(Quote-Sh $remoteArchive) && rm -f $(Quote-Sh $remoteArchive)"

    Write-Host "[deploy] updating current symlink..."
    Invoke-AutodlSsh "ln -sfn $(Quote-Sh $remoteRelease) $(Quote-Sh "$remoteBase/current")"

    Save-AutodlState -Name "last-release" -Value $releaseId
    Write-Host "[deploy] done: $releaseId"
} finally {
    if (Test-Path -LiteralPath $archive) {
        Remove-Item -LiteralPath $archive -Force
    }
}
