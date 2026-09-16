param(
    [string]$RunId = ""
)

. "$PSScriptRoot/Autodl.Common.ps1"

$root = Get-AutodlProjectRoot
$remoteBase = Get-AutodlRemoteBase
if ([string]::IsNullOrWhiteSpace($RunId)) {
    $RunId = Get-AutodlLatestRunId
}

$remoteDir = "$remoteBase/runs/$RunId"
$localBase = Join-Path $root "research_log/remote_runs"
$localDir = Join-Path $localBase $RunId
New-Item -ItemType Directory -Path $localBase -Force | Out-Null

Write-Host "[fetch] run: $RunId"
Write-Host "[fetch] remote: $remoteDir"
Write-Host "[fetch] local: $localDir"

Copy-FromAutodl -RemotePath $remoteDir -LocalPath $localBase

Write-Host "[fetch] done"
