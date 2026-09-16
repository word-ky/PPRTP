param(
    [string]$RunId = "",
    [int]$Lines = 100,
    [switch]$Follow
)

. "$PSScriptRoot/Autodl.Common.ps1"

$remoteBase = Get-AutodlRemoteBase
if ([string]::IsNullOrWhiteSpace($RunId)) {
    $RunId = Get-AutodlLatestRunId
}

$logFile = "$remoteBase/runs/$RunId/train.log"
Write-Host "[logs] run: $RunId"
Write-Host "[logs] file: $logFile"
Write-Host "---"

if ($Follow) {
    Invoke-AutodlSsh "tail -f $(Quote-Sh $logFile)"
} else {
    Invoke-AutodlSsh "tail -n $Lines $(Quote-Sh $logFile) 2>/dev/null || echo 'Log file not found.'"
}
