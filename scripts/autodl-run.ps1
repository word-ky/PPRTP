param(
    [Parameter(Mandatory = $true)][string]$Name,
    [string]$Cmd = ""
)

. "$PSScriptRoot/Autodl.Common.ps1"

$root = Get-AutodlProjectRoot
$config = Get-AutodlConfig
$remoteBase = Get-AutodlRemoteBase

if ([string]::IsNullOrWhiteSpace($Cmd)) {
    if (-not $config.default_command) {
        throw "No command provided and config field default_command is missing."
    }
    $Cmd = [string]$config.default_command
}

$runId = New-AutodlId -Suffix $Name -Default "run"
$runDir = "$remoteBase/runs/$runId"
$workDir = "$remoteBase/current"
$releaseFile = Join-Path $root ".autodl/last-release"
$releaseId = if (Test-Path -LiteralPath $releaseFile) { Get-Content -LiteralPath $releaseFile -Raw } else { "unknown" }
$releaseId = $releaseId.Trim()
$session = "autodl-$runId"

Write-Host "[run] run id: $runId"
Write-Host "[run] remote dir: $runDir"
Write-Host "[run] command: $Cmd"

Invoke-AutodlSsh "mkdir -p $(Quote-Sh "$runDir/artifacts")"

$meta = [ordered]@{
    runId = $runId
    releaseId = $releaseId
    sessionName = $session
    command = $Cmd
} | ConvertTo-Json -Depth 5
$meta64 = ConvertTo-Base64Utf8 $meta
Invoke-AutodlSsh "printf %s $(Quote-Sh $meta64) | base64 -d > $(Quote-Sh "$runDir/meta.json")"

$runScript = @(
    "#!/usr/bin/env bash",
    "set -uo pipefail",
    "cd $(Quote-Sh $workDir)",
    "export AUTODL_RUN_ID=$(Quote-Sh $runId)",
    "export AUTODL_RUN_DIR=$(Quote-Sh $runDir)",
    "export AUTODL_ARTIFACTS_DIR=$(Quote-Sh "$runDir/artifacts")",
    'mkdir -p "$AUTODL_ARTIFACTS_DIR"',
    'echo "[autodl] run_id=$AUTODL_RUN_ID"',
    'echo "[autodl] started_at=$(date -Is)"',
    "{",
    $Cmd,
    "}",
    'status=$?',
    'echo "[autodl] finished_at=$(date -Is)"',
    'echo "[autodl] exit_code=$status"',
    'exit $status'
) -join "`n"
$runScript += "`n"

$runScript64 = ConvertTo-Base64Utf8 $runScript
Invoke-AutodlSsh "printf %s $(Quote-Sh $runScript64) | base64 -d > $(Quote-Sh "$runDir/run.sh") && chmod +x $(Quote-Sh "$runDir/run.sh")"

$base = Get-AutodlSshBaseArgs
$hasTmux = (& ssh @($base.Common) -p ([string]$base.Config.port) $base.Target "command -v tmux >/dev/null 2>&1 && echo yes || echo no").Trim()

if ($hasTmux -eq "yes") {
    Write-Host "[run] starting tmux session: $session"
    Invoke-AutodlSsh "tmux new-session -d -s $(Quote-Sh $session) ""bash $(Quote-Sh "$runDir/run.sh") > $(Quote-Sh "$runDir/train.log") 2>&1"""
} else {
    Write-Host "[run] tmux not found, using nohup"
    Invoke-AutodlSsh "nohup bash $(Quote-Sh "$runDir/run.sh") > $(Quote-Sh "$runDir/train.log") 2>&1 < /dev/null &"
}

Save-AutodlState -Name "last-run" -Value $runId
Write-Host "[run] started: $runId"
