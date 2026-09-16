. "$PSScriptRoot/Autodl.Common.ps1"

$remoteBase = Get-AutodlRemoteBase

Write-Host "=== AutoDL Status ==="
Write-Host ""

Write-Host "--- GPU ---"
Invoke-AutodlSsh "nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv 2>/dev/null || echo 'nvidia-smi not available'"
Write-Host ""

Write-Host "--- Disk ---"
Invoke-AutodlSsh "df -h /autodl-fs/data 2>/dev/null || df -h / 2>/dev/null"
Write-Host ""

Write-Host "--- Current Release ---"
Invoke-AutodlSsh "readlink $(Quote-Sh "$remoteBase/current") 2>/dev/null || echo 'no current release'"
Write-Host ""

Write-Host "--- Recent Runs ---"
Invoke-AutodlSsh "ls -1t $(Quote-Sh "$remoteBase/runs") 2>/dev/null | head -5 || echo 'no runs'"
Write-Host ""

Write-Host "--- tmux Sessions ---"
Invoke-AutodlSsh "tmux list-sessions 2>/dev/null || echo 'no tmux sessions'"
Write-Host ""

Write-Host "--- Running Processes ---"
Invoke-AutodlSsh "ps -u $(Quote-Sh ([string](Get-AutodlConfig).user)) -o pid,ppid,etime,%cpu,%mem,cmd --sort=-%cpu 2>/dev/null | head -20 || true"
Invoke-AutodlSsh "ps aux | grep -E 'python|train|torchrun|accelerate' | grep -v grep | head -20 || echo 'none'"
