Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Get-AutodlProjectRoot {
    $scriptDir = Split-Path -Parent $PSCommandPath
    return (Resolve-Path (Join-Path $scriptDir "..")).Path
}

function Get-AutodlConfigPath {
    if ($env:AUTODL_CONFIG_PATH) {
        return (Resolve-Path -LiteralPath $env:AUTODL_CONFIG_PATH).Path
    }
    $root = Get-AutodlProjectRoot
    return Join-Path $root ".autodl/config.json"
}

function Get-AutodlConfig {
    $path = Get-AutodlConfigPath
    if (-not (Test-Path -LiteralPath $path)) {
        $example = Join-Path (Get-AutodlProjectRoot) ".autodl/config.example.json"
        throw "Missing $path. Copy $example to .autodl/config.json and fill in your AutoDL SSH details."
    }

    return Get-Content -LiteralPath $path -Raw | ConvertFrom-Json
}

function Get-AutodlRemoteBase {
    $config = Get-AutodlConfig
    if (-not $config.remote_base) {
        throw "Config field remote_base is required."
    }
    return [string]$config.remote_base
}

function Get-AutodlSshBaseArgs {
    $config = Get-AutodlConfig
    foreach ($field in @("host", "port", "user")) {
        if (-not $config.$field) {
            throw "Config field $field is required."
        }
    }

    $knownHostsDir = Join-Path ([System.IO.Path]::GetTempPath()) "autodl-workflow"
    New-Item -ItemType Directory -Path $knownHostsDir -Force | Out-Null
    $knownHosts = Join-Path $knownHostsDir "known_hosts"

    $common = @(
        "-o", "ConnectTimeout=15",
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "BatchMode=yes",
        "-o", "UserKnownHostsFile=$knownHosts"
    )

    if ($config.identity_file) {
        $identity = [string]$config.identity_file
        if ($identity.StartsWith("~/")) {
            $identity = Join-Path $HOME $identity.Substring(2)
        }
        if (Test-Path -LiteralPath $identity) {
            $common += @("-i", $identity)
        }
    }

    return [pscustomobject]@{
        Config = $config
        Common = $common
        Target = "$($config.user)@$($config.host)"
    }
}

function Quote-Sh {
    param([Parameter(Mandatory = $true)][string]$Value)
    return "'" + $Value.Replace("'", "'""'""'") + "'"
}

function Invoke-AutodlSsh {
    param([Parameter(Mandatory = $true)][string]$Command)

    $base = Get-AutodlSshBaseArgs
    $args = @($base.Common) + @("-p", [string]$base.Config.port, $base.Target, $Command)
    & ssh @args
    if ($LASTEXITCODE -ne 0) {
        throw "ssh failed with exit code $LASTEXITCODE"
    }
}

function Open-AutodlSsh {
    $base = Get-AutodlSshBaseArgs
    $args = @($base.Common) + @("-p", [string]$base.Config.port, $base.Target)
    & ssh @args
    if ($LASTEXITCODE -ne 0) {
        throw "ssh failed with exit code $LASTEXITCODE"
    }
}

function Copy-ToAutodl {
    param(
        [Parameter(Mandatory = $true)][string]$LocalPath,
        [Parameter(Mandatory = $true)][string]$RemotePath
    )

    $base = Get-AutodlSshBaseArgs
    $remote = "$($base.Target):$RemotePath"
    $args = @($base.Common) + @("-P", [string]$base.Config.port, "-r", $LocalPath, $remote)
    & scp @args
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "default scp upload failed; retrying with legacy SCP protocol"
        $legacyArgs = @($base.Common) + @("-O", "-P", [string]$base.Config.port, "-r", $LocalPath, $remote)
        & scp @legacyArgs
        if ($LASTEXITCODE -ne 0) {
            throw "scp upload failed with exit code $LASTEXITCODE"
        }
    }
}

function Copy-FromAutodl {
    param(
        [Parameter(Mandatory = $true)][string]$RemotePath,
        [Parameter(Mandatory = $true)][string]$LocalPath
    )

    $base = Get-AutodlSshBaseArgs
    $remote = "$($base.Target):$RemotePath"
    $args = @($base.Common) + @("-P", [string]$base.Config.port, "-r", $remote, $LocalPath)
    & scp @args
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "default scp download failed; retrying with legacy SCP protocol"
        $legacyArgs = @($base.Common) + @("-O", "-P", [string]$base.Config.port, "-r", $remote, $LocalPath)
        & scp @legacyArgs
        if ($LASTEXITCODE -ne 0) {
            throw "scp download failed with exit code $LASTEXITCODE"
        }
    }
}

function New-AutodlId {
    param(
        [Parameter(Mandatory = $true)][string]$Suffix,
        [Parameter(Mandatory = $true)][string]$Default
    )

    if ([string]::IsNullOrWhiteSpace($Suffix)) {
        $Suffix = $Default
    }
    $safeSuffix = ($Suffix -replace "[^A-Za-z0-9._-]", "-").Trim("-")
    if ([string]::IsNullOrWhiteSpace($safeSuffix)) {
        $safeSuffix = $Default
    }
    return "$(Get-Date -Format 'yyyyMMdd-HHmmss')-$safeSuffix"
}

function Save-AutodlState {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$Value
    )

    $root = Get-AutodlProjectRoot
    $dir = Join-Path $root ".autodl"
    New-Item -ItemType Directory -Path $dir -Force | Out-Null
    Set-Content -LiteralPath (Join-Path $dir $Name) -Value $Value -NoNewline -Encoding utf8
}

function Get-AutodlLatestRunId {
    $remoteBase = Get-AutodlRemoteBase
    $runsPath = Quote-Sh "$remoteBase/runs"
    $base = Get-AutodlSshBaseArgs
    $args = @($base.Common) + @("-p", [string]$base.Config.port, $base.Target, "ls -1t $runsPath 2>/dev/null | head -1")
    $output = & ssh @args
    if ($LASTEXITCODE -ne 0) {
        throw "ssh failed while listing remote runs."
    }
    $runId = ($output | Select-Object -First 1).Trim()
    if ([string]::IsNullOrWhiteSpace($runId)) {
        throw "No remote runs found."
    }
    return $runId
}

function ConvertTo-Base64Utf8 {
    param([Parameter(Mandatory = $true)][string]$Text)
    return [Convert]::ToBase64String([System.Text.Encoding]::UTF8.GetBytes($Text))
}
