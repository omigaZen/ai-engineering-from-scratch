param(
    [Parameter(Mandatory = $false)]
    [string]$ApiKey = $env:OPENAI_API_KEY,
    [Parameter(Mandatory = $false)]
    [string]$Model = "gpt-4o-mini",
    [Parameter(Mandatory = $false)]
    [string]$ApiBase = "https://api.openai.com/v1",
    [Parameter(Mandatory = $false)]
    [int]$Timeout = 180,
    [Parameter(Mandatory = $false)]
    [string[]]$Phases = @("all"),
    [Parameter(Mandatory = $false)]
    [switch]$DryRun,
    [Parameter(Mandatory = $false)]
    [string]$LogFile = "translation.log"
)

$ErrorActionPreference = "Stop"

if (-not $ApiKey) {
    throw "OPENAI_API_KEY is required. Pass -ApiKey or set the environment variable."
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repoRoot

function Invoke-Phase {
    param([string]$phase)

    if ($phase -eq "all") {
        Write-Host "=== Running full translation for all missing lessons ==="
        $cmd = "python scripts/run_zh_translation.py --api-key `"$ApiKey`" --model `"$Model`" --api-base `"$ApiBase`" --timeout $Timeout --all --force"
    }
    else {
        Write-Host "=== Running phase $phase ==="
        $cmd = "python scripts/run_zh_translation.py --api-key `"$ApiKey`" --model `"$Model`" --api-base `"$ApiBase`" --timeout $Timeout --phase $phase --all --force"
    }

    if ($DryRun) { $cmd += " --dry-run" }
    Write-Host $cmd
    Invoke-Expression $cmd

    if (-not $DryRun) {
        if ($phase -eq "all") {
            python scripts/audit_i18n.py --lang zh --require-all | Tee-Object -FilePath $LogFile -Append
        }
        else {
            python scripts/audit_i18n.py --lang zh --phase $phase --require-all | Tee-Object -FilePath $LogFile -Append
        }
    }
}

foreach ($phase in $Phases) {
    try {
        Invoke-Phase $phase
    }
    catch {
        Write-Host "Phase $phase failed: $($_.Exception.Message)" | Tee-Object -FilePath $LogFile -Append
        throw
    }
}

if (-not $DryRun) {
    Write-Host "=== Final full audit ==="
    python scripts/audit_i18n.py --lang zh --require-all | Tee-Object -FilePath $LogFile -Append
    Write-Host "Done. If the audit passes, run: git add -A && git commit -m \"feat(i18n): complete zh translations for all lessons\" && git push"
}
