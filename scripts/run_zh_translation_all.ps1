param(
    [Parameter(Mandatory = $false)]
    [string]$ApiKey = $env:OPENAI_API_KEY,

    [Parameter(Mandatory = $false)]
    [string]$Model = "gpt-4o-mini",

    [Parameter(Mandatory = $false)]
    [string]$ApiBase = "https://api.openai.com/v1",

    [Parameter(Mandatory = $false)]
    [int]$Timeout = 120,

    [Parameter(Mandatory = $false)]
    [string[]]$Phases = @("all"),

    [Parameter(Mandatory = $false)]
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

if (-not $ApiKey) {
    throw "请设置 OPENAI_API_KEY 环境变量，或通过 -ApiKey 传入。"
}

function Invoke-Phase {
    param([string]$phase)

    if ($phase -eq "all") {
        Write-Host "=== 全量翻译: --all --force ==="
        $cmd = "python scripts/run_zh_translation.py --api-key `"$ApiKey`" --model `"$Model`" --api-base `"$ApiBase`" --timeout $Timeout --all --force"
        if ($DryRun) { $cmd += " --dry-run" }
        Write-Host $cmd
        Invoke-Expression $cmd
    }
    else {
        Write-Host "=== Phase $phase 翻译 ==="
        $cmd = "python scripts/run_zh_translation.py --api-key `"$ApiKey`" --model `"$Model`" --api-base `"$ApiBase`" --timeout $Timeout --phase $phase --all --force"
        if ($DryRun) { $cmd += " --dry-run" }
        Write-Host $cmd
        Invoke-Expression $cmd
    }
}

foreach ($phase in $Phases) {
    Invoke-Phase $phase
    if (-not $DryRun) {
        Write-Host "`n>> 该阶段完成后做一次部分验收"
        python scripts/audit_i18n.py --lang zh --phase $phase --require-all
    }
}

if (-not $DryRun) {
    Write-Host "`n>> 全量验收"
    python scripts/audit_i18n.py --lang zh --require-all
    Write-Host "完成提示：若无问题，直接执行 git add/commit/push。"
}
