# How to run Chinese i18n backfill in one pass

Scripts in this repo:

- `scripts/openai_translator.py`: stdin/stdout translator bridge (OpenAI-compatible chat/completions).
- `scripts/translate_missing_zh.py`: batch runner for lessons missing `docs/zh.md` / `quiz.zh.json`.
- `scripts/run_zh_translation.py`: single entry wrapper (`--all` + per-phase execution).
- `scripts/run_zh_translation_all.ps1`: PowerShell helper to run multiple phases sequentially.

Recommended usage:

```powershell
# one-time setup
$env:OPENAI_API_KEY = "sk-..."

# full backfill (recommended)
python scripts/run_zh_translation.py --api-key $env:OPENAI_API_KEY --model gpt-4o-mini --all --force

# phase-by-phase (safer for large runs)
.\scripts\run_zh_translation_all.ps1 -Phases @("02","03","04","19") -ApiKey $env:OPENAI_API_KEY -Model "gpt-4o-mini" -Timeout 120
```

Validation after translate:

```powershell
python scripts/audit_i18n.py --lang zh --require-all
```

When `audit_i18n` returns exit code `0`, perform one final commit of all `docs/zh.md` and `quiz.zh.json` changes.
