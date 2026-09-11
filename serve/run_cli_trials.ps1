# Measure the END-TO-END solve rate of a coding CLI driven by our 85M model.
#
# WHY TRIALS AND NOT ONE RUN. The model's pass@1 on this class of task is in
# the tens of percent, so a single success and a single failure are equally
# uninformative -- twice in this project a single good run was mistaken for
# "it works" and contradicted by the next observation.
#
# WHY TWO BREAKAGE TYPES. `mutate` (one token changed) is the flattering tier;
# `stub` (body replaced by pass) is the honest one. Reporting only the first
# is how this project produced a 70% figure that collapsed to 0%.
#
# Usage: run_cli_trials.ps1 [-Trials 6] [-Kind mutate|stub|both]
param([int]$Trials = 6, [string]$Kind = "both")

$env:ANTHROPIC_BASE_URL = "http://127.0.0.1:8788"
$env:ANTHROPIC_AUTH_TOKEN = "dummy-local-shim"
$env:ANTHROPIC_MODEL = "blackwell-nanogpt-85m"
# 200k, NOT the model's real 32k: the CLI counts ITS OWN ~18.8k prompt against
# this. Declaring 32768 made it decide it was near the limit and AUTO-COMPACT
# -- asking our 85M model to summarise the conversation, which produced word
# salad that then REPLACED the conversation and destroyed every later turn.
# The shim strips the prompt to ~220 tokens, so the model never sees 18.8k.
$env:CLAUDE_CODE_MAX_CONTEXT_TOKENS = "200000"
$env:CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC = "1"

$d = "C:\Users\E20263395\AppData\Local\Temp\bnano_repo_test"
if (-not (Test-Path $d)) { New-Item -ItemType Directory $d | Out-Null }
Set-Location $d

# The grader is pytest and the tests are fixed: the model never supplies them.
@'
from solution import add

def test_add():
    assert add(2, 3) == 5
    assert add(0, 0) == 0
    assert add(-1, 1) == 0
'@ | Set-Content -Path "$d\test_solution.py" -Encoding utf8

$broken = @{
    # one token changed: the flattering tier
    mutate = "def add(a, b):`n    return a - b`n"
    # body removed entirely: the honest tier
    stub   = "def add(a, b):`n    pass`n"
}
$kinds = if ($Kind -eq "both") { @("mutate", "stub") } else { @($Kind) }

$summary = @()
foreach ($k in $kinds) {
    $solved = 0; $wrote = 0
    for ($i = 1; $i -le $Trials; $i++) {
        $broken[$k] | Set-Content -Path "$d\solution.py" -Encoding utf8 -NoNewline
        $before = Get-Content "$d\solution.py" -Raw
        if (Test-Path "$d\__pycache__") { Remove-Item -Recurse -Force "$d\__pycache__" }

        claude -p "The file solution.py in this repo is failing its tests. Read it, fix it, and verify with run_tests." `
            --max-turns 8 --permission-mode bypassPermissions 2>&1 | Out-Null

        $after = Get-Content "$d\solution.py" -Raw
        if ($after -ne $before) { $wrote++ }
        if (Test-Path "$d\__pycache__") { Remove-Item -Recurse -Force "$d\__pycache__" }
        $res = & python -m pytest -q 2>&1 | Out-String
        if ($res -match "1 passed") { $solved++; $verdict = "SOLVED" }
        elseif ($after -ne $before) { $verdict = "failed (changed-but-wrong)" }
        else { $verdict = "failed (unchanged)" }
        Write-Output ("[{0}] trial {1}/{2}: {3}" -f $k, $i, $Trials, $verdict)
    }
    $pct = [math]::Round(100.0 * $solved / $Trials, 1)
    $summary += ("{0,-7} solved {1}/{2} ({3}%)  edited the file {4}/{2}" -f $k, $solved, $Trials, $pct, $wrote)
}

Write-Output ""
Write-Output "=== claude-cli + blackwell-nanogpt-85m ==="
$summary | ForEach-Object { Write-Output $_ }
