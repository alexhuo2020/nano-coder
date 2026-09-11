# Measure the END-TO-END solve rate of claude-cli driven by our 85M model.
#
# One attempt proves nothing here: the model's pass@1 on this class of
# breakage is ~48%, so a single failure and a single success are equally
# uninformative. This runs N independent trials, resetting the repo each time,
# and reports how many left a file that actually passes pytest.
param([int]$Trials = 8)

$env:ANTHROPIC_BASE_URL = "http://127.0.0.1:8788"
$env:ANTHROPIC_AUTH_TOKEN = "dummy-local-shim"
$env:ANTHROPIC_MODEL = "blackwell-nanogpt-85m"
# 200k, NOT the model's real 32k: the CLI counts ITS OWN ~18.8k prompt
# against this. Declaring 32768 made it decide it was near the limit and
# AUTO-COMPACT -- asking our 85M model to summarise the conversation, which
# produced word salad that then REPLACED the conversation and destroyed
# every turn after it. The shim strips the prompt to ~220 tokens, so the
# model never actually sees 18.8k.
$env:CLAUDE_CODE_MAX_CONTEXT_TOKENS = "200000"
$env:CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC = "1"

$d = "C:\Users\E20263395\AppData\Local\Temp\bnano_repo_test"
Set-Location $d

$solved = 0
$wrote = 0
for ($i = 1; $i -le $Trials; $i++) {
    # reset to the broken state
    "def add(a, b):`n    return a - b`n" | Set-Content -Path "$d\solution.py" -Encoding utf8 -NoNewline
    $before = Get-Content "$d\solution.py" -Raw

    claude -p "The file solution.py in this repo is failing its tests. Read it, fix it, and verify with run_tests." --max-turns 6 --permission-mode bypassPermissions 2>&1 | Out-Null

    $after = Get-Content "$d\solution.py" -Raw
    if ($after -ne $before) { $wrote++ }

    # the grader is pytest, not a string match on the file
    $res = & python -m pytest -q 2>&1 | Out-String
    if ($res -match "1 passed") {
        $solved++
        Write-Output "trial ${i}: SOLVED"
    } else {
        $changed = if ($after -ne $before) { "changed-but-wrong" } else { "unchanged" }
        Write-Output "trial ${i}: failed ($changed)"
    }
}
Write-Output ""
Write-Output "=== claude-cli + blackwell-nanogpt-85m ==="
Write-Output "trials:  $Trials"
Write-Output "solved:  $solved  ($([math]::Round(100.0 * $solved / $Trials, 1))%)"
Write-Output "edited the file: $wrote"
