# End-to-end claude-cli trials against the local 85M model.
#
# Start-Process, not Start-Job: a job spawns a whole PowerShell runspace per
# attempt and the batch was killed twice for host memory pressure. The timeout
# still has to exist -- a hung CLI must not stall the batch, and a timeout is
# recorded as a failure rather than silently dropped.
#
# The tunnel is probed before every attempt, because a dead forward previously
# produced "failures" that never reached the model at all.
param([int]$Trials = 5, [int]$TimeoutSec = 200)

$env:ANTHROPIC_BASE_URL = "http://127.0.0.1:8788"
$env:ANTHROPIC_AUTH_TOKEN = "dummy-local-shim"
$env:ANTHROPIC_MODEL = "blackwell-nanogpt-85m"
$env:CLAUDE_CODE_MAX_CONTEXT_TOKENS = "200000"
$env:CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC = "1"

$d = "C:\Users\USER\AppData\Local\Temp\bnano_demo"
Set-Location $d

@'
from solution import add

def test_add():
    assert add(2, 3) == 5
    assert add(0, 0) == 0
    assert add(-1, 1) == 0
'@ | Set-Content -Path "$d\test_solution.py" -Encoding utf8

# The FULL path to claude.exe. `claude` on PATH is a shell wrapper, and
# Start-Process rejects it with "%1 is not a valid Win32 application" -- which
# silently produced a 0/5 in which the CLI never launched at all.
$exe = Join-Path $env:USERPROFILE "node-v22.18.0-win-x64\node-v22.18.0-win-x64\node_modules\@anthropic-ai\claude-code\bin\claude.exe"
if (-not (Test-Path $exe)) { Write-Output "claude.exe not found at $exe"; exit 1 }

$prompt = "The file solution.py in this repo is failing its tests. Read it, fix it, and verify with run_tests."
$solved = 0; $reached = 0

for ($i = 1; $i -le $Trials; $i++) {
    try { $null = Invoke-WebRequest -Uri "http://127.0.0.1:8788/" -TimeoutSec 5 -UseBasicParsing }
    catch { Write-Output ("trial {0}: SKIPPED (tunnel down)" -f $i); continue }
    $reached++

    "def add(a, b):`n    return a - b`n" | Set-Content -Path "$d\solution.py" -Encoding utf8 -NoNewline
    if (Test-Path "$d\__pycache__") { Remove-Item -Recurse -Force "$d\__pycache__" }

    $p = Start-Process -FilePath $exe `
        -ArgumentList @("-p", $prompt, "--max-turns", "6", "--permission-mode", "bypassPermissions") `
        -NoNewWindow -PassThru -RedirectStandardOutput "$d\last_run.txt" -RedirectStandardError "$d\last_err.txt"
    $timedOut = $false
    if (-not $p.WaitForExit($TimeoutSec * 1000)) {
        $timedOut = $true
        try { $p.Kill() } catch {}
    }

    if (Test-Path "$d\__pycache__") { Remove-Item -Recurse -Force "$d\__pycache__" }
    $res = & python -m pytest -q 2>&1 | Out-String
    if ($res -match "1 passed") {
        $solved++
        Write-Output ("trial {0}: SOLVED" -f $i)
        if ($solved -eq 1) {
            Write-Output "--- solution.py as written by the model ---"
            Get-Content "$d\solution.py"
            Write-Output "-------------------------------------------"
        }
    } elseif ($timedOut) {
        Write-Output ("trial {0}: failed (CLI timeout)" -f $i)
    } else {
        Write-Output ("trial {0}: failed" -f $i)
    }
}
Write-Output ""
Write-Output ("=== claude-cli + blackwell-nanogpt-85m: {0}/{1} solved ===" -f $solved, $reached)
