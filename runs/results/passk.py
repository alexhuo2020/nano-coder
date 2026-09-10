"""Is the `stub` tier learnable AT ALL, or structurally impossible?

pass@1 tells you the current rate. pass@k tells you whether RL has anything to
work with: GRPO's advantage is group-relative, so a task where no sample ever
succeeds contributes a zero-variance group and gets DROPPED. If pass@k is 0 for
every task, an RL run on this tier cannot take a single update step -- which is
exactly the 0/100-episode failure this project already hit once. Measuring this
costs 20 minutes; discovering it from a dead training run costs GPU-hours.

Also dumps what the model actually WRITES, because the failure mode matters:
plausible-but-wrong code means a capability floor (teacher demos cannot fix it),
whereas garbage or an echoed stub would mean something mechanical and fixable.
"""
import os, sys, math, random, shutil, collections
os.environ.setdefault("NVTE_ALLOW_UNSAFE_PICKLE_EXTRA_STATE", "1")
sys.path.insert(0, "/home/ubuntu/bnano")
import torch
from blackwell_lm.checkpoint import load_stage_checkpoint
from blackwell_lm.model import BlackwellLM, ModelConfig
from blackwell_lm.repo_agent import run_repo_episode
from blackwell_lm.scenario import build_scenarios
from blackwell_lm.tasks import get_tasks
from blackwell_lm.tokenizer import EOS, load_tokenizer

TIER = sys.argv[1] if len(sys.argv) > 1 else "stub"
NTASK, K = 20, 12

tok = load_tokenizer("/home/ubuntu/bnano/tokenizer.json"); eos = tok.token_to_id(EOS)
p = "/home/ubuntu/bnano/agent_repo_ab.pt"
ck = torch.load(p, map_location="cpu", weights_only=False)
cfg = ModelConfig(**ck["cfg"]); cfg.max_seq_len = max(cfg.max_seq_len, 2048)
m = BlackwellLM(cfg, precision="bf16", device="cuda", dtype=torch.bfloat16)
load_stage_checkpoint(p, m)

scns = build_scenarios(get_tasks("mbpp"), seed=11, limit=NTASK, difficulty=TIER)
random.Random(0).shuffle(scns)
scns = scns[:NTASK]

any_solved = 0        # tasks with >=1 success in K samples  -> pass@K
tot_solved = tot = 0  # per-sample rate                      -> pass@1
partial = 0           # samples earning ANY partial credit (a nonzero gradient)
best_reward = []
kinds = collections.Counter()
samples = []

for sc in scns:
    hit = 0
    for _ in range(K):
        ep = run_repo_episode(m, tok, sc, eos, max_turns=5, max_new_tokens=256,
                              n_loops=cfg.n_loops, keep_repo=True)
        tot += 1
        if ep.reward >= 1.0: hit += 1; tot_solved += 1
        if ep.reward > 0.0: partial += 1
        # classify what it left in the file
        sol = os.path.join(getattr(ep, "repo", "") or "", "solution.py")
        code = ""
        for cand in (sol,):
            if cand and os.path.isfile(cand):
                code = open(cand, encoding="utf-8", errors="replace").read()
        if not code.strip():
            kinds["empty/missing"] += 1
        elif code.strip().endswith("pass") and "return" not in code:
            kinds["still a stub"] += 1
        else:
            try:
                compile(code, "s.py", "exec"); kinds["valid python"] += 1
            except SyntaxError:
                kinds["syntax error"] += 1
            if len(samples) < 6: samples.append((sc.name, ep.reward, code[:400]))
        r = getattr(ep, "repo", None)
        if r: shutil.rmtree(r, ignore_errors=True)
    best_reward.append(hit)
    any_solved += 1 if hit else 0

se = lambda q, n: math.sqrt(max(q * (1 - q), 1e-9) / n) * 100
print()
print(f"=== tier={TIER}  {NTASK} tasks x {K} samples = {tot} episodes ===")
print(f"pass@1            {tot_solved/tot*100:5.1f}% +-{se(tot_solved/tot, tot):.1f}   ({tot_solved}/{tot} samples)")
print(f"pass@{K:<2}           {any_solved/NTASK*100:5.1f}%          ({any_solved}/{NTASK} tasks ever solved)")
print(f"any partial credit{partial/tot*100:5.1f}%          ({partial}/{tot} samples with reward>0)")
print("what it left in solution.py:", dict(kinds))
print()
for name, r, code in samples:
    print(f"--- {name}  reward={r:.2f} " + "-" * 40)
    print(code.rstrip()[:400])
