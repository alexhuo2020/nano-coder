import os, sys, math, random
os.environ.setdefault("NVTE_ALLOW_UNSAFE_PICKLE_EXTRA_STATE","1")
sys.path.insert(0,"/home/ubuntu/bnano")
import torch
from blackwell_lm.checkpoint import load_stage_checkpoint
from blackwell_lm.model import BlackwellLM, ModelConfig
from blackwell_lm.repo_agent import run_repo_episode
from blackwell_lm.scenario import build_scenarios
from blackwell_lm.tasks import get_tasks
from blackwell_lm.tokenizer import EOS, load_tokenizer

tok = load_tokenizer("/home/ubuntu/bnano/tokenizer.json"); eos = tok.token_to_id(EOS)
p = "/home/ubuntu/bnano/agent_repo_ab.pt"
ck = torch.load(p, map_location="cpu", weights_only=False)
cfg = ModelConfig(**ck["cfg"]); cfg.max_seq_len = max(cfg.max_seq_len, 2048)
m = BlackwellLM(cfg, precision="bf16", device="cuda", dtype=torch.bfloat16)
load_stage_checkpoint(p, m)
mtasks = get_tasks("mbpp")
print()
print("%-8s %6s %12s %12s %8s" % ("tier","n","solved+-SE","wrote+-SE","reward"))
for d in ("mutate","multi","stub","swap"):
    scns = build_scenarios(mtasks, seed=11, limit=40, difficulty=d)
    random.Random(0).shuffle(scns)
    sample = scns[:40]
    solved=wrote=n=0; rs=[]
    for sc in sample:
        for _ in range(3):
            ep = run_repo_episode(m, tok, sc, eos, max_turns=5, max_new_tokens=256, n_loops=cfg.n_loops)
            n+=1; rs.append(ep.reward)
            wrote += 1 if ep.wrote_file else 0
            solved += 1 if ep.reward>=1.0 else 0
    se=lambda q: math.sqrt(max(q*(1-q),1e-9)/n)*100
    print("%-8s %6d %7.0f%%+-%-3.0f %7.0f%%+-%-3.0f %8.3f" % (
        d, n, solved/n*100, se(solved/n), wrote/n*100, se(wrote/n), sum(rs)/n))
