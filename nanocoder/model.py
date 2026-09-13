"""A small decoder LM whose every shape and precision choice was picked from
measurements on an RTX PRO Blackwell (sm_120), not from vendor guidance.

The measurements that drove each decision are cited inline; the raw tables are
in ../README.md. The short version:

  * FP8, NOT FP4. NVFP4 is Blackwell's headline feature and at this model width
    it is a LOSS: measured 42,583 tok/s (nvfp4) vs 43,762 (fp8) at d_model=1024.
    FP4 only starts paying at d_model>=3072 (1.227x), which is a model size you
    cannot train to competence on one GPU. This is the single most important
    finding baked into this file.
  * head_dim = 128 and every dimension a multiple of 128, so the tensor cores
    see clean tiles.
  * Grouped-query attention: the attention core stays BF16 (no low-precision
    path), so its K/V traffic lands on Blackwell's WEAK axis -- GDDR7 has
    roughly half an H100's bandwidth. GQA 3:1 cuts that traffic ~3x.
  * Deliberately boring elementwise. A profile of modded-nanogpt on this
    hardware attributed 17.6% of the step to elementwise kernels (gating, MUDD
    skips, per-head lambdas) and 13.2% to a large embedding backward -- 31%
    of runtime that three separate toolchain upgrades could not move, because
    it is bandwidth-bound. So: RMSNorm + RoPE + SwiGLU, tied embeddings, a
    modest vocab, and nothing clever.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # optional: the low-precision paths need NVIDIA Transformer Engine
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import DelayedScaling, Format, NVFP4BlockScaling
except Exception as _e:  # pragma: no cover - exercised only on machines without TE
    te = None
    DelayedScaling = Format = NVFP4BlockScaling = None
    _TE_IMPORT_ERROR = f"{type(_e).__name__}: {_e}"
else:
    _TE_IMPORT_ERROR = None


@dataclass
class ModelConfig:
    # Defaults are config "B" from the README's measured comparison: one shared
    # block looped 8x at d_model 1536. Chosen because NVFP4's benefit is
    # governed by d_model (measured: worthless at 768, 1.084x at 1536, 1.227x
    # at 3072) while trainable-parameter count must stay small enough to reach
    # ~190 tokens/param in ~2 days on one GPU. Looping is what lets those two
    # constraints coexist.
    vocab_size: int = 32768          # multiple of 128; small on purpose (embedding backward is bandwidth-bound)
    d_model: int = 1536
    n_layers: int = 1                # UNIQUE blocks
    n_heads: int = 12                # 1536/12 = 128 head_dim
    n_kv_heads: int = 4              # GQA 3:1
    ffn_hidden: int = 6144           # 4x d_model
    max_seq_len: int = 4096
    window: int = 1024               # sliding-window attention; 0 disables (full causal)
    global_every: int = 4            # every Nth UNIQUE layer sees the full context
    # Apply the full-context cadence over the LOOP index instead of the layer
    # index. This is the one that does anything for a looped model (n_layers=1
    # makes global_every inert -- see Attention.__init__). 0 = off, which is
    # what the current run trained with.
    global_every_loop: int = 0
    rope_theta: float = 10000.0
    tie_embeddings: bool = True
    # Looped / recurrent depth (Universal-Transformer style weight sharing).
    # n_layers UNIQUE blocks are applied n_loops times each, giving an effective
    # depth of n_layers*n_loops for the parameter cost of n_layers. This is what
    # makes d_model>=3072 -- the only regime where NVFP4 actually pays (measured
    # 1.227x) -- affordable in parameters on a single GPU.
    n_loops: int = 8
    # Recurrent-depth robustness: train with the loop count SAMPLED from this
    # inclusive range so the model learns to be depth-agnostic, which is what
    # makes test-time compute scaling actually work. Fixed-depth training gives
    # a model no reason to behave sensibly at any other depth, so without this
    # the n_loops override is decoration. None = always use n_loops.
    loop_sample: tuple[int, int] | None = (4, 12)
    # A purely weight-shared block cannot tell loop 1 from loop 8 -- it sees only
    # its input. Universal Transformers add a timestep signal for exactly this
    # reason; without it looped depth is strictly weaker than dense depth at
    # equal FLOPs. Costs max_loops*d_model params (~18K here).
    loop_conditioning: bool = True
    max_loops: int = 16
    # QK-Norm (Gemma / Chameleon / modded-nanogpt): RMSNorm q and k over head_dim
    # before attention. Bounds attention logits, which matters MORE under FP8/FP4
    # than in BF16 because the logit range is what low precision clips first.
    qk_norm: bool = True
    # Zero-init the residual-writing projections (muP-like). Each block then
    # starts as an identity map, which matters more for a LOOPED model: the same
    # perturbation is applied n_loops times, so a bad init compounds.
    zero_init_residual: bool = True

    @property
    def effective_depth(self) -> int:
        return self.n_layers * self.n_loops

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def qkv_width(self) -> int:
        return (self.n_heads + 2 * self.n_kv_heads) * self.head_dim

    def validate(self) -> None:
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.n_heads % self.n_kv_heads:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if self.head_dim != 128:
            raise ValueError(
                f"head_dim is {self.head_dim}; 128 is what the Blackwell tensor cores "
                "tile cleanly. Pick n_heads = d_model/128."
            )
        # TE's low-precision GEMMs require dimensions divisible by 16. The FUSED
        # QKV width is a GEMM dimension too, and a GQA split that is individually
        # legal can still produce an illegal width -- so check it explicitly.
        for name, v in (("d_model", self.d_model), ("ffn_hidden", self.ffn_hidden),
                        ("vocab_size", self.vocab_size), ("qkv_width", self.qkv_width)):
            if v % 16:
                raise ValueError(f"{name}={v} must be divisible by 16 for the FP8 path")


def make_fp8_recipe():
    """HYBRID = E4M3 forward, E5M2 backward, the standard split. Returns None
    when TE is unavailable so the model still runs in plain BF16."""
    if te is None:
        return None
    return DelayedScaling(margin=0, fp8_format=Format.HYBRID,
                          amax_history_len=16, amax_compute_algo="max")


def make_recipes(precision: str):
    """Returns (attn_recipe, mlp_recipe).

    NVFP4 is applied to the MLP only, with FP8 kept on the attention
    projections. That split is not arbitrary: the MLP is ~57% of FLOPs here and
    is the largest, most compute-bound GEMM, which is where 4-bit's 2x tensor
    rate can actually be realised; the attention projections are smaller and
    sit next to a BF16 attention core, so 4-bit buys less there and costs more
    numerically.
    """
    if precision == "bf16" or te is None:
        return None, None
    fp8 = make_fp8_recipe()
    if precision == "fp8":
        return fp8, fp8
    if precision == "nvfp4":
        return fp8, NVFP4BlockScaling()
    raise ValueError(f"unknown precision {precision!r}")


def nvfp4_stochastic_rounding_ok() -> bool:
    """NVFP4 on sm_120 is silently broken in the released TE wheel: it is built
    for plain sm_120, so the stochastic-rounding FP4 cast has no valid PTX. TE
    then falls back to biased rounding and reports NOTHING through Python --
    only a per-thread CUDA stderr spew, which also destroys throughput. Detect
    it by driving one real cast and watching the stderr channel."""
    if te is None or not torch.cuda.is_available():
        return False
    import ctypes, os, tempfile
    try:
        libc = ctypes.CDLL(None)
        saved = os.dup(2)
        with tempfile.TemporaryFile() as tmp:
            os.dup2(tmp.fileno(), 2)
            try:
                lin = te.Linear(64, 64, bias=False, device="cuda", params_dtype=torch.bfloat16)
                with te.autocast(enabled=True, recipe=NVFP4BlockScaling()):
                    lin(torch.randn(64, 64, device="cuda", dtype=torch.bfloat16))
                torch.cuda.synchronize()
                libc.fflush(None)
            finally:
                os.dup2(saved, 2)
                os.close(saved)
            tmp.seek(0)
            noise = tmp.read().decode("utf-8", "replace")
        return "architecture-specific" not in noise
    except Exception:
        return False


class Linear(nn.Module):
    """te.Linear when a low-precision recipe is active, nn.Linear otherwise,
    behind one name so the model body does not branch on precision."""

    def __init__(self, fan_in: int, fan_out: int, lowp: bool, device, dtype):
        super().__init__()
        if lowp and te is not None:
            self.inner = te.Linear(fan_in, fan_out, bias=False, device=device, params_dtype=dtype)
        else:
            self.inner = nn.Linear(fan_in, fan_out, bias=False, device=device, dtype=dtype)

    def forward(self, x):
        return self.inner(x)


def _ctx(recipe):
    return te.autocast(enabled=True, recipe=recipe) if recipe is not None else nullcontext()


class RoPE(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int, theta: float, device):
        super().__init__()
        inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
        t = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        f = torch.outer(t, inv)
        # kept in fp32: RoPE angles are cheap and precision here is free
        self.register_buffer("cos", f.cos(), persistent=False)
        self.register_buffer("sin", f.sin(), persistent=False)

    def forward(self, x: torch.Tensor, offset: int = 0):
        """x: [B, H, T, D]. `offset` supports incremental decoding, where the
        new token's absolute position is not 0."""
        t = x.shape[-2]
        cos = self.cos[offset:offset + t].repeat_interleave(2, dim=-1)[None, None]
        sin = self.sin[offset:offset + t].repeat_interleave(2, dim=-1)[None, None]
        x1, x2 = x[..., ::2], x[..., 1::2]
        rot = torch.stack((-x2, x1), dim=-1).flatten(-2)
        return (x * cos.to(x.dtype) + rot * sin.to(x.dtype))


try:
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention
    _flex = torch.compile(flex_attention, dynamic=False)
except Exception:  # pragma: no cover
    create_block_mask = flex_attention = _flex = None

_BLOCK_MASK_CACHE: dict = {}


def flex_windowed_attention(q, k, v, window: int):
    """Causal sliding-window attention over [B, H, T, D] via FlexAttention.

    The BlockMask depends only on (seq_len, window) -- both static within a
    training run -- so it is built once and cached. Falls back to an SDPA
    boolean mask if FlexAttention is unavailable, which is correct but ~1.4x
    slower (see the call site).
    """
    T = q.shape[-2]
    # FlexAttention has no CPU backward, so the CPU path (tests, debugging)
    # always takes the SDPA fallback below.
    if _flex is None or not q.is_cuda:
        qi = torch.arange(T, device=q.device)[:, None]
        ki = torch.arange(T, device=q.device)[None, :]
        m = (ki <= qi) & (qi - ki < window)
        return F.scaled_dot_product_attention(q, k, v, attn_mask=m, enable_gqa=True)

    key = (T, window, q.device.index)
    bm = _BLOCK_MASK_CACHE.get(key)
    if bm is None:
        def mask_mod(b, h, qi, ki):
            return (ki <= qi) & (qi - ki < window)
        bm = create_block_mask(mask_mod, None, None, T, T, device=q.device)
        _BLOCK_MASK_CACHE[key] = bm
    # FlexAttention broadcasts KV heads against Q heads for GQA, so k/v stay narrow.
    return _flex(q, k, v, block_mask=bm, enable_gqa=True)


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int, lowp: bool, attn_recipe, device, dtype):
        super().__init__()
        self.cfg = cfg
        self.attn_recipe = attn_recipe
        # A few full-context layers among cheap windowed ones: windowed attention
        # is O(T*w) instead of O(T^2), and the attention core is the part FP8
        # cannot accelerate, so most layers should be cheap.
        # NOTE: with the default n_layers=1 this cadence is INERT -- layer_idx is
        # always 0, so (0+1) % global_every is never 0 and every application is
        # windowed. That is the behaviour the current pretraining run has, so it
        # is kept as the default rather than changed underneath a live run. Set
        # cfg.global_every_loop to apply the cadence over the LOOP index instead,
        # which is what actually gives a looped model full-context applications.
        self.window = 0 if (cfg.global_every and (layer_idx + 1) % cfg.global_every == 0) else cfg.window
        self.global_every_loop = cfg.global_every_loop
        self.qkv = Linear(cfg.d_model, cfg.qkv_width, lowp, device, dtype)
        self.out = Linear(cfg.d_model, cfg.d_model, lowp, device, dtype)
        if cfg.qk_norm:
            self.q_norm = nn.RMSNorm(cfg.head_dim, eps=1e-6, device=device, dtype=dtype)
            self.k_norm = nn.RMSNorm(cfg.head_dim, eps=1e-6, device=device, dtype=dtype)
        else:
            self.q_norm = self.k_norm = None
        self.rope = RoPE(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta, device)

    def forward(self, x, cache=None, loop_idx: int = 0):
        B, T, _ = x.shape
        c = self.cfg
        qw = c.n_heads * c.head_dim
        kw = c.n_kv_heads * c.head_dim
        with _ctx(self.attn_recipe):
            qkv = self.qkv(x)
        q, k, v = qkv.split([qw, kw, kw], dim=-1)
        q = q.view(B, T, c.n_heads, c.head_dim).transpose(1, 2)
        k = k.view(B, T, c.n_kv_heads, c.head_dim).transpose(1, 2)
        v = v.view(B, T, c.n_kv_heads, c.head_dim).transpose(1, 2)

        # effective window for THIS application (see global_every_loop above)
        win = self.window
        if self.global_every_loop and (loop_idx + 1) % self.global_every_loop == 0:
            win = 0

        if self.q_norm is not None:
            # normalise per head BEFORE RoPE, so the rotation acts on unit-scale
            # vectors and the logit range stays bounded
            q, k = self.q_norm(q), self.k_norm(k)
        offset = 0 if cache is None else cache[0].shape[-2]
        q, k = self.rope(q, offset), self.rope(k, offset)

        if cache is not None:
            k = torch.cat([cache[0], k], dim=-2)
            v = torch.cat([cache[1], v], dim=-2)
        new_cache = (k, v)

        if cache is not None and T == 1:
            # Incremental decode: one query, so causality is automatic and no
            # mask is needed -- but the WINDOW still has to be applied. Letting
            # the single query see the whole cache would decode a windowed layer
            # with context it never saw in training, which is a silent
            # train/inference mismatch that grows with transcript length (and
            # the agentic phase has the longest transcripts).
            #
            # The window is applied by SLICING the keys rather than by trimming
            # the cache, because RoPE's `offset` is derived from the cache
            # length and must stay ABSOLUTE; a trimmed cache would restart
            # positions and silently corrupt them.
            if win:
                k_a, v_a = k[..., -win:, :], v[..., -win:, :]
            else:
                k_a, v_a = k, v
            y = F.scaled_dot_product_attention(q, k_a, v_a, enable_gqa=True)
        elif win:
            # MEASURED, and the reason this is not just an SDPA attn_mask:
            # handing SDPA an explicit boolean mask disables FlashAttention and
            # cost 1.42x here (58,653 vs 83,087 tok/s, d_model=768, seq 2048).
            # FlexAttention keeps a fused kernel AND skips fully-masked blocks,
            # so the window becomes a saving instead of a penalty.
            y = flex_windowed_attention(q, k, v, win)
        else:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)

        y = y.transpose(1, 2).reshape(B, T, c.d_model)
        with _ctx(self.attn_recipe):
            y = self.out(y)
        return y, new_cache


class SwiGLU(nn.Module):
    """The FP8-eligible bulk of the model: ~57% of FLOPs at this shape."""

    def __init__(self, cfg: ModelConfig, lowp: bool, mlp_recipe, device, dtype):
        super().__init__()
        self.mlp_recipe = mlp_recipe
        self.up_gate = Linear(cfg.d_model, 2 * cfg.ffn_hidden, lowp, device, dtype)
        self.down = Linear(cfg.ffn_hidden, cfg.d_model, lowp, device, dtype)

    def forward(self, x):
        with _ctx(self.mlp_recipe):
            gate, up = self.up_gate(x).chunk(2, dim=-1)
            return self.down(F.silu(gate) * up)


class Block(nn.Module):
    def __init__(self, cfg, layer_idx, lowp, attn_recipe, mlp_recipe, device, dtype):
        super().__init__()
        self.n1 = nn.RMSNorm(cfg.d_model, eps=1e-5, device=device, dtype=dtype)
        self.attn = Attention(cfg, layer_idx, lowp, attn_recipe, device, dtype)
        self.n2 = nn.RMSNorm(cfg.d_model, eps=1e-5, device=device, dtype=dtype)
        self.mlp = SwiGLU(cfg, lowp, mlp_recipe, device, dtype)

    def forward(self, x, cache=None, loop_idx: int = 0):
        h, new_cache = self.attn(self.n1(x), cache, loop_idx)
        x = x + h
        return x + self.mlp(self.n2(x)), new_cache


class BlackwellLM(nn.Module):
    def __init__(self, cfg: ModelConfig, precision: str = "fp8", device="cuda", dtype=torch.bfloat16):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        # FAIL LOUDLY on a requested-but-unavailable precision. Silently
        # downgrading to BF16 once produced an "fp8 vs nvfp4" comparison in which
        # BOTH arms were actually BF16 and came out identical to 0.1% -- a result
        # that looked like a finding and was an artefact. If the caller asked for
        # low precision, they must get it or hear why not.
        if precision in ("fp8", "nvfp4") and te is None:
            raise RuntimeError(
                f"precision={precision!r} requires NVIDIA Transformer Engine, which failed "
                f"to import ({_TE_IMPORT_ERROR}). Install it, or pass precision='bf16' "
                "explicitly to accept the slower path."
            )
        self.precision = precision
        self.attn_recipe, self.mlp_recipe = make_recipes(self.precision)
        lowp = self.attn_recipe is not None or self.mlp_recipe is not None
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model, device=device, dtype=dtype)
        self.blocks = nn.ModuleList([
            Block(cfg, i, lowp, self.attn_recipe, self.mlp_recipe, device, dtype)
            for i in range(cfg.n_layers)
        ])
        self.norm = nn.RMSNorm(cfg.d_model, eps=1e-5, device=device, dtype=dtype)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False, device=device, dtype=dtype)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight
        self.loop_embed = (
            nn.Embedding(cfg.max_loops, cfg.d_model, device=device, dtype=dtype)
            if cfg.loop_conditioning else None
        )
        self.apply(self._init)
        if self.loop_embed is not None:
            # start as a no-op so conditioning is learned, not imposed
            nn.init.zeros_(self.loop_embed.weight)
        if cfg.zero_init_residual:
            for blk in self.blocks:
                for proj in (blk.attn.out, blk.mlp.down):
                    w = getattr(proj.inner, "weight", None)
                    if w is not None:
                        nn.init.zeros_(w)

    @staticmethod
    def _init(m):
        # TE linears initialise themselves; only touch the native modules.
        if isinstance(m, (nn.Embedding, nn.Linear)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, ids, cache=None, n_loops: int | None = None):
        """`n_loops` overrides the configured loop count, which is what makes
        test-time compute scaling possible: loop more on hard inputs without
        touching the weights. The KV cache is indexed by APPLICATION, not by
        unique block, since each application attends over its own history."""
        loops = self.sample_loops() if n_loops is None else n_loops
        x = self.embed(ids)
        new_caches = []
        step = 0
        for loop_idx in range(loops):
            if self.loop_embed is not None:
                # which iteration this is; without it a shared block is blind to depth
                x = x + self.loop_embed.weight[min(loop_idx, self.cfg.max_loops - 1)]
            for blk in self.blocks:
                x, c = blk(x, None if cache is None else cache[step], loop_idx)
                new_caches.append(c)
                step += 1
        logits = self.lm_head(self.norm(x))
        return (logits, new_caches) if cache is not None else logits

    def sample_loops(self) -> int:
        """Training-time loop count. Sampling it (rather than fixing it) is what
        teaches the model to work at any depth, which is the precondition for
        scaling test-time compute by looping more. Eval always uses n_loops."""
        r = self.cfg.loop_sample
        if not self.training or r is None:
            return self.cfg.n_loops
        lo, hi = r
        return int(torch.randint(lo, hi + 1, (1,)).item())

    def n_params(self) -> int:
        seen, total = set(), 0
        for p in self.parameters():  # tied weights must not be double counted
            if id(p) not in seen:
                seen.add(id(p))
                total += p.numel()
        return total
