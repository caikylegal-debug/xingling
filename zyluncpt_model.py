import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


from dataclasses import dataclass

@dataclass
class ZylunConfig:
    vocab_size: int = 32000
    seq_len: int = 1024
    n_layers: int = 16
    d_model: int = 768
    n_heads: int = 12
    n_kv_heads: int = 4
    ffn_dim: int = 2048
    dropout: float = 0.0
    rope_theta: float = 10000.0
    rms_eps: float = 1e-6
    tie_embeddings: bool = True

cfg = ZylunConfig()


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return self.weight * x * torch.rsqrt(
            x.pow(2).mean(-1, keepdim=True) + self.eps
        )


def precompute_rope(seq_len: int, head_dim: int, theta: float, device=None):
    inv_freq = 1.0 / (
        theta ** (
            torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim
        )
    )
    t = torch.arange(seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(t, inv_freq)
    return freqs.cos()[None, :, None, :], freqs.sin()[None, :, None, :]


def apply_rope(x, cos, sin):
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]

    x_rope_even = x_even * cos - x_odd * sin
    x_rope_odd = x_even * sin + x_odd * cos

    return torch.stack((x_rope_even, x_rope_odd), dim=-1).flatten(-2)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ZylunConfig):
        super().__init__()

        assert cfg.d_model % cfg.n_heads == 0
        assert cfg.n_heads % cfg.n_kv_heads == 0

        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.dropout = cfg.dropout

        self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.n_heads * self.head_dim, cfg.d_model, bias=False)

    def forward(self, x, cos, sin):
        B, T, C = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim)

        q = apply_rope(q, cos[:, :T], sin[:, :T])
        k = apply_rope(k, cos[:, :T], sin[:, :T])

        if self.n_kv_heads != self.n_heads:
            repeat = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(repeat, dim=2)
            v = v.repeat_interleave(repeat, dim=2)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(y)


class SwiGLU(nn.Module):
    def __init__(self, cfg: ZylunConfig):
        super().__init__()

        self.w1 = nn.Linear(cfg.d_model, cfg.ffn_dim, bias=False)
        self.w2 = nn.Linear(cfg.ffn_dim, cfg.d_model, bias=False)
        self.w3 = nn.Linear(cfg.d_model, cfg.ffn_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, cfg: ZylunConfig):
        super().__init__()

        self.attn_norm = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.ffn_norm = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.attn = CausalSelfAttention(cfg)
        self.ffn = SwiGLU(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.attn_norm(x), cos, sin)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class ZylunCPT(nn.Module):
    def __init__(self, cfg: ZylunConfig):
        super().__init__()

        self.cfg = cfg

        self.tok_embeddings = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_embeddings.weight

        self.register_buffer("rope_cos", torch.empty(0), persistent=False)
        self.register_buffer("rope_sin", torch.empty(0), persistent=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _get_rope(self, T, device):
        if (
            self.rope_cos.numel() == 0
            or self.rope_cos.shape[1] < T
            or self.rope_cos.device != device
        ):
            cos, sin = precompute_rope(
                self.cfg.seq_len,
                self.cfg.d_model // self.cfg.n_heads,
                self.cfg.rope_theta,
                device=device,
            )
            self.rope_cos = cos
            self.rope_sin = sin

        return self.rope_cos[:, :T], self.rope_sin[:, :T]

    def forward(self, input_ids, labels=None):
        B, T = input_ids.shape

        assert T <= self.cfg.seq_len, f"Sequência {T} > seq_len {self.cfg.seq_len}"

        cos, sin = self._get_rope(T, input_ids.device)

        x = self.tok_embeddings(input_ids)

        for block in self.blocks:
            x = block(x, cos, sin)

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None

        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids,
        max_new_tokens=120,
        temperature=0.8,
        top_k=50,
        eos_id=None,
    ):
        self.eval()

        for _ in range(max_new_tokens):
            idx_cond = input_ids[:, -self.cfg.seq_len:]

            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]

            if temperature <= 0:
                next_id = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / max(temperature, 1e-5)

                if top_k is not None and top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("inf")

                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)

            input_ids = torch.cat([input_ids, next_id], dim=1)

            if eos_id is not None and next_id.item() == eos_id:
                break

        return input_ids
