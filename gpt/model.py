import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 6000
    block_size: int = 128       # max context length
    n_layer: int = 6
    n_embd: int = 256
    n_head: int = 8              # query heads
    n_kv_head: int = 2           # key/value heads (GQA); n_head % n_kv_head == 0
    mlp_hidden: int = 704        # SwiGLU hidden size (~ 8/3 * n_embd, rounded)
    dropout: float = 0.0
    rope_theta: float = 10000.0


class RMSNorm(nn.Module):
    """Root-mean-square layer norm -- no mean-centering, no bias term."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


def precompute_rope(head_dim: int, max_seq_len: int, theta: float, device, dtype):
    """Precompute the RoPE cos/sin tables for every position up to max_seq_len."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(max_seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)  # [seq, head_dim/2]
    emb = torch.cat([freqs, freqs], dim=-1)  # [seq, head_dim]
    return emb.cos().to(dtype), emb.sin().to(dtype)


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(q, k, cos, sin):
    # q, k: [B, n_head, T, head_dim]; cos/sin: [T, head_dim]
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
    return q_rot, k_rot


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with grouped-query attention and RoPE."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        assert cfg.n_head % cfg.n_kv_head == 0
        self.n_head = cfg.n_head
        self.n_kv_head = cfg.n_kv_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.group_size = cfg.n_head // cfg.n_kv_head

        self.q_proj = nn.Linear(cfg.n_embd, cfg.n_head * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.n_embd, cfg.n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.n_embd, cfg.n_kv_head * self.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.n_head * self.head_dim, cfg.n_embd, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        q, k = apply_rope(q, k, cos, sin)

        # Expand KV heads up to the number of query heads (grouped-query attention).
        if self.group_size > 1:
            k = k.repeat_interleave(self.group_size, dim=1)
            v = v.repeat_interleave(self.group_size, dim=1)

        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(out)


class SwiGLU(nn.Module):
    """SwiGLU feed-forward block: silu(W1 x) * (W3 x), then W2 back to n_embd."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.w1 = nn.Linear(cfg.n_embd, cfg.mlp_hidden, bias=False)
        self.w3 = nn.Linear(cfg.n_embd, cfg.mlp_hidden, bias=False)
        self.w2 = nn.Linear(cfg.mlp_hidden, cfg.n_embd, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = RMSNorm(cfg.n_embd)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.mlp(self.norm2(x))
        return x


class DequantizingEmbedding(nn.Module):
    """Drop-in replacement for nn.Embedding backed by an int8 weight table.

    Used for the "matmul + embedding" quantization variant: the tied
    embedding/output-head table is the single largest tensor in a model
    this size, so quantizing it too (on top of the Linear layers) matters
    a lot for final size, at a small, measured accuracy cost.
    """

    def __init__(self, int8_weight: torch.Tensor, scale: float):
        super().__init__()
        self.register_buffer("int8_weight", int8_weight)
        self.scale = scale

    def forward(self, idx):
        return self.int8_weight[idx].float() * self.scale

    @staticmethod
    def quantize(weight: torch.Tensor) -> "DequantizingEmbedding":
        scale = weight.abs().max().item() / 127.0
        int8_weight = torch.clamp(torch.round(weight / scale), -127, 127).to(torch.int8)
        return DequantizingEmbedding(int8_weight, scale)


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm_f = RMSNorm(cfg.n_embd)

        # Tied embeddings: the output head reuses the input embedding matrix,
        # saving vocab_size * n_embd parameters (significant at small scale).
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight

        head_dim = cfg.n_embd // cfg.n_head
        cos, sin = precompute_rope(head_dim, cfg.block_size, cfg.rope_theta, device="cpu", dtype=torch.float32)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding=True):
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok_emb.weight.numel()
        return n

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"sequence length {T} exceeds block_size {self.cfg.block_size}"

        x = self.tok_emb(idx)
        cos = self.rope_cos[:T].to(x.dtype)
        sin = self.rope_sin[:T].to(x.dtype)

        for block in self.blocks:
            x = block(x, cos, sin)
        x = self.norm_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def next_token_logits(self, idx):
        """Logits for the very next token given a (possibly truncated) context."""
        idx = idx[:, -self.cfg.block_size :]
        logits, _ = self(idx)
        return logits[:, -1, :]
