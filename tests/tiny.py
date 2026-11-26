"""A tiny random causal language model in plain torch, named like a Hugging Face decoder: linear layers are
``model.layers.<i>.self_attn.{q,k,v,o}_proj`` and ``model.layers.<i>.mlp.{gate,up,down}_proj``, RMS norms are
``input_layernorm`` / ``post_attention_layernorm``; ``forward(input_ids=...)`` returns an object with ``.logits``."""
from types import SimpleNamespace

import torch
import torch.nn.functional as F

VOCAB = 50
HIDDEN = 32
FFN = 64
LINEARS_PER_BLOCK = 7


class RMSNorm(torch.nn.Module):
    def __init__(self, n):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(n))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.weight


class Attn(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = torch.nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.k_proj = torch.nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.v_proj = torch.nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.o_proj = torch.nn.Linear(HIDDEN, HIDDEN, bias=False)

    def forward(self, x):
        a = F.scaled_dot_product_attention(self.q_proj(x)[:, None], self.k_proj(x)[:, None], self.v_proj(x)[:, None], is_causal=True)[:, 0]
        return self.o_proj(a)


class Mlp(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = torch.nn.Linear(HIDDEN, FFN, bias=False)
        self.up_proj = torch.nn.Linear(HIDDEN, FFN, bias=False)
        self.down_proj = torch.nn.Linear(FFN, HIDDEN, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = Attn()
        self.mlp = Mlp()
        self.input_layernorm = RMSNorm(HIDDEN)
        self.post_attention_layernorm = RMSNorm(HIDDEN)

    def forward(self, x):
        x = x + self.self_attn(self.input_layernorm(x))
        return x + self.mlp(self.post_attention_layernorm(x))


class Inner(torch.nn.Module):
    def __init__(self, n_layers):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(VOCAB, HIDDEN)
        self.layers = torch.nn.ModuleList([Block() for _ in range(n_layers)])


class TinyLM(torch.nn.Module):
    def __init__(self, n_layers=2):
        super().__init__()
        self.model = Inner(n_layers)
        self.lm_head = torch.nn.Linear(HIDDEN, VOCAB, bias=False)
        self.config = SimpleNamespace(hidden_size=HIDDEN, num_hidden_layers=n_layers)

    def forward(self, input_ids):
        x = self.model.embed_tokens(input_ids)
        for blk in self.model.layers:
            x = blk(x)
        return SimpleNamespace(logits=self.lm_head(x))


def make_model(seed=0, n_layers=2):
    torch.manual_seed(seed)
    m = TinyLM(n_layers).eval()
    with torch.no_grad():
        for blk in m.model.layers:
            blk.input_layernorm.weight.copy_(0.5 + torch.rand(HIDDEN))
            blk.post_attention_layernorm.weight.copy_(0.5 + torch.rand(HIDDEN))
    return m


def make_windows(n, length, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.randint(0, VOCAB, (length,), generator=g).tolist() for _ in range(n)]


class ByteTokenizer:
    """Maps each character to its code point modulo the tiny vocabulary; same call shape as a Hugging Face tokenizer."""

    def __call__(self, text, add_special_tokens=False):
        return SimpleNamespace(input_ids=[ord(c) % VOCAB for c in text])
