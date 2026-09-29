"""Regularized GPT with a stateless, strictly causal suffix-copy mixture.

The neural component is trained from scratch. Copying uses only successors that
are already visible in the current input prefix; it learns no evaluation assets.
Set copy_weights to all zeros for an exact neural-only mechanism ablation.
"""
import torch
from torch import nn
from torch.nn import functional as F
from model import GPT, Block


class RegularizedBlock(Block):
    def __init__(self, width, heads, dropout):
        super().__init__(width, heads)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch, length, width = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(
            batch, length, 3, self.heads, width // self.heads).permute(2, 0, 3, 1, 4)
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.dropout(self.proj(attended.transpose(1, 2).reshape(batch, length, width)))
        return x + self.dropout(self.mlp(self.norm2(x)))


class CopyGPT(GPT):
    def __init__(self, config):
        super().__init__(config)
        self.copy_weights = tuple(float(w) for w in config.get('copy_weights', [.05, .15, .3]))
        if len(self.copy_weights) > 8 or any(not 0 <= w < 1 for w in self.copy_weights):
            raise ValueError('Use at most eight suffix weights, each in [0, 1).')
        dropout = float(config.get('dropout', 0.))
        self.embedding_dropout = nn.Dropout(dropout)
        if dropout:
            # Preserve parameter names and initialization; dropout has no weights.
            for i, old in enumerate(self.blocks):
                block = RegularizedBlock(config['width'], config['heads'], dropout)
                block.load_state_dict(old.state_dict())
                self.blocks[i] = block

    def features(self, ids):
        x = self.embedding_dropout(self.token(ids) + self.pos(torch.arange(ids.shape[1], device=ids.device)))
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

    def predict_log_probs(self, ids):
        neural = F.log_softmax(self(ids).float(), dim=-1)
        batch, length = ids.shape
        if length < 2 or not any(self.copy_weights):
            return neural

        # Key j names a context ending at j; its successor ids[j+1] is visible
        # at query t iff j < t. This strict inequality is the causality boundary.
        positions = torch.arange(length, device=ids.device)
        equal = ids[:, :, None] == ids[:, None, :]
        matches = equal & (positions[:, None] > positions[None, :])
        chosen = torch.zeros_like(matches, dtype=torch.float32)
        mass = torch.zeros((batch, length, 1), device=ids.device)
        for order, weight in enumerate(self.copy_weights, start=1):
            if order > length:
                break
            if order > 1:
                previous_equal = torch.zeros_like(equal)
                offset = order - 1
                previous_equal[:, offset:, offset:] = equal[:, :-offset, :-offset]
                matches = matches & previous_equal
            count = matches.sum(-1, keepdim=True)
            available = count > 0
            distribution = matches.float() / count.clamp_min(1)
            chosen = torch.where(available, distribution, chosen)
            mass = torch.where(available, float(weight), mass)

        # Last key can never have a visible successor. Discard it explicitly.
        probabilities = neural.exp() * (1 - mass)
        successors = ids[:, None, 1:].expand(batch, length, length - 1)
        probabilities.scatter_add_(-1, successors, chosen[:, :, :-1] * mass)
        mixed = probabilities.clamp_min(torch.finfo(torch.float32).tiny).log()
        return torch.where(mass > 0, mixed, neural)


def apply_rotary(x, positions):
    """Rotate adjacent channel pairs; attention depends on relative positions."""
    dimensions = x.shape[-1]
    frequencies = 10000. ** (-torch.arange(0, dimensions, 2, device=x.device).float() / dimensions)
    angles = positions.float()[:, None] * frequencies[None, :]
    cosine, sine = angles.cos(), angles.sin()
    even, odd = x[..., ::2], x[..., 1::2]
    return torch.stack((even * cosine - odd * sine,
                        even * sine + odd * cosine), dim=-1).flatten(-2)


class RotaryBlock(nn.Module):
    def __init__(self, width, heads, dropout):
        super().__init__()
        self.heads = heads
        hidden = 64 * ((8 * width + 191) // 192)
        self.norm1, self.norm2 = nn.RMSNorm(width, eps=1e-5), nn.RMSNorm(width, eps=1e-5)
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.proj = nn.Linear(width, width, bias=False)
        self.gate_up = nn.Linear(width, 2 * hidden, bias=False)
        self.down = nn.Linear(hidden, width, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch, length, width = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(
            batch, length, 3, self.heads, width // self.heads).permute(2, 0, 3, 1, 4)
        positions = torch.arange(length, device=x.device)
        attended = F.scaled_dot_product_attention(
            apply_rotary(q, positions), apply_rotary(k, positions), v, is_causal=True)
        x = x + self.dropout(self.proj(attended.transpose(1, 2).reshape(batch, length, width)))
        gate, up = self.gate_up(self.norm2(x)).chunk(2, dim=-1)
        return x + self.dropout(self.down(F.silu(gate) * up))


class RotaryCopyGPT(CopyGPT):
    """From-scratch RoPE/RMSNorm/SwiGLU backbone with the same causal mixture."""
    def __init__(self, config):
        super().__init__(config)
        if (config['width'] // config['heads']) % 2:
            raise ValueError('Rotary attention requires an even head dimension.')
        del self.pos
        self.blocks = nn.ModuleList([
            RotaryBlock(config['width'], config['heads'], config.get('dropout', 0.))
            for _ in range(config['depth'])])
        self.blocks.apply(self.initialize)
        self.norm = nn.RMSNorm(config['width'], eps=1e-5)
        for block in self.blocks:
            nn.init.normal_(block.proj.weight, std=.02 / (2 * config['depth']) ** .5)
            nn.init.normal_(block.down.weight, std=.02 / (2 * config['depth']) ** .5)

    def features(self, ids):
        x = self.embedding_dropout(self.token(ids))
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


def build_model(config):
    if config.get('architecture') == 'rotary':
        return RotaryCopyGPT(config)
    return CopyGPT(config)
