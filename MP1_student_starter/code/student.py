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


def build_model(config):
    return CopyGPT(config)
