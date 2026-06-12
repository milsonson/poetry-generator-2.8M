from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def apply_repetition_penalty(
    logits: torch.Tensor,
    generated: torch.Tensor,
    penalty: float = 1.0,
    window: int = 64,
    exempt_token_ids: Optional[Iterable[int]] = None,
) -> torch.Tensor:
    if penalty <= 1.0 or generated.numel() == 0:
        return logits

    exempt = set(exempt_token_ids or [])
    recent = generated[:, -window:] if window > 0 else generated
    for batch_idx in range(logits.size(0)):
        values, counts = torch.unique(recent[batch_idx], return_counts=True)
        for token, count in zip(values.tolist(), counts.tolist()):
            if token in exempt:
                continue
            factor = penalty ** int(count)
            current = logits[batch_idx, token]
            logits[batch_idx, token] = current / factor if current > 0 else current * factor
    return logits


def adaptive_temperature_from_logits(
    logits: torch.Tensor,
    base_temperature: float,
    target_entropy: float = 0.55,
    strength: float = 0.65,
    min_temperature: float = 0.55,
    max_temperature: float = 1.35,
) -> float:
    if base_temperature <= 0:
        return base_temperature
    probs = F.softmax(logits, dim=-1)
    entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1)
    max_entropy = math.log(logits.size(-1))
    normalized_entropy = float((entropy / max_entropy).mean().item())
    scale = math.exp((target_entropy - normalized_entropy) * strength)
    value = base_temperature * scale
    return max(min_temperature, min(max_temperature, value))


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 128
    n_layer: int = 4
    n_head: int = 4
    d_model: int = 128
    d_ff: int = 512
    dropout: float = 0.1

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        block_size: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert d_model % n_head == 0, "d_model must be divisible by n_head"
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.w_q = nn.Linear(d_model, d_model, bias=True)
        self.w_k = nn.Linear(d_model, d_model, bias=True)
        self.w_v = nn.Linear(d_model, d_model, bias=True)
        self.w_o = nn.Linear(d_model, d_model, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.block_size = block_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, C), C == d_model.

        Implements multi-head scaled dot-product attention with a causal mask.
        Query position i can only attend to key positions j <= i.
        """
        b, t, c = x.size()
        if t > self.block_size:
            raise ValueError(f"sequence length {t} exceeds block_size {self.block_size}")

        Q, K, V = self.w_q(x), self.w_k(x), self.w_v(x)
        Q = Q.view(b, t, self.n_head, self.d_head).transpose(1, 2)
        K = K.view(b, t, self.n_head, self.d_head).transpose(1, 2)
        V = V.view(b, t, self.n_head, self.d_head).transpose(1, 2)

        att = Q @ K.transpose(-2, -1) / math.sqrt(self.d_head)
        mask = torch.triu(
            torch.ones(t, t, device=x.device, dtype=torch.bool),
            diagonal=1,
        )
        att = att.masked_fill(mask, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)

        y = att @ V
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.w_o(y)


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network used inside each Transformer block.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout_p = float(dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ff(x)


class TransformerBlock(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(
            d_model=config.d_model,
            n_head=config.n_head,
            block_size=config.block_size,
            dropout=config.dropout,
        )
        self.ln_2 = nn.LayerNorm(config.d_model)
        self.ff = FeedForward(
            d_model=config.d_model,
            d_ff=config.d_ff,
            dropout=config.dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.ff(self.ln_2(x))
        return x


class PoetryTransformer(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.block_size, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layer)]
        )
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        loss_ignore_index: int = -100,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        b, t = idx.shape
        if t > self.config.block_size:
            raise ValueError(f"sequence length {t} exceeds block_size {self.config.block_size}")

        pos = torch.arange(0, t, dtype=torch.long, device=idx.device)
        x = self.token_embedding(idx) + self.position_embedding(pos)[None, :, :]
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=loss_ignore_index,
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        repetition_penalty: float = 1.0,
        repetition_window: int = 64,
        exempt_token_ids: Optional[Iterable[int]] = None,
        adaptive_temperature: bool = False,
        target_entropy: float = 0.55,
        temperature_strength: float = 0.65,
        min_temperature: float = 0.55,
        max_temperature: float = 1.35,
    ) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            logits = apply_repetition_penalty(
                logits,
                idx,
                penalty=repetition_penalty,
                window=repetition_window,
                exempt_token_ids=exempt_token_ids,
            )

            if temperature <= 0:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                current_temperature = temperature
                if adaptive_temperature:
                    current_temperature = adaptive_temperature_from_logits(
                        logits,
                        base_temperature=temperature,
                        target_entropy=target_entropy,
                        strength=temperature_strength,
                        min_temperature=min_temperature,
                        max_temperature=max_temperature,
                    )
                logits = logits / current_temperature
                if top_k is not None and top_k > 0:
                    values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits = logits.masked_fill(logits < values[:, [-1]], float("-inf"))
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, idx_next), dim=1)
        return idx
