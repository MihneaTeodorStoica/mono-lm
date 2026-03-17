from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(slots=True)
class TransformerConfig:
    vocab_size: int
    context_length: int
    d_model: int
    num_layers: int
    num_heads: int
    ffw_multiplier: float = 4.0
    dropout: float = 0.1
    bias: bool = True

    def to_dict(self) -> dict[str, int | float | bool]:
        return asdict(self)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        if config.d_model % config.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")
        self.num_heads = config.num_heads
        self.head_dim = config.d_model // config.num_heads
        self.dropout = config.dropout
        self.qkv = nn.Linear(config.d_model, config.d_model * 3, bias=config.bias)
        self.proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)

    def forward(self, x: Tensor) -> Tensor:
        batch_size, sequence_length, channels = x.shape
        qkv = self.qkv(x)
        query, key, value = qkv.chunk(3, dim=-1)
        query = query.view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)
        attn_output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, sequence_length, channels)
        return self.proj(attn_output)


class FeedForward(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        hidden_size = int(config.ffw_multiplier * config.d_model)
        self.net = nn.Sequential(
            nn.Linear(config.d_model, hidden_size, bias=config.bias),
            nn.GELU(),
            nn.Linear(hidden_size, config.d_model, bias=config.bias),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.d_model)
        self.ffw = FeedForward(config)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.ffw(self.ln_2(x))
        return x


class MonoLMModel(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.context_length, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)

    def forward(self, indices: Tensor, targets: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        _, sequence_length = indices.shape
        if sequence_length > self.config.context_length:
            raise ValueError(
                f"Sequence length {sequence_length} exceeds model context length {self.config.context_length}."
            )

        positions = torch.arange(sequence_length, device=indices.device, dtype=torch.long)
        x = self.token_embedding(indices) + self.position_embedding(positions)[None, :, :]
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        indices: Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> Tensor:
        output = indices
        for _ in range(max_new_tokens):
            model_input = output[:, -self.config.context_length :]
            logits, _ = self(model_input)
            logits = logits[:, -1, :]
            if temperature <= 0:
                next_index = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None and top_k > 0:
                    values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits = logits.masked_fill(logits < values[:, [-1]], float("-inf"))
                probabilities = F.softmax(logits, dim=-1)
                next_index = torch.multinomial(probabilities, num_samples=1)
            output = torch.cat([output, next_index], dim=1)
        return output

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
