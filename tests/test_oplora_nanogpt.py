import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from peft import get_peft_model
from peft.peft_model import PeftModel
from peft.tuners.lora.config import LoraConfig
from peft.utils.peft_types import TaskType


@dataclass
class NanoGPTConfig:
    vocab_size: int = 32
    block_size: int = 16
    n_layer: int = 2
    n_head: int = 2
    n_embd: int = 32
    dropout: float = 0.0
    model_type: str = "nanogpt"


class CausalSelfAttention(nn.Module):
    def __init__(self, config: NanoGPTConfig) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0, "embedding dimension must be divisible by number of heads"
        self.n_head = config.n_head
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        bias = torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size)
        self.register_buffer("bias", bias, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.chunk(3, dim=2)

        head_dim = C // self.n_head
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(head_dim))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, config: NanoGPTConfig) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.c_fc(x))
        x = self.c_proj(x)
        return self.dropout(x)


class Block(nn.Module):
    def __init__(self, config: NanoGPTConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class NanoGPT(nn.Module):
    def __init__(self, config: NanoGPTConfig) -> None:
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.n_embd),
                "wpe": nn.Embedding(config.block_size, config.n_embd),
                "drop": nn.Dropout(config.dropout),
                "h": nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                "ln_f": nn.LayerNorm(config.n_embd),
            }
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer["wte"].weight = self.lm_head.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ):
        _ = kwargs
        _ = attention_mask
        idx = input_ids
        B, T = idx.shape
        device = idx.device
        pos = torch.arange(0, T, dtype=torch.long, device=device).unsqueeze(0)
        tok_emb = self.transformer["wte"](idx)
        pos_emb = self.transformer["wpe"](pos)
        x = tok_emb + pos_emb
        x = self.transformer["drop"](x)
        for block in self.transformer["h"]:
            x = block(x)
        x = self.transformer["ln_f"](x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(B * T, -1), labels.view(B * T))
        return logits, loss

    def prepare_inputs_for_generation(self, input_ids: torch.Tensor, **kwargs):
        return {"input_ids": input_ids, **kwargs}


def test_oplora_with_nanogpt_forward_backward(tmp_path):
    torch.manual_seed(0)
    config = NanoGPTConfig(vocab_size=48, block_size=12, n_layer=2, n_head=4, n_embd=32)
    model = NanoGPT(config)

    lora_config = LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=["c_attn", "c_proj", "c_fc"],
        use_oplora=True,
        op_lora_k=4,
        task_type=TaskType.CAUSAL_LM,
    )

    peft_model: PeftModel = get_peft_model(model, lora_config)
    peft_model.train()

    optimizer = torch.optim.Adam(peft_model.parameters(), lr=1e-3)

    inputs = torch.randint(0, config.vocab_size, (2, config.block_size))
    targets = torch.randint(0, config.vocab_size, (2, config.block_size))

    outputs = peft_model(input_ids=inputs, labels=targets)
    if isinstance(outputs, tuple):
        if len(outputs) == 0:
            raise AssertionError("Expected non-empty output tuple from NanoGPT")
        logits = outputs[0]
        loss = outputs[1] if len(outputs) > 1 else None
    else:
        logits = outputs.logits
        loss = outputs.loss
    assert logits.shape == (2, config.block_size, config.vocab_size)
    assert loss is not None and torch.isfinite(loss)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    adapter_name = peft_model.active_adapters[0]
    first_block = peft_model.base_model.model.transformer["h"][0]
    modules_to_check = [
        first_block.attn.c_attn,
        first_block.attn.c_proj,
        first_block.mlp.c_fc,
        first_block.mlp.c_proj,
    ]

    for module in modules_to_check:
        assert module.use_oplora.get(adapter_name, False)
        assert module.oplora_rank[adapter_name] == 4

        projected_A, projected_B, projected_bias = module._get_projected_lora_weights(
            adapter_name,
            module.lora_A[adapter_name].weight,
            module.lora_B[adapter_name].weight,
            module.lora_B[adapter_name].bias,
        )
        assert projected_bias is None
        U = module._oplora_left_vectors[adapter_name]
        V = module._oplora_right_vectors[adapter_name]

        with torch.no_grad():
            left_projection = U.transpose(0, 1) @ projected_B
            right_projection = projected_A @ V
            delta_weight = projected_B @ projected_A
            interference = U.transpose(0, 1) @ delta_weight @ V

        assert torch.allclose(left_projection, torch.zeros_like(left_projection), atol=1e-4, rtol=1e-4)
        assert torch.allclose(right_projection, torch.zeros_like(right_projection), atol=1e-4, rtol=1e-4)
        assert torch.allclose(interference, torch.zeros_like(interference), atol=1e-4, rtol=1e-4)

        alignment = module.compute_subspace_alignment(adapter_name)
        assert 0.0 <= alignment <= 1.0

