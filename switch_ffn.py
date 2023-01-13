# influenced by labml.ai https://nn.labml.ai/transformers/switch/index.html
import torch
import torch.nn as nn
from .moe import Expert


class SwitchFFN(nn.Module):
    def __init__(
        self,
        n_experts: int,
        d_model: int,
        capacity_factor: int,
        drop_tokens: bool,
        is_scale_prob: bool,
    ):
        super().__init__(self)
        self.n_experts = n_experts
        self.d_model = d_model
        self.capacity_factor = capacity_factor
        self.drop_tokens = drop_tokens
        self.is_scale_prob = is_scale_prob
        self.experts = nn.ModuleList([Expert(d_model, 2) for _ in range(n_experts)])
        self.router = nn.Linear(d_model, n_experts)

    def forward(self, x):
        # x - [seq_len, batch_size, d_model]
        pass


if __name__ == "__main__":
    pass
