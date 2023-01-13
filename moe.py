"""
Implement 3 PyTorch modules that do top-1 MoE with the following routing options (https://arxiv.org/abs/2209.01667 Fig.6):

- Tokens choosing the expert

- Expert choosing the token

- A global allocation of tokens to experts (S-BASE)

The implementation does not need to be fast, but it should use the theoretical flops. (No computing everything as dense and then multiplying by a mask)

Additionally, add an option to use the Avg-K strategy from: https://openreview.net/pdf?id=lX478WYy0Up to obtain the scores.

Additionally, add an option to user more than one expert!

"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    def __init__(self, hidden_size, expansion_factor):
        super().__init__()
        self.hidden_size = hidden_size
        self.expansion_factor = expansion_factor
        self.fc1 = nn.Linear(hidden_size, hidden_size * expansion_factor)
        self.fc2 = nn.Linear(hidden_size * expansion_factor, hidden_size)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class MOELayer1(nn.Module):
    pass


class MOELayer2(nn.Module):
    pass


class MOELayer3(nn.Module):
    pass


@torch.no_grad()
def set_avg_k_gate(moe_layer):
    pass


if __name__ == "__main__":
    pass
