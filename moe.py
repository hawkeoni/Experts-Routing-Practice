"""
Implement 3 PyTorch modules that do top-1 MoE with the following routing options (https://arxiv.org/abs/2209.01667 Fig.6):
- Tokens choosing the expert
- Expert choosing the token
- A global allocation of tokens to experts (S-BASE)

The implementation does not need to be fast, but it should use the theoretical flops. (No computing everything as dense and then multiplying by a mask)
Additionally, add an option to use the Avg-K strategy from: https://openreview.net/pdf?id=lX478WYy0Up to obtain the scores.
Additionally, add an option to user more than one expert!

"""
from argparse import ArgumentParser

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


class MOEBase(nn.Module):

    def __init__(self, hidden_size: int, expansion_factor: float, n_experts: int, dry_run: bool):
        super().__init__()
        self.n_experts = n_experts
        self.experts = nn.ModuleList([Expert(hidden_size, expansion_factor) for _ in range(n_experts)])
        self.router = nn.Linear(hidden_size, n_experts, bias=False)
        self.dry_run = dry_run
    
    def forward(self, x):
        pass


class MOELayer1(MOEBase):
    """
    Tokens Choosing The experts
    """
    def forward(self, x):
        # x - [batch, seq_len, d_model]
        expert_scores = self.router(x)  # [batch, seq_len, n_experts]
        expert_norm = torch.softmax(expert_scores, dim=2)
        probs, indices = torch.max(expert_norm, dim=2)
        output = torch.zeros_like(x)
        for i in range(self.n_experts):
            expert_indices = (indices == i).nonzero(as_tuple=True)
            if not len(expert_indices[0]):
                continue
            expert_inputs = x[expert_indices]
            if self.dry_run:
                expert_output = expert_inputs
            else:
                expert_output = self.experts[i](expert_inputs)
            output[expert_indices] = expert_output

        if not args.dry_run:
            output = output * probs.unsqueeze(2)
        return output



class MOELayer2(MOEBase):
    pass


class MOELayer3(MOEBase):
    pass


@torch.no_grad()
def set_avg_k_gate(moe_layer):
    pass


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="""Use this option to check correctness of scatters/gathers of tensors. 
        Loss should always be 0, because we do not apply experts. This option disables 
        loss calculation and allows us to check the validity of our tensor gathering/scattering code.""",
    )
    parser.add_argument("--seq-len", type=int, default=5)
    parser.add_argument("--n-experts", type=int, default=11)
    parser.add_argument("--scale-prob", action="store_true", default=False)
    parser.add_argument("--model", type=str, choices=["tokens", "experts", "global"], required=True, help="Define who is choosing the route.")
    parser.add_argument("--capacity", type=float, default=1.0)
    args = parser.parse_args()
    batch_size = 3
    seq_len = args.seq_len
    d_model = 17
    n_experts = args.n_experts
    if args.model == "tokens":
        ffn = MOELayer1(d_model, 2, n_experts, args.dry_run)
    elif args.model == "experts":
        ffn = MOELayer2(d_model, 2, n_experts, args.dry_run)
    elif args.model == "global":
        ffn = MOELayer3(d_model, 2, n_experts, args.dry_run)
    torch.use_deterministic_algorithms(True)
    torch.random.manual_seed(12)
    x = torch.rand(batch_size, seq_len, d_model)
    for param_name, param in ffn.named_parameters():
        if len(param.shape) > 1:
            torch.nn.init.xavier_normal_(param)
        else:
            torch.nn.init.ones_(param)
    
    optimizer = torch.optim.SGD(ffn.parameters(), lr=1e-5, momentum=0.9)
    router_pre = ffn.router.weight.detach().clone()
    for epoch in range(1, 10001):
        y = ffn(x)
        loss = (x - y).abs().sum()
        if not args.dry_run:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        else:
            assert loss.item() == 0, loss.item()
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, loss: {loss.item()}")

    assert (args.dry_run or router_pre != ffn.router.weight).all(), "Gradients did not flow through router!"
