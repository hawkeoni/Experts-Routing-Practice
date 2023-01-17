# influenced by labml.ai https://nn.labml.ai/transformers/switch/index.html
from argparse import ArgumentParser

import torch
import torch.nn as nn
from moe import Expert


class SwitchFFNSimplified(nn.Module):
    def __init__(
        self,
        n_experts: int,
        d_model: int,
        capacity_factor: int,
        drop_tokens: bool,
        is_scale_prob: bool,
        dry_run: bool = False,
    ):
        super().__init__()
        self.n_experts = n_experts
        self.d_model = d_model
        self.capacity_factor = capacity_factor
        self.drop_tokens = drop_tokens
        self.is_scale_prob = is_scale_prob
        self.experts = nn.ModuleList([Expert(d_model, 2) for _ in range(n_experts)])
        self.router = nn.Linear(d_model, n_experts)
        self.dry_run = dry_run

    def forward(self, x):
        # x - [batch_size, seq_len, d_model]
        expert_scores = torch.softmax(
            self.router(x), dim=2
        )  # [batch_size, seq_len, n_experts]
        expert_probs, expert_idx = torch.max(expert_scores, dim=2)  # [batch_size, seq_len]
        output = torch.zeros_like(x)
        for expert_id in range(self.n_experts):
            mask = (expert_idx == expert_id).unsqueeze(2)  # batch_size, seq_len, 1
            # NB: can use x[mask] without using `masked_select` & `masked_scatter`!!
            expert_inp = x.masked_select(mask).view(
                -1, self.d_model
            )  # n_expert_tokens, d_model
            if not self.dry_run:
                expert_out = self.experts[expert_id](
                    expert_inp
                )  # n_expert_tokens, d_model
            else:
                expert_out = expert_inp
            output.masked_scatter_(mask, expert_out)
        if self.is_scale_prob:
            output = output * expert_probs.unsqueeze(2)
        else:
            # NB: this is a hack to pass the losses to the router
            output = output * (expert_probs / expert_probs.detach()).unsqueeze(2)

        return output



class SwitchFFNLabml(SwitchFFNSimplified):

    def forward(self, x):
        # x - [batch_size, seq_len, d_model]
        batch_size, seq_len, d_model = x.shape
        x = x.view(-1, d_model)
        capacity = int(x.size(0) * self.capacity_factor)
        expert_scores = torch.softmax(
            self.router(x), dim=1
        )  # [batch_size * seq_len, n_experts]
        expert_probs, expert_idx = torch.max(expert_scores, dim=1)  # [batch_size * seq_len]
        output = torch.zeros_like(x)
        expert_index_list = [(expert_idx == expert_id).nonzero(as_tuple=True)[0] for expert_id in range(self.n_experts)]
        dropped = []
        if self.drop_tokens:
            for i in range(self.n_experts):
                if len(expert_index_list[i]) <= capacity:
                    continue
                expert_index_list[i] = expert_index_list[i][torch.randperm(len(expert_index_list[i]))]
                dropped.append(expert_index_list[i][capacity:])
                expert_index_list[i] = expert_index_list[i][:capacity]
        
        if not self.dry_run:
            expert_output = [self.experts[i](x[expert_index_list[i]]) for i in range(self.n_experts)]
        else:
            expert_output = [x[expert_index_list[i]] for i in range(self.n_experts)]
        for i in range(self.n_experts):
            output[expert_index_list[i]] = expert_output[i]
        if dropped:
            dropped = torch.cat(dropped)
            output[dropped] = x[dropped]

        if self.is_scale_prob:
            output = output * expert_probs.unsqueeze(1)
        else:
            # NB: this is a hack to pass the losses to the router
            output = output * (expert_probs / expert_probs.detach()).unsqueeze(1)
        return output.view(batch_size, seq_len, d_model)


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
    parser.add_argument("--seq-len", type=int, default=2)
    parser.add_argument("--n-experts", type=int, default=11)
    parser.add_argument("--scale-prob", action="store_true", default=False)
    parser.add_argument("--model", type=str, choices=["simplified", "labml"], default="simplified")
    parser.add_argument("--capacity", type=float, default=1.0)
    args = parser.parse_args()
    batch_size = 3
    seq_len = args.seq_len
    d_model = 3
    n_experts = args.n_experts
    if args.model == "simplified":
        ffn = SwitchFFNSimplified(n_experts, d_model, args.capacity, True, args.scale_prob, args.dry_run)
    else:
        ffn = SwitchFFNLabml(n_experts, d_model, args.capacity, True, args.scale_prob, args.dry_run)
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
