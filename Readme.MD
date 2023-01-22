# Routing Experts Practice
Implementations of two strategies for top-1 expert routing from [A Review of Sparse Expert Models in Deep Learning](https://arxiv.org/abs/2209.01667) Fig.6:
* Tokens choosing experts from [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538) and 
[GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding](https://arxiv.org/abs/2006.16668)
* Experts choosing tokens from [Mixture-of-Experts with Expert Choice Routing](https://arxiv.org/abs/2202.09368)

The implementations are written for a single process, there are no communications and it just serves as a demonstration for complex tensor indexing.

# How to Run
Two routing experts are in the file `moe.py`.

```
usage: moe.py [-h] [--dry-run] [--seq-len SEQ_LEN] [--n-experts N_EXPERTS] [--scale-prob] --model
              {tokens,experts} [--capacity CAPACITY]

optional arguments:
  -h, --help            show this help message and exit
  --dry-run             Use this option to check correctness of scatters/gathers of tensors. Loss
                        should always be 0, because we do not apply experts. This option disables loss
                        calculation and allows us to check the validity of our tensor
                        gathering/scattering code.
  --seq-len SEQ_LEN
  --n-experts N_EXPERTS
  --scale-prob
  --model {tokens,experts}
                        Define who is choosing the route.
```