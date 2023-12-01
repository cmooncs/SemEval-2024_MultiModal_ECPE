import torch

tens = torch.randn(4, 10, 3)
print(tens)
tens = tens.unsqueeze(1)
print(tens)

W = torch.randn(2, 3, 6)

h = torch.matmul(tens, W)
print(h.size())

a_src = torch.randn(2, 6, 1)

attn_src = torch.matmul(h, a_src)
print(attn_src.size())
