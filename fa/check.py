import numpy as np
import torch

B = 1
N = 1024
d = 64
shape = f'{B}x{N}x{d}xf16'
torch.manual_seed(7)

def compute_attention_reference(q, k, v):
    kT = torch.permute(k, (0, 2, 1))
    s = torch.matmul(q, kT)
    p = torch.nn.Softmax(dim=2)(s)
    return torch.matmul(p, v)

def construct_inputs(B, N, d):
    q = torch.rand((B, N, d), dtype=torch.float16).cuda() / 5.0
    k = torch.rand((B, N, d), dtype=torch.float16).cuda() / 5.0
    v = torch.rand((B, N, d), dtype=torch.float16).cuda() / 5.0
    return q, k, v

q, k, v = construct_inputs(B, N, d)
output = compute_attention_reference(q, k, v)

# Write matrices
with open(f'query_{shape}.npy', 'wb') as f:
    np.save(f, q.detach().cpu().numpy())
with open(f'key_{shape}.npy', 'wb') as f:
    np.save(f, k.detach().cpu().numpy())
with open(f'value_{shape}.npy', 'wb') as f:
    np.save(f, v.detach().cpu().numpy())
with open(f'output_{shape}.npy', 'wb') as f:
    np.save(f, output.detach().cpu().numpy())
