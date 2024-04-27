import math
import torch
from torch.nn import functional as F
from torch.utils.cpp_extension import load
import argparse

# Load the CUDA kernel as a python module
minimal_flash = load(name='flash', sources=['src/main.cpp', 'src/flashattention.cu'], extra_cuda_cflags=['-O3'])

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
parser.add_argument('--seq_len', type=int, default=8192, help='Sequence length')
args = parser.parse_args()

batch_size = args.batch_size
n_head = 8
seq_len = args.seq_len
head_embd = 64

print(f"Using {batch_size} batch size, {n_head} heads, {seq_len} sequence length, {head_embd} head embedding size")

torch.cuda.empty_cache()

q = torch.randn(batch_size * n_head, seq_len, head_embd).cuda()
k = torch.randn(batch_size * n_head, seq_len, head_embd).cuda()
v = torch.randn(batch_size * n_head, seq_len, head_embd).cuda()
       

# Compare to Pytroch's matmul
def manual_attention(q, k):
    S = torch.matmul(q, k.transpose(-2, -1))#/math.sqrt(head_embd)
    A = F.softmax(S, dim=-1)
    O = torch.matmul(A, v)
    return O

print('=== profiling manual attention ===')
with torch.autograd.profiler.profile(use_cuda=True) as prof:
    manual_result = manual_attention(q, k)

print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))

print('=== profiling minimal flash attention === ')

with torch.autograd.profiler.profile(use_cuda=True) as prof:
    minimal_flash = minimal_flash.forward(q, k, v)

print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))

if torch.allclose(minimal_flash, manual_result, rtol=0, atol=1e-01):
    print('[Correctness] attn values sanity check: PASSED')
else:
    print('Correctness] attn values sanity check: FAILED')
    print(minimal_flash.cpu())
    print(manual_result.cpu())

