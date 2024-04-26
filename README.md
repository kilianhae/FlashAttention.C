## FlashAttention.C
A minimal FLashattention implementation in pure Cuda C. It consistently performs as fast as pytorch or faster for settings where the sequence length is limiting.

Inspired by recent efforts like: [flashattention minimal](https://github.com/tspeterkim/flash-attention-minimal.git), the goal of this project is to provide a readable implementation in pure Cuda, whilst also being fast and scalable.

## Implementation:
Right now I have implemented two kernels, one is a naive tiling approach that suffers from SMEM stalls. The second uses thread coarsening (tiling on a single thread level in its registers) and reduces the SMEM stalls effectively.

## Benchmarks:
These are results profiled on a RTX 3060:
- Batchsize=2, Heads=8, Head Dim. = 64, Sequence Length=8192: Pytorch=120 ms, Ours=119 ms
- Batchsize=2, Heads=8, Head Dim. = 32, Sequence Length=8192: Pytorch=117 ms, Ours=62 ms
- Batchsize=8, Heads=16, Head Dim. = 64, Sequence Length=1024: Pytorch=24 ms, Ours=22 ms
- Batchsize=8, Heads=16, Head Dim. = 32, Sequence Length=1024: Pytorch=22 ms, Ours=15 ms

## Run:
To test against pytorch, run:
```
python bench_flashattention.py
```
or if you want to run it without pytorch wrapping run:
```
nvcc test.cu -o test
./test
```
