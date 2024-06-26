## FlashAttention.C
A minimal Flashattention implementation in pure Cuda C. It consistently performs as fast as Pytorch or faster for settings where the sequence length is limiting.

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

## Experimental:
Instead of caching the whole Q one can get some SMEM size savings by setting CACHE_Q to 0 in flashattention.cu and now must change uncomment line 150. This results in more SMEM stores but the gained occupancy by reducing the size of the SMEM has shown increase in performance in our tests.

## Todo:
- Optimize the Softmax part to let threads collaborate on computing the row-wise max instead. Getting down from linea complexity to log2 complexity with minimal overhead should be easily possible. As the batchsize are small overhead needs to be very low to make it work.
- Optimize for lower SMEM footprint (maybe don't cache full Value rows): DONE (20% speedup on 3060)
- Optimize for using larger TN,TM > 4. For this probably need to optimize resister usage do lower because right now the kernel does not execute on 8x8 register tiles
