## FlashAttention.C
A minimal FLashattention implementation in Cuda C. \
The goal of this project is to provide a readable implementation in pure Cuda as a playground to test modifications and optimizations to flash attention.\
Although being faster than native Pytorch is the goal.
## Implementation:
Right now I have implemented a tiled and a tiled + thread coarsened kernel, the second has a weird race-condition somewhere as certain rows of the output are corrupted, but the speed up is noticeable. The first kernel is clearly SMEM bound as it requires quite a lot SMEM loads which the thread coarsening reduces linearly. Generally, the kernels are very SMEM-capacity bound which affects the overall latency hiding and the occupancy.
## Run:
To test run:
```
python bench_flashattention.py
```
or if you want to run it without pytorch wrapping run:
```
nvcc test.cu -o test
./test
```
