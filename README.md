## FlashAttention.C
A minimal FLashattention implementation in Cuda C.
The goal of this project is to provide a readable implementation in pure Cuda as a playground to test modifications and optimizations to flashattention.
Although the goal is NOT to provide the fastest implementation (for this look at the CUTLASS implementation by Tri Dao), we want to it be at least as fast as naive Attention without FlashAttention on large sequence lengths.
