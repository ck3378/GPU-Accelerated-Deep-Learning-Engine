# GPU-Accelerated-Deep-Learning-Engine
Custom CUDA kernels, GPU optimization, and deep learning performance benchmarking

Exploring GPU programming and performance optimization through CUDA C++, Triton, and PyTorch. This project implements fundamental computational operations using custom GPU kernels and compares their performance across different execution strategies on an NVIDIA Tesla T4 GPU.

Tech Stack

Python · CUDA C++ · Triton · PyTorch · NumPy · Google Colab

1. Matrix Multiplication

Implemented and benchmarked Basic CUDA, Shared-Memory Tiling, Register Tiling, and PyTorch GPU implementations.

Benchmark (512 × 512):
<img width="1546" height="776" alt="image" src="https://github.com/user-attachments/assets/392eeeeb-ea16-40ca-bc66-c9b0179fb36d" />
**Key Takeaways**
- Shared-memory tiling reduced execution time by **8.84%** compared to Basic CUDA (2.6901 ms → 2.4524 ms).
- Register tiling achieved a **1.89% improvement** over Basic CUDA but was **7.62% slower** than Shared-Memory Tiling.
- PyTorch GPU achieved **20.20× speedup** over Basic CUDA, reducing execution time from 2.6901 ms to 0.1332 ms.
- PyTorch GPU was approximately **18.41× faster** than the best custom CUDA implementation (Shared-Memory Tiling).
