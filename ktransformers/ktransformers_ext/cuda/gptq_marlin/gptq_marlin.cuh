// Adapted from
// https://github.com/vllm-project/vllm/tree/main/csrc/quantization/gptq_marlin
// Copyrigth 2024 The vLLM team.
// Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
#pragma once

#include <torch/all.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>

namespace gptq_marlin {

// 8 warps are a good choice since every SM has 4 schedulers and having more
// than 1 warp per schedule allows some more latency hiding. At the same time,
// we want relatively few warps to have many registers per warp and small tiles.
static constexpr int default_threads = 256;

static constexpr int pipe_stages =
    4;  // 4 pipeline stages fit into shared memory

static constexpr int min_thread_n = 64;
static constexpr int min_thread_k = 64;

static constexpr int tile_size = 16;
static constexpr int max_par = 16;

template <typename T, int n>
struct Vec {
  T elems[n];
  __device__ T& operator[](int i) { return elems[i]; }
};

using I4 = Vec<int, 4>;

constexpr int div_ceil(int a, int b) { return (a + b - 1) / b; }

#ifdef USE_ROCM
// Convert generic pointer to shared memory address for ROCm
template<typename T>
__device__ __forceinline__ uint32_t cvta_to_shared(const T* ptr) {
    // First get the address as a size_t to handle all pointer sizes
    size_t addr = reinterpret_cast<size_t>(ptr);

    // Extract the lower 32 bits which represent the shared memory offset
    // This is safe because shared memory addresses are always within 32-bit range
    return static_cast<uint32_t>(addr & 0xFFFFFFFF);
}
#else
// For CUDA, use the native intrinsic
template<typename T>
__device__ __forceinline__ uint32_t cvta_to_shared(const T* ptr) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}
#endif



#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800)
// No support for async
#else

__device__ inline void cp_async4_pred(void* smem_ptr, const void* glob_ptr,
                                      bool pred = true) {
  const int BYTES = 16;
  uint32_t smem = cvta_to_shared(smem_ptr);
  #ifdef USE_ROCM
  #if __has_builtin(__builtin_amdgcn_global_load_lds)
  __builtin_amdgcn_global_load_lds(static_cast<const uint32_t*>(glob_ptr), &smem, BYTES, 0, 0);
  #else
  // Simple approach using standard C++ operations
  if (pred) {
    // Load from global memory
    uint4 data;
    data = *reinterpret_cast<const uint4 *>(glob_ptr);

    // Store to shared memory
    *reinterpret_cast<uint4 *>(smem_ptr) = data;

    // Ensure visibility
    __threadfence_block();
  }
  #endif
  #else
  asm volatile(
      "{\n"
      "   .reg .pred p;\n"
      "   setp.ne.b32 p, %0, 0;\n"
      "   @p cp.async.cg.shared.global [%1], [%2], %3;\n"
      "}\n" ::"r"((int)pred),
      "r"(smem), "l"(glob_ptr), "n"(BYTES));
  #endif
}

__device__ inline void cp_async4(void* smem_ptr, const void* glob_ptr) {
  const int BYTES = 16;
  uint32_t smem = cvta_to_shared(smem_ptr);
  #ifdef USE_ROCM
  #if __has_builtin(__builtin_amdgcn_global_load_lds)
  __builtin_amdgcn_global_load_lds(static_cast<const uint32_t*>(glob_ptr), &smem, BYTES, 0, 0);
  #else
  // Simple approach using standard C++ operations
  if (true) {
    // Load from global memory
    uint4 data;
    data = *reinterpret_cast<const uint4 *>(glob_ptr);

    // Store to shared memory
    *reinterpret_cast<uint4 *>(smem_ptr) = data;

    // Ensure visibility
    __threadfence_block();
  }
  #endif
  #else
  asm volatile(
      "{\n"
      "   cp.async.cg.shared.global [%0], [%1], %2;\n"
      "}\n" ::"r"(smem),
      "l"(glob_ptr), "n"(BYTES));
  #endif
}

__device__ inline void cp_async_fence() {
#ifdef USE_ROCM
  __builtin_amdgcn_s_waitcnt(0);
#else
  asm volatile("cp.async.commit_group;\n" ::);
#endif
}

template <int n>
__device__ inline void cp_async_wait() {
#ifdef USE_ROCM
  // For AMD GPUs, we use s_waitcnt
  // This waits for all outstanding memory operations to complete
  __builtin_amdgcn_s_waitcnt(0);
#else
  // For NVIDIA GPUs, use the original instruction
  asm volatile("cp.async.wait_group %0;\n" ::"n"(n));
#endif
}

#endif

}  // namespace gptq_marlin
