// CASPAR - Copyright 2024, Emil Martens, SFI Autoship, NTNU
// This source code is under the Apache 2.0 license found in the LICENSE file.
#pragma once
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;

template <typename T, int elem, bool row_vector>
__forceinline__ __device__ void write_together(
    const cg::coalesced_group subgroup,
    const uint *const vec_idxs_shared,
    const uint ord,
    const T value,
    cuda::atomic<float, cuda::thread_scope_block> *out_shared,
    T *const out,
    const uint stride)
{
    cg::reduce_update_async(
        subgroup,
        out_shared[ord],
        (float)value,
        cg::plus<float>());

    __syncthreads();
    if (vec_idxs_shared[threadIdx.x] != 0xffffffff)
    {
        if constexpr (row_vector)
            atomicAdd_system(&out[elem + stride * vec_idxs_shared[threadIdx.x]], out_shared[threadIdx.x].load());
        else
            atomicAdd_system(&out[elem * stride + vec_idxs_shared[threadIdx.x]], out_shared[threadIdx.x].load());
        out_shared[threadIdx.x].store(0.0);
    }
    __syncthreads();
}

template <typename T, int elem>
__forceinline__ __device__ void write_unique(
    const cg::coalesced_group warp,
    const T value,
    cuda::atomic<float, cuda::thread_scope_block> *out_shared,
    T *const out)
{
    cg::reduce_update_async(
        warp,
        out_shared[0],
        (float)value,
        cg::plus<float>());

    __syncthreads();
    if (threadIdx.x == 0)
    {
        atomicAdd_system(&out[elem], out_shared[0].load());
        out_shared[0].store(0.0);
    }
    __syncthreads();
}

template <typename T, int elem, bool row_vector>
__forceinline__ __device__ T read_together(
    const uint *const vec_idxs_shared,
    const uint ord,
    T *in_shared,
    const T *const in,
    const uint stride)
{
    __syncthreads();
    if (vec_idxs_shared[threadIdx.x] != 0xffffffff)
    {
        if constexpr (row_vector)
            in_shared[threadIdx.x] = in[elem + stride * vec_idxs_shared[threadIdx.x]];
        else
            in_shared[threadIdx.x] = in[elem * stride + vec_idxs_shared[threadIdx.x]];
    }
    __syncthreads();
    return in_shared[ord];
}

template <typename T, int elem>
__forceinline__ __device__ T read_unique(
    T *in_shared,
    const T *const in)
{
    __syncthreads();
    if (threadIdx.x == 0)
        in_shared[0] = in[elem];
    __syncthreads();
    return in_shared[0];
}