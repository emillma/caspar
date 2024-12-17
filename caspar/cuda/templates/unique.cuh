// CASPAR - Copyright 2024, Emil Martens, SFI Autoship, NTNU
// This source code is under the Apache 2.0 license found in the LICENSE file.
#pragma once
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

// Batcher odd–even mergesort, sorts at most 1024 elements
template <typename T, uint arrayLength>
__forceinline__ __device__ void sort(T *const values, uint &shared_int)
{

    if (threadIdx.x == 0)
        shared_int = 1;

    __syncthreads();
    if (threadIdx.x + 1 < arrayLength)
        if (values[threadIdx.x] > values[threadIdx.x + 1])
            shared_int = 0;

    __syncthreads();
    if (shared_int == 1)
        return;

#pragma unroll
    for (uint size = 2; size <= arrayLength; size <<= 1)
    {
        uint stride = size / 2;
        uint offset = threadIdx.x & (stride - 1);
        {
            uint pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
            if (pos + stride < arrayLength)
                if (values[pos] > values[pos + stride])
                {
                    T temp = values[pos];
                    values[pos] = values[pos + stride];
                    values[pos + stride] = temp;
                }
            stride >>= 1;
        }
#pragma unroll
        for (; stride > 0; stride >>= 1)
        {
            if (size <= 32)
                __syncwarp();
            else
                __syncthreads();
            uint pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
            if (offset >= stride && pos < arrayLength)
                if (values[pos - stride] > values[pos])
                {
                    T temp = values[pos - stride];
                    values[pos - stride] = values[pos];
                    values[pos] = temp;
                }
        }
        if (size <= 16)
            __syncwarp();
        else
            __syncthreads();
    }
}

__forceinline__ __device__ uint get_ord(const uint *values,
                                        const uint length,
                                        const uint val)
{
    int lo = 0;
    int hi = length;
    while (lo < hi)
    {
        int mid = (lo + hi) / 2;
        if (values[mid] < val)
            lo = mid + 1;
        else
            hi = mid;
    }
    return lo;
}

template <int arrayLength>
__forceinline__ __device__ uint unique(
    uint *values, const uint value, uint &shared_int)
{
    const auto warp = cg::tiled_partition<32>(cg::this_thread_block());
    values[threadIdx.x] = value;
    __syncthreads();
    sort<uint, arrayLength>(values, shared_int);
    bool pred = false;
    if (threadIdx.x < arrayLength - 1)
        pred = values[threadIdx.x] == values[threadIdx.x + 1];
    __syncthreads();
    if (pred)
        values[threadIdx.x + 1] = 0xFFFFFFFF;

    sort<uint, arrayLength>(values, shared_int);

    if (threadIdx.x == 0)
        shared_int = arrayLength;
    __syncthreads();
    if (threadIdx.x < arrayLength - 1)
        if (values[threadIdx.x + 1] == 0xFFFFFFFF && values[threadIdx.x] != 0xFFFFFFFF)
            shared_int = threadIdx.x + 1;
    __syncthreads();
    const int ord = get_ord(values, shared_int, value);
    __syncthreads();
    return ord;
}

__forceinline__ __device__ uint load_if_valid(const uint *const values,
                                              const uint idx,
                                              const uint length,
                                              const uint default_value)
{
    if (idx < length)
        return values[idx];
    return default_value;
}