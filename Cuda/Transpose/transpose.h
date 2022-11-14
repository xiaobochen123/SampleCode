#pragma once

#include "int_fastdivmod.h"
#include "utils.h"

/**
 * @brief Navie Implementation
 */

template <int NUM_AXES, typename T>
__global__ void transpose_kernel_v0(const T* data_in, T* data_out,
                                    const Array<uint32_t, NUM_AXES> perms,
                                    const Array<uint32_t, NUM_AXES> strides_in,
                                    const Array<uint32_t, NUM_AXES> strides_out,
                                    const size_t all_cnt) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < all_cnt) {
        size_t offset_out = tid;
        size_t offset_tmp = offset_out;
        size_t offset_in = 0;
        for (int i = 0; i < NUM_AXES; i++) {
            offset_in += (offset_tmp / strides_out[i]) * strides_in[perms[i]];
            offset_tmp %= strides_out[i];
        }
        data_out[offset_out] = data_in[offset_in];
    }
}

/**
 * @brief Optimize: UNROLL
 */

template <int NUM_AXES, int UNROLL, int BLOCK_SIZE, typename T>
__global__ void transpose_kernel_v1(const T* data_in, T* data_out,
                                    const Array<uint32_t, NUM_AXES> perms,
                                    const Array<uint32_t, NUM_AXES> strides_in,
                                    const Array<uint32_t, NUM_AXES> strides_out,
                                    const size_t all_cnt) {
#pragma unroll
    for (int idx = 0; idx < UNROLL; ++idx) {
        const uint32_t offset_out = blockIdx.x * BLOCK_SIZE + threadIdx.x +
                                    idx * gridDim.x * BLOCK_SIZE;
        if (offset_out >= all_cnt) return;

        uint32_t offset_tmp = offset_out;
        uint32_t offset_in = 0;
#pragma unroll
        for (int i = 0; i < NUM_AXES; ++i) {
            offset_in += (offset_tmp / strides_out[i]) * strides_in[perms[i]];
            offset_tmp %= strides_out[i];
        }
        data_out[offset_out] = data_in[offset_in];
    }
}

/**
 * @brief Optimize : FastIntDivider
 *
 */
template <int NUM_AXES, int UNROLL, int BLOCK_SIZE, typename T>
__global__ void transpose_kernel_v2(
    const T* data_in, T* data_out, const Array<uint32_t, NUM_AXES> perms,
    const Array<uint32_t, NUM_AXES> strides_in,
    Array<FastIntDivider<uint32_t>, NUM_AXES> out_strides,
    const size_t all_cnt) {
#pragma unroll
    for (int idx = 0; idx < UNROLL; ++idx) {
        const uint32_t offset_out = blockIdx.x * BLOCK_SIZE + threadIdx.x +
                                    idx * gridDim.x * BLOCK_SIZE;
        if (offset_out >= all_cnt) return;

        uint32_t offset_tmp = offset_out;
        uint32_t offset_in = 0;
#pragma unroll
        for (int i = 0; i < NUM_AXES; ++i) {
            QuotientMod<uint32_t> d = out_strides[i].divmod(offset_tmp);
            offset_in += d.quotient * strides_in[perms[i]];
            offset_tmp = d.mod;
        }
        data_out[offset_out] = data_in[offset_in];
    }
}

/**
 * @brief Optimize : Reorganize
 *
 */
template <int NUM_AXES, int UNROLL, int BLOCK_SIZE, typename T>
__global__ void transpose_kernel_v3(
    const T* data_in, T* data_out, const Array<uint32_t, NUM_AXES> perm_strides,
    Array<FastIntDivider<uint32_t>, NUM_AXES> out_strides,
    const size_t all_cnt) {
    uint32_t out_offset_reg[UNROLL];
    uint32_t in_offset_reg[UNROLL];
#pragma unroll
    for (int i = 0; i < UNROLL; ++i) {
        out_offset_reg[i] =
            blockIdx.x * BLOCK_SIZE * UNROLL + threadIdx.x + i * BLOCK_SIZE;
        in_offset_reg[i] = 0;
    }

#pragma unroll
    for (int j = 0; j < NUM_AXES; ++j) {
#pragma unroll
        for (int i = 0; i < UNROLL; ++i) {
            QuotientMod<uint32_t> d = out_strides[j].divmod(out_offset_reg[i]);
            in_offset_reg[i] += d.quotient * perm_strides[j];
            out_offset_reg[i] = d.mod;
        }
    }

    T ld_reg[UNROLL];
    uint32_t offset = blockIdx.x * BLOCK_SIZE * UNROLL + threadIdx.x;
    if (offset + BLOCK_SIZE * UNROLL <= all_cnt) {
#pragma unroll
        for (int i = 0; i < UNROLL; ++i) {
            ld_reg[i] = data_in[in_offset_reg[i]];
        }
#pragma unroll
        for (int i = 0; i < UNROLL; ++i) {
            data_out[offset + i * BLOCK_SIZE] = ld_reg[i];
        }
    } else {
#pragma unroll
        for (int i = 0; i < UNROLL; ++i) {
            if (offset + i * BLOCK_SIZE < all_cnt) {
                ld_reg[i] = data_in[in_offset_reg[i]];
            }
        }
#pragma unroll
        for (int i = 0; i < UNROLL; ++i) {
            if (offset + i * BLOCK_SIZE < all_cnt) {
                data_out[offset + i * BLOCK_SIZE] = ld_reg[i];
            }
        }
    }
}

std::vector<uint32_t> TransposeDim(std::vector<uint32_t> dims_in,
                                   std::vector<uint32_t> perms) {
    std::vector<uint32_t> dims_out(perms.size());
    for (int i = 0; i < perms.size(); ++i) {
        dims_out[i] = dims_in[perms[i]];
    }
    return dims_out;
}

std::vector<uint32_t> GetStrides(std::vector<uint32_t> dims) {
    std::vector<uint32_t> strides(dims.size(), 1);
    for (int i = dims.size() - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * dims[i + 1];
    }
    return strides;
}

template <typename T, int TheoryBW>
void ComputeTheory(size_t cnt, float time, std::string tag) {
    float mem = cnt * sizeof(T) * 2;
    float t1 = mem / 1024 / 1024 / 1024;
    float t2 = t1 / (time / 1000);
    printf("[%s] %f ms, %f G, %f %%\n", tag.c_str(), time, t2,
           t2 / (float)TheoryBW * 100.0f);
}

template <typename T, int NUM_AXES>
void TestTranspose(std::vector<T>& data_in, std::vector<T>& data_out_ref,
                   std::vector<uint32_t> dims_in, std::vector<uint32_t> perms) {
    std::vector<uint32_t> dims_out = TransposeDim(dims_in, perms);
    std::vector<T> data_out(AccMul(dims_out), T(0));
    auto strides_in = GetStrides(dims_in);
    auto strides_out = GetStrides(dims_out);

    T* data_in_dev = nullptr;
    T* data_out_dev = nullptr;
    CheckCudaStatus(cudaMalloc(reinterpret_cast<void**>(&data_in_dev),
                               AccMul(dims_in) * sizeof(T)));
    CheckCudaStatus(cudaMalloc(reinterpret_cast<void**>(&data_out_dev),
                               AccMul(dims_out) * sizeof(T)));
    CheckCudaStatus(cudaMemcpy(data_in_dev, data_in.data(),
                               data_in.size() * sizeof(T),
                               cudaMemcpyHostToDevice));

    cudaStream_t stream = nullptr;
    CheckCudaStatus(cudaStreamCreate(&stream));
    size_t all_cnt = AccMul(dims_in);

    // Test Kernel-0
    auto kernel_0 = [&]() {
        const int BLOCK_SIZE = 128;
        const int grid = (all_cnt + BLOCK_SIZE - 1) / BLOCK_SIZE;
        Array<uint32_t, NUM_AXES> perms_arr;
        Array<uint32_t, NUM_AXES> strides_in_arr;
        Array<uint32_t, NUM_AXES> strides_out_arr;
        for (int i = 0; i < NUM_AXES; i++) {
            perms_arr[i] = perms[i];
            strides_in_arr[i] = strides_in[i];
            strides_out_arr[i] = strides_out[i];
        }
        transpose_kernel_v0<NUM_AXES, T><<<grid, BLOCK_SIZE, 0, stream>>>(
            data_in_dev, data_out_dev, perms_arr, strides_in_arr,
            strides_out_arr, all_cnt);
    };

    // Test Kernel-1
    auto kernel_1 = [&]() {
        const int UNROLL = 8 / sizeof(T);
        const int BLOCK_SIZE = 128;
        const int grid =
            (all_cnt + BLOCK_SIZE * UNROLL - 1) / (BLOCK_SIZE * UNROLL);
        Array<uint32_t, NUM_AXES> perms_arr;
        Array<uint32_t, NUM_AXES> strides_in_arr;
        Array<uint32_t, NUM_AXES> strides_out_arr;
        for (int i = 0; i < NUM_AXES; i++) {
            perms_arr[i] = perms[i];
            strides_in_arr[i] = strides_in[i];
            strides_out_arr[i] = strides_out[i];
        }
        transpose_kernel_v1<NUM_AXES, UNROLL, BLOCK_SIZE, T>
            <<<grid, BLOCK_SIZE, 0, stream>>>(data_in_dev, data_out_dev,
                                              perms_arr, strides_in_arr,
                                              strides_out_arr, all_cnt);
    };

    // Test Kernel-2
    auto kernel_2 = [&]() {
        const int UNROLL = 8 / sizeof(T);
        const int BLOCK_SIZE = 128;
        const int grid =
            (all_cnt + BLOCK_SIZE * UNROLL - 1) / (BLOCK_SIZE * UNROLL);
        Array<uint32_t, NUM_AXES> perms_arr;
        Array<uint32_t, NUM_AXES> strides_in_arr;
        Array<FastIntDivider<uint32_t>, NUM_AXES> out_strides_fast;
        for (int i = 0; i < NUM_AXES; ++i) {
            perms_arr[i] = perms[i];
            strides_in_arr[i] = strides_in[i];
            out_strides_fast[i] = FastIntDivider<uint32_t>(strides_out[i]);
        }
        transpose_kernel_v2<NUM_AXES, UNROLL, BLOCK_SIZE, T>
            <<<grid, BLOCK_SIZE, 0, stream>>>(data_in_dev, data_out_dev,
                                              perms_arr, strides_in_arr,
                                              out_strides_fast, all_cnt);
    };

    // Test Kernel-3
    auto kernel_3 = [&]() {
        const int UNROLL = 8 / sizeof(T);
        const int BLOCK_SIZE = 128;
        const int grid =
            (all_cnt + BLOCK_SIZE * UNROLL - 1) / (BLOCK_SIZE * UNROLL);
        Array<uint32_t, NUM_AXES> perm_strides;
        Array<FastIntDivider<uint32_t>, NUM_AXES> out_strides_fast;
        for (int i = 0; i < NUM_AXES; ++i) {
            out_strides_fast[i] = FastIntDivider<uint32_t>(strides_out[i]);
            perm_strides[i] = strides_in[perms[i]];
        }
        transpose_kernel_v3<NUM_AXES, UNROLL, BLOCK_SIZE, T>
            <<<grid, BLOCK_SIZE, 0, stream>>>(data_in_dev, data_out_dev,
                                              perm_strides, out_strides_fast,
                                              all_cnt);
    };

    // RTX2070 Theoretical bandwidth is 448.0 GB/s
    const int TheoryBW = 448;

    // 0
    float k0_cost_time = ProfileKernel(kernel_0);
    cudaStreamSynchronize(stream);
    ComputeTheory<T, TheoryBW>(all_cnt, k0_cost_time, "Kernel-0");
    CheckCudaStatus(cudaMemcpy(data_out.data(), data_out_dev,
                               data_out.size() * sizeof(T),
                               cudaMemcpyDeviceToHost));
    CheckVec<T>(data_out_ref, data_out);

    // 1
    float k1_cost_time = ProfileKernel(kernel_1);
    cudaStreamSynchronize(stream);
    ComputeTheory<T, TheoryBW>(all_cnt, k1_cost_time, "Kernel-1");
    CheckCudaStatus(cudaMemcpy(data_out.data(), data_out_dev,
                               data_out.size() * sizeof(T),
                               cudaMemcpyDeviceToHost));
    CheckVec<T>(data_out_ref, data_out);

    // 2
    float k2_cost_time = ProfileKernel(kernel_2);
    cudaStreamSynchronize(stream);
    ComputeTheory<T, TheoryBW>(all_cnt, k2_cost_time, "Kernel-2");
    CheckCudaStatus(cudaMemcpy(data_out.data(), data_out_dev,
                               data_out.size() * sizeof(T),
                               cudaMemcpyDeviceToHost));
    CheckVec<T>(data_out_ref, data_out);

    // 3
    float k3_cost_time = ProfileKernel(kernel_3);
    cudaStreamSynchronize(stream);
    ComputeTheory<T, TheoryBW>(all_cnt, k3_cost_time, "Kernel-3");
    CheckCudaStatus(cudaMemcpy(data_out.data(), data_out_dev,
                               data_out.size() * sizeof(T),
                               cudaMemcpyDeviceToHost));
    CheckVec<T>(data_out_ref, data_out);

    if (data_in_dev != nullptr) {
        CheckCudaStatus(cudaFree(data_in_dev));
    }
    if (data_out_dev != nullptr) {
        CheckCudaStatus(cudaFree(data_out_dev));
    }
    if (stream) {
        CheckCudaStatus(cudaStreamDestroy(stream));
    }
}