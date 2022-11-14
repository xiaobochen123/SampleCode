#pragma once

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

typedef __half float16;

#define WARP_SIZE 32;

#define CUDA_HOST_DEVICE __forceinline__ __device__ __host__
#define CUDA_DEVICE __forceinline__ __device__

//----------------------------------------------------------------
inline void CheckCudaStatus(cudaError_t status) {
    if (status != cudaSuccess) {
        printf("Cuda API failed with status %d: %s\n", status,
               cudaGetErrorString(status));
        throw std::logic_error("Cuda API failed");
    }
}

inline void CheckCublasStatus(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("CuBLAS API failed with status %d, %d, %s\n", status, __LINE__,
               __FILE__);
        throw std::logic_error("Cuda API failed");
    }
}

//----------------------------------------------------------------
template <typename T>
void GenerateRandomData(std::vector<T>& data, size_t size, float low = -1.0f,
                        float high = 1.0f) {
    data.clear();
    data.resize(size);
    std::uniform_real_distribution<T> distribution(low, high);
    std::default_random_engine generator;
    generator.seed(1);
    std::generate(data.begin(), data.end(),
                  [&]() { return T(distribution(generator)); });
}

template <>
void GenerateRandomData<float16>(std::vector<float16>& data, size_t size,
                                 float low, float high) {
    data.clear();
    data.resize(size);
    std::uniform_real_distribution<float> distribution(low, high);
    std::default_random_engine generator;
    generator.seed(1);
    std::generate(data.begin(), data.end(),
                  [&]() { return float16(distribution(generator)); });
}

template <>
void GenerateRandomData<int8_t>(std::vector<int8_t>& data, size_t size,
                                float low, float high) {
    data.clear();
    data.resize(size);
    std::uniform_real_distribution<float> distribution(-128, 127);
    std::default_random_engine generator;
    generator.seed(1);
    std::generate(data.begin(), data.end(),
                  [&]() { return int8_t(distribution(generator)); });
}

CUDA_HOST_DEVICE int CeilDiv(const int a, const int b) {
    return (a + b - 1) / b;
}

//----------------------------------------------------------------
template <typename T>
void PrintMatmul(std::vector<T> data, const int rows, const int cols,
                 const std::string tag = "") {
    std::cout << tag << std::endl;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << data[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
}

template <>
void PrintMatmul<float>(std::vector<float> data, const int rows, const int cols,
                        const std::string tag) {
    std::cout << tag << std::endl;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << std::setw(5) << std::right << std::fixed
                      << std::setprecision(2) << data[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
}

//----------------------------------------------------------------

template <typename T>
T AccMul(std::vector<T> vec) {
    return std::accumulate(vec.begin(), vec.end(), T(1), std::multiplies<T>());
}

//----------------------------------------------------------------
//----------------------------------------------------------------

template <typename T, int NUM>
struct Array {
    CUDA_HOST_DEVICE T& operator[](unsigned int index) { return data[index]; }
    CUDA_HOST_DEVICE const T& operator[](unsigned int index) const {
        return data[index];
    }
    CUDA_HOST_DEVICE constexpr int size() const { return NUM; }

    CUDA_HOST_DEVICE Array() {
#ifndef __CUDA_ARCH__
        for (int i = 0; i < NUM; i++) {
            data[i] = T();
        }
#endif
    }

    T data[NUM];
};

// profile function
float ProfileKernel(std::function<void()> kernel, int test_epoch = 10) {
    // warn up
    for (int i = 0; i < 5; ++i) {
        kernel();
    }
    CheckCudaStatus(cudaGetLastError());
    cudaEvent_t beg, end;
    cudaEventCreate(&beg);
    cudaEventCreate(&end);
    cudaProfilerStart();
    cudaEventRecord(beg);
    for (int i = 0; i < test_epoch; ++i) {
        kernel();
    }
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaProfilerStop();
    float cost_time;
    cudaEventElapsedTime(&cost_time, beg, end);
    CheckCudaStatus(cudaGetLastError());
    return cost_time / (float)test_epoch;
}

//------------------------------
// Check data

template <typename T>
void CheckVec(const std::vector<T> x, const std::vector<T> y,
             const float eps = 1e-4) {
    std::cout << "Check data\n";
    int error_cnt = 0;
    for (int i = 0; i < x.size(); ++i) {
        if (error_cnt == 10) return;
        if (abs(x[i] - y[i]) > eps) {
            error_cnt++;
            std::cout << "Idx " << i << ", Diff=" << abs(x[i] - y[i])
                      << ", x=" << x[i] << ", y=" << y[i] << std::endl;
        }
    }
}

template <>
void CheckVec<float16>(const std::vector<float16> x,
                      const std::vector<float16> y, const float eps) {
    std::cout << "Check data\n";
    int error_cnt = 0;
    for (int i = 0; i < x.size(); ++i) {
        if (error_cnt == 10) return;
        if (abs(float(x[i]) - float(y[i])) > eps) {
            error_cnt++;
            std::cout << "Idx " << i
                      << ", Diff=" << abs(float(x[i]) - float(y[i]))
                      << ", x=" << float(x[i]) << ", y=" << float(y[i])
                      << std::endl;
        }
    }
}