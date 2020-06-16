//=======================================================================
// Copyright (c) 2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <thrust/execution_policy.h>
#include <thrust/reduce.h>

#include "egblas/assert.hpp"
#include "egblas/cuda_check.hpp"
#include "egblas/sum.hpp"
#include "egblas/utils.hpp"
#include "sum_reduce.hpp"

template <bool Mean, typename T>
__global__ void bias_batch_sum_4d_kernel(size_t B, size_t M, size_t N, size_t S0, size_t S1, const T* x, size_t incx, T* y, size_t incy) {
    auto n = threadIdx.x + blockIdx.x * blockDim.x;

    if (n < N) {
        T sum(0);
        for (int b = 0; b < M; ++b) {
            for (int i = 0; i < S1; i++) {
                sum += x[S0 * b + S1 * (n) + i];
            }
        }

        if (Mean) {
            y[incy * n] = sum / N;
        } else {
            y[incy * n] = sum;
        }
    }
}

void egblas_sbias_batch_sum_4d(size_t b, size_t m, size_t n, size_t s0, size_t s1, float* x, size_t incx, float* y, size_t incy) {
    const int blockSize = 64;
    const int gridSize  = (n + blockSize - 1) / blockSize;

    bias_batch_sum_4d_kernel<false><<<gridSize, blockSize>>>(b, m, n, s0, s1, x, incx, y, incy);
}

void egblas_dbias_batch_sum_4d(size_t b, size_t m, size_t n, size_t s0, size_t s1, double* x, size_t incx, double* y, size_t incy) {
    const int blockSize = 64;
    const int gridSize  = (n + blockSize - 1) / blockSize;

    bias_batch_sum_4d_kernel<false><<<gridSize, blockSize>>>(b, m, n, s0, s1, x, incx, y, incy);
}

void egblas_sbias_batch_mean_4d(size_t b, size_t m, size_t n, size_t s0, size_t s1, float* x, size_t incx, float* y, size_t incy) {
    const int blockSize = 64;
    const int gridSize  = (n + blockSize - 1) / blockSize;

    bias_batch_sum_4d_kernel<true><<<gridSize, blockSize>>>(b, m, n, s0, s1, x, incx, y, incy);
}

void egblas_dbias_batch_mean_4d(size_t b, size_t m, size_t n, size_t s0, size_t s1, double* x, size_t incx, double* y, size_t incy) {
    const int blockSize = 64;
    const int gridSize  = (n + blockSize - 1) / blockSize;

    bias_batch_sum_4d_kernel<true><<<gridSize, blockSize>>>(b, m, n, s0, s1, x, incx, y, incy);
}
