//=======================================================================
// Copyright (c) 2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "egblas/scalar_set.hpp"

template <typename T>
__global__ void scalar_set_kernel(T* x, size_t n, size_t s, const T beta) {
    auto index  = 1 * (threadIdx.x + blockIdx.x * blockDim.x);
    auto stride = 1 * (blockDim.x * gridDim.x);

    for (; index < n; index += stride) {
        x[s * index] = beta;
    }
}

template <typename T>
void scalar_set_kernel_run(T* x, size_t n, size_t s, T beta) {
    int blockSize;
    int minGridSize;

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, scalar_set_kernel<T>, 0, 0);

    int gridSize = ((n / s) + blockSize - 1) / blockSize;

    scalar_set_kernel<T><<<gridSize, blockSize>>>(x, n, s, beta);

    cudaDeviceSynchronize();
}

void egblas_scalar_sset(float* x, size_t n, size_t s, float beta) {
    scalar_set_kernel_run(x, n, s, beta);
}

void egblas_scalar_dset(double* x, size_t n, size_t s, double beta) {
    scalar_set_kernel_run(x, n, s, beta);
}
