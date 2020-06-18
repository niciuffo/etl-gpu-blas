//=======================================================================
// Copyright (c) 2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <complex>
#include <iostream>

#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"

#include "egblas.hpp"
#include "test.hpp"

#include "catch.hpp"

TEST_CASE("bias_batch_sum_4d/s/0", "[float][bias_batch_sum]") {
    const size_t D0 = 2;
    const size_t D1 = 3;
    const size_t D2 = 4;
    const size_t D3 = 5;
    const size_t DIM = D0 * D1 * D2 * D3;

    const size_t SIZE = D0 * D1 * D2 * D3;
    const size_t S0   = SIZE / D0;
    const size_t S1   = S0   / D1;


    float* x_cpu = new float[SIZE];
    float* y_cpu = new float[D1];

    for (size_t i = 0; i < SIZE; ++i) {
        x_cpu[i] = i;
    }

    float* x_gpu;
    float* y_gpu;

    cuda_check(cudaMalloc((void**)&x_gpu, SIZE * sizeof(float)));
    cuda_check(cudaMalloc((void**)&y_gpu, D1 * sizeof(float)));

    cuda_check(cudaMemcpy(x_gpu, x_cpu, SIZE * sizeof(float), cudaMemcpyHostToDevice));

    egblas_sbias_batch_sum_4d(D0, D1, S0, S1, x_gpu, y_gpu, 1);

    cuda_check(cudaMemcpy(y_cpu, y_gpu, D1 * sizeof(float), cudaMemcpyDeviceToHost));

    REQUIRE(y_cpu[0] == 1580.0f);
    REQUIRE(y_cpu[1] == 2380.0f);
    REQUIRE(y_cpu[2] == 3180.0f);

    cuda_check(cudaFree(x_gpu));

    delete[] x_cpu;
}