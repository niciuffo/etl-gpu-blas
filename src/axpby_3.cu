//=======================================================================
// Copyright (c) 2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "egblas/axpby_3.hpp"

#include "complex.hpp"

template <typename T>
__global__ void axpby_3_kernel(size_t n, T alpha, const T* x, size_t incx, T beta, T* y, size_t incy, T* yy, size_t incyy) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;
    auto stride = blockDim.x * gridDim.x;

    for (; index < n; index += stride) {
        yy[incyy * index] = alpha * x[incx * index] + beta * y[incy * index];
    }
}

template <typename T>
__global__ void axpby_3_kernel1(size_t n, const T* x, size_t incx, T* y, size_t incy, T* yy, size_t incyy) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;
    auto stride = blockDim.x * gridDim.x;

    for (; index < n; index += stride) {
        yy[incyy * index] = x[incx * index] + y[incy * index];
    }
}

template <typename T>
__global__ void axpby_3_kernel0(size_t n, T* yy, size_t incyy) {
    auto index  = threadIdx.x + blockIdx.x * blockDim.x;
    auto stride = blockDim.x * gridDim.x;

    for (; index < n; index += stride) {
        yy[incyy * index] = zero<T>();
    }
}

template <typename T>
void axpby_3_kernel_run(size_t n, T alpha, const T* x, size_t incx, T beta, T* y, size_t incy, T* yy, size_t incyy) {
    int blockSize;
    int minGridSize;

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, axpby_3_kernel<T>, 0, 0);

    int gridSize = ((n / incyy) + blockSize - 1) / blockSize;

    axpby_3_kernel<T><<<gridSize, blockSize>>>(n, alpha, x, incx, beta, y, incy, yy, incyy);

#ifdef EGBLAS_SYNCHRONIZE
    cudaDeviceSynchronize();
#endif
}

template <typename T>
void axpby_3_kernel1_run(size_t n, const T* x, size_t incx, T* y, size_t incy, T* yy, size_t incyy) {
    int blockSize;
    int minGridSize;

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, axpby_3_kernel1<T>, 0, 0);

    int gridSize = ((n / incyy) + blockSize - 1) / blockSize;

    axpby_3_kernel1<T><<<gridSize, blockSize>>>(n, x, incx, y, incy, yy, incyy);

#ifdef EGBLAS_SYNCHRONIZE
    cudaDeviceSynchronize();
#endif
}

template <typename T>
void axpby_3_kernel0_run(size_t n, T* yy, size_t incyy) {
    int blockSize;
    int minGridSize;

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, axpby_3_kernel0<T>, 0, 0);

    int gridSize = ((n / incyy) + blockSize - 1) / blockSize;

    axpby_3_kernel0<T><<<gridSize, blockSize>>>(n, yy, incyy);

#ifdef EGBLAS_SYNCHRONIZE
    cudaDeviceSynchronize();
#endif
}

void egblas_saxpby_3(size_t n, float alpha, const float* x, size_t incx, float beta, float* y, size_t incy, float* yy, size_t incyy) {
    if (alpha == 1.0f && beta == 1.0f) {
        axpby_3_kernel1_run(n, x, incx, y, incy, yy, incyy);
    } else if (alpha == 0.0f && beta == 0.0f) {
        axpby_3_kernel0_run(n, yy, incyy);
    } else {
        axpby_3_kernel_run(n, alpha, x, incx, beta, y, incy, yy, incyy);
    }
}

void egblas_daxpby_3(size_t n, double alpha, const double* x, size_t incx, double beta, double* y, size_t incy, double* yy, size_t incyy) {
    if (alpha == 1.0 && beta == 1.0) {
        axpby_3_kernel1_run(n, x, incx, y, incy, yy, incyy);
    } else if (alpha == 0.0 && beta == 0.0) {
        axpby_3_kernel0_run(n, yy, incyy);
    } else {
        axpby_3_kernel_run(n, alpha, x, incx, beta, y, incy, yy, incyy);
    }
}

void egblas_caxpby_3(size_t n, cuComplex alpha, const cuComplex* x, size_t incx, cuComplex beta, cuComplex* y, size_t incy, cuComplex* yy, size_t incyy) {
    if (alpha.x == 1.0f && alpha.y == 0.0f && beta.x == 1.0f && beta.y == 0.0f) {
        axpby_3_kernel1_run(n, x, incx, y, incy, yy, incyy);
    } else if (alpha.x == 0.0f && alpha.y == 0.0f && beta.x == 1.0f && beta.y == 0.0f) {
        axpby_3_kernel0_run(n, yy, incyy);
    } else {
        axpby_3_kernel_run(n, alpha, x, incx, beta, y, incy, yy, incyy);
    }
}

void egblas_zaxpby_3(size_t n, cuDoubleComplex alpha, const cuDoubleComplex* x, size_t incx, cuDoubleComplex beta, cuDoubleComplex* y, size_t incy, cuDoubleComplex* yy, size_t incyy) {
    if (alpha.x == 1.0 && alpha.y == 0.0 && beta.x == 1.0 && beta.y == 0.0) {
        axpby_3_kernel1_run(n, x, incx, y, incy, yy, incyy);
    } else if (alpha.x == 0.0 && alpha.y == 0.0 && beta.x == 1.0 && beta.y == 0.0) {
        axpby_3_kernel0_run(n, yy, incyy);
    } else {
        axpby_3_kernel_run(n, alpha, x, incx, beta, y, incy, yy, incyy);
    }
}

void egblas_iaxpby_3(size_t n, int32_t alpha, const int32_t* x, size_t incx, int32_t beta, int32_t* y, size_t incy, int32_t* yy, size_t incyy) {
    if (alpha == 1 && beta == 1) {
        axpby_3_kernel1_run(n, x, incx, y, incy, yy, incyy);
    } else if (alpha == 0 && beta == 0) {
        axpby_3_kernel0_run(n, yy, incyy);
    } else {
        axpby_3_kernel_run(n, alpha, x, incx, beta, y, incy, yy, incyy);
    }
}

void egblas_laxpby_3(size_t n, int64_t alpha, const int64_t* x, size_t incx, int64_t beta, int64_t* y, size_t incy, int64_t* yy, size_t incyy) {
    if (alpha == 1 && beta == 1) {
        axpby_3_kernel1_run(n, x, incx, y, incy, yy, incyy);
    } else if (alpha == 0 && beta == 0) {
        axpby_3_kernel0_run(n, yy, incyy);
    } else {
        axpby_3_kernel_run(n, alpha, x, incx, beta, y, incy, yy, incyy);
    }
}
