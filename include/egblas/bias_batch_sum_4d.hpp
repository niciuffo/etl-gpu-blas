//=======================================================================
// Copyright (c) 2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include <cuComplex.h>

void egblas_sbias_batch_sum_4d(size_t b, size_t m, size_t n, size_t s0, size_t s1, float* x, size_t incx, float* y, size_t incy);

void egblas_dbias_batch_sum_4d(size_t b, size_t m, size_t n, size_t s0, size_t s1, double* x, size_t incx, double* y, size_t incy);

void egblas_sbias_batch_mean_4d(size_t b, size_t m, size_t n, size_t s0, size_t s1, float* x, size_t incx, float* y, size_t incy);

void egblas_dbias_batch_mean_4d(size_t b, size_t m, size_t n, size_t s0, size_t s1, double* x, size_t incx, double* y, size_t incy);

#define EGBLAS_HAS_SBIAS_BATCH_SUM_4D true
#define EGBLAS_HAS_DBIAS_BATCH_SUM_4D true

#define EGBLAS_HAS_SBIAS_BATCH_MEAN_4D true
#define EGBLAS_HAS_DBIAS_BATCH_MEAN_4D true
