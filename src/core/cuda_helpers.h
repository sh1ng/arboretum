#ifndef SRC_CORE_CUDA_HELPERS_H
#define SRC_CORE_CUDA_HELPERS_H

#include <stdio.h>
#include <cassert>
#include <cmath>
#include "cub/cub.cuh"
#include "cuda_runtime.h"

#define MAX_THREADS 1024

#define OK(cmd)                                               \
  do {                                                        \
    cudaError_t e = cmd;                                      \
    if (e != cudaSuccess) {                                   \
      printf("Cuda failure %s:%d '%s'\n", __FILE__, __LINE__, \
             cudaGetErrorString(e));                          \
      exit(EXIT_FAILURE);                                     \
    }                                                         \
  } while (0)

#define DEVICE_OK(cmd)                                                  \
  do {                                                                  \
    cudaError_t e = cmd;                                                \
    if (e != cudaSuccess) {                                             \
      printf("Cuda failure in kernel %s:%d '%s'\n", __FILE__, __LINE__, \
             cudaGetErrorString(e));                                    \
      assert(0);                                                        \
    }                                                                   \
  } while (0)

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600
__device__ __forceinline__ double atomicAdd(double *address, double val) {
  unsigned long long int *address_as_ull = (unsigned long long int *)address;
  unsigned long long int old = *address_as_ull, assumed;

  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(val + __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN !=
    // NaN)
  } while (assumed != old);

  return __longlong_as_double(old);
}
#endif

inline __host__ __device__ float to_float(float f) { return f; }

inline __host__ __device__ float to_float(double f) { return float(f); }

inline __host__ __device__ float to_float(float2 f) { return float(f.x); }

inline __host__ __device__ float to_float(double2 f) { return float(f.x); }

inline __host__ __device__ float to_float_y(float f) { return 0; }

inline __host__ __device__ float to_float_y(double f) { return 0; }

inline __host__ __device__ float to_float_y(float2 f) { return float(f.y); }

inline __host__ __device__ float to_float_y(double2 f) { return float(f.y); }

inline bool _isnan(const double2 a) {
  return std::isnan(a.x) || std::isnan(a.y);
}

inline bool _isnan(const float2 a) {
  return std::isnan(a.x) || std::isnan(a.y);
}
inline bool _isnan(const float a) { return std::isnan(a); }

struct mydouble2 : double2 {
  inline __host__ __device__ mydouble2() : double2(make_double2(0.0, 0.0)) {}
  inline __host__ __device__ mydouble2(double2 other) : double2(other) {}
  inline __host__ __device__ mydouble2(float2 other)
      : double2(make_double2(other.x, other.y)) {}
};

inline __host__ __device__ float2 operator/(const float2 a, const int b) {
  return make_float2(a.x / b, a.y / b);
}

inline __host__ __device__ float2 operator*(const float2 a, const float2 b) {
  return make_float2(a.x * b.x, a.y * b.y);
}

inline __host__ __device__ float2 operator-(const float2 a, const float2 b) {
  return make_float2(a.x - b.x, a.y - b.y);
}

inline __host__ __device__ float2 operator+(const float2 a, const float2 b) {
  return make_float2(a.x + b.x, a.y + b.y);
}

inline __host__ __device__ void operator+=(float2 &a, float2 b) {
  a.x += b.x;
  a.y += b.y;
}

inline __host__ __device__ void init(float2 &v) { v = make_float2(0.0, 0.0); }

inline __host__ __device__ void init(float &v) { v = 0.0; }

inline __host__ __device__ double2 operator/(const double2 a, const int b) {
  return make_double2(a.x / b, a.y / b);
}

inline __host__ __device__ double2 operator*(const double2 a, const double2 b) {
  return make_double2(a.x * b.x, a.y * b.y);
}

inline __host__ __device__ double2 operator-(const double2 a, const double2 b) {
  return make_double2(a.x - b.x, a.y - b.y);
}

inline __host__ __device__ double2 operator+(const double2 a, const double2 b) {
  return make_double2(a.x + b.x, a.y + b.y);
}

inline __host__ __device__ uint2 operator+(const uint2 a, const uint2 b) {
  return make_uint2(a.x + b.x, a.y + b.y);
}

// inline __host__ __device__ double2 operator=(float2 b) {
//  return make_double2(b.x, b.y);
//}

inline __host__ __device__ void operator+=(double2 &a, double2 b) {
  a.x += b.x;
  a.y += b.y;
}

inline __host__ __device__ void operator+=(double2 &a, float2 b) {
  a.x += b.x;
  a.y += b.y;
}

inline __host__ __device__ void init(double2 &v) { v = make_double2(0.0, 0.0); }

inline __host__ __device__ void init(double &v) { v = 0.0; }

__forceinline__ __device__ void atomicAdd(float2 *address, const float2 val) {
  atomicAdd(&(address->x), val.x);
  atomicAdd(&(address->y), val.y);
}

__forceinline__ __device__ void atomicAdd(double2 *address, const double2 val) {
  atomicAdd(&(address->x), val.x);
  atomicAdd(&(address->y), val.y);
}

inline __host__ __device__ void print_grad(double &v) {}
inline __host__ __device__ void print_grad(double2 &v) {}
inline __host__ __device__ void print_grad(float &v) {}
inline __host__ __device__ void print_grad(float2 &v) {}

template <class T>
inline __host__ void compute1DInvokeConfig(size_t n, int *minGridSize,
                                           int *blockSize, T func,
                                           size_t dynamicSMemSize = 0,
                                           int blockSizeLimit = 0) {
  OK(cudaOccupancyMaxPotentialBlockSize(minGridSize, blockSize, func,
                                        dynamicSMemSize, blockSizeLimit));
  *minGridSize = int((n + *blockSize - 1) / *blockSize);
}

template <class T>
__forceinline__ __host__ __device__ T linear_search(const T *sorted_segments,
                                                    const T item, const T size,
                                                    const T start = 0) {
  T segment = start;
  while ((item >= sorted_segments[segment + 1])) {
    segment++;
  }
  return segment;
}

template <class T>
__forceinline__ __host__ __device__ T linear_search_2steps(
  const T *sorted_segments, const T item, const T size, const T start = 0) {
  T segment = start;
  if (item >= sorted_segments[segment + 1]) {
    return segment + 1;
  }
  return segment;
}

template <class T>
__forceinline__ __host__ __device__ T binary_search(const T *sorted_segments,
                                                    const T item,
                                                    const T size) {
  T left = 0;
  T right = size - 1;
  while (left < right) {
    T m = (left + right) / 2;
    if (sorted_segments[m] > item)
      right = m;
    else
      left = m + 1;
  }
  return left - 1;
}

template <class T>
__forceinline__ __host__ __device__ unsigned lower_bound(const T *values,
                                                         const T x,
                                                         const int size) {
  unsigned l = 0;
  unsigned h = size;  // Not n - 1
  while (l < h) {
    unsigned mid = (l + h) / 2;
    if (x <= values[mid]) {
      h = mid;
    } else {
      l = mid + 1;
    }
  }
  return l;
}

#endif  // SRC_CORE_CUDA_HELPERS_H
