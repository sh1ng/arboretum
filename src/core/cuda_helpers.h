#ifndef CUDA_HELPERS_H
#define CUDA_HELPERS_H

#include <stdio.h>
#include <cmath>
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

#endif  // CUDA_HELPERS_H
