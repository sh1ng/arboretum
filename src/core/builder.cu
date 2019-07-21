#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/system/cuda/experimental/pinned_allocator.h>
#include "builder.h"
#include "cub/cub.cuh"
#include "cub/iterator/discard_output_iterator.cuh"
#include "cuda_helpers.h"
#include "cuda_runtime.h"

namespace arboretum {
namespace core {

template <typename NODE_T, typename NODE_VALUE_T>
__global__ void assign_kernel(const unsigned int *const __restrict__ fvalue,
                              const NODE_T *const __restrict__ segments,
                              const unsigned char fvalue_size,
                              NODE_VALUE_T *out, const size_t n) {
  for (unsigned int i = blockDim.x * blockIdx.x + threadIdx.x; i < n;
       i += gridDim.x * blockDim.x) {
    const NODE_VALUE_T node = segments[i];
    out[i] = (node << fvalue_size) | (NODE_VALUE_T)fvalue[i];
  }
}

template __global__ void assign_kernel<unsigned char, unsigned char>(
  const unsigned int *const __restrict__ fvalue,
  const unsigned char *const __restrict__ segments,
  const unsigned char fvalue_size, unsigned char *out, const size_t n);

template __global__ void assign_kernel<unsigned short, unsigned char>(
  const unsigned int *const __restrict__ fvalue,
  const unsigned short *const __restrict__ segments,
  const unsigned char fvalue_size, unsigned char *out, const size_t n);

template __global__ void assign_kernel<unsigned int, unsigned char>(
  const unsigned int *const __restrict__ fvalue,
  const unsigned int *const __restrict__ segments,
  const unsigned char fvalue_size, unsigned char *out, const size_t n);

template __global__ void assign_kernel<unsigned long, unsigned char>(
  const unsigned int *const __restrict__ fvalue,
  const unsigned long *const __restrict__ segments,
  const unsigned char fvalue_size, unsigned char *out, const size_t n);

template __global__ void assign_kernel<unsigned long long, unsigned char>(
  const unsigned int *const __restrict__ fvalue,
  const unsigned long long *const __restrict__ segments,
  const unsigned char fvalue_size, unsigned char *out, const size_t n);

template __global__ void assign_kernel<unsigned short, unsigned short>(
  const unsigned int *const __restrict__ fvalue,
  const unsigned short *const __restrict__ segments,
  const unsigned char fvalue_size, unsigned short *out, const size_t n);

template __global__ void assign_kernel<unsigned int, unsigned short>(
  const unsigned int *const __restrict__ fvalue,
  const unsigned int *const __restrict__ segments,
  const unsigned char fvalue_size, unsigned short *out, const size_t n);

template __global__ void assign_kernel<unsigned long, unsigned short>(
  const unsigned int *const __restrict__ fvalue,
  const unsigned long *const __restrict__ segments,
  const unsigned char fvalue_size, unsigned short *out, const size_t n);

template __global__ void assign_kernel<unsigned long long, unsigned short>(
  const unsigned int *const __restrict__ fvalue,
  const unsigned long long *const __restrict__ segments,
  const unsigned char fvalue_size, unsigned short *out, const size_t n);

template __global__ void assign_kernel<unsigned int, unsigned int>(
  const unsigned int *const __restrict__ fvalue,
  const unsigned int *const __restrict__ segments,
  const unsigned char fvalue_size, unsigned int *out, const size_t n);

template __global__ void assign_kernel<unsigned long, unsigned int>(
  const unsigned int *const __restrict__ fvalue,
  const unsigned long *const __restrict__ segments,
  const unsigned char fvalue_size, unsigned int *out, const size_t n);

template __global__ void assign_kernel<unsigned long long, unsigned int>(
  const unsigned int *const __restrict__ fvalue,
  const unsigned long long *const __restrict__ segments,
  const unsigned char fvalue_size, unsigned int *out, const size_t n);

template __global__ void assign_kernel<unsigned long, unsigned long>(
  const unsigned int *const __restrict__ fvalue,
  const unsigned long *const __restrict__ segments,
  const unsigned char fvalue_size, unsigned long *out, const size_t n);

template __global__ void assign_kernel<unsigned long long, unsigned long long>(
  const unsigned int *const __restrict__ fvalue,
  const unsigned long long *const __restrict__ segments,
  const unsigned char fvalue_size, unsigned long long *out, const size_t n);

}  // namespace core
}  // namespace arboretum