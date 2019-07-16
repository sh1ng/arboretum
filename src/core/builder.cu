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

template <typename SUM_T>
__global__ void update(SUM_T *sum_dst, unsigned *count_dst,
                       const SUM_T *parent_sum, const unsigned *parent_count,
                       const SUM_T *sum_src, const unsigned *count_src,
                       const unsigned n) {
  for (unsigned int i = blockDim.x * blockIdx.x + threadIdx.x; i < n;
       i += gridDim.x * blockDim.x) {
    sum_dst[i] = parent_sum[i] - sum_src[i];
    count_dst[i] = parent_count[i] - count_src[i];
  }
}

template __global__ void update<float>(float *sum_dst, unsigned *count_dst,
                                       const float *parent_sum,
                                       const unsigned *parent_count,
                                       const float *sum_src,
                                       const unsigned *count_src,
                                       const unsigned n);

template __global__ void update<double>(double *sum_dst, unsigned *count_dst,
                                        const double *parent_sum,
                                        const unsigned *parent_count,
                                        const double *sum_src,
                                        const unsigned *count_src,
                                        const unsigned n);

template __global__ void update<float2>(float2 *sum_dst, unsigned *count_dst,
                                        const float2 *parent_sum,
                                        const unsigned *parent_count,
                                        const float2 *sum_src,
                                        const unsigned *count_src,
                                        const unsigned n);

template __global__ void update<mydouble2>(
  mydouble2 *sum_dst, unsigned *count_dst, const mydouble2 *parent_sum,
  const unsigned *parent_count, const mydouble2 *sum_src,
  const unsigned *count_src, const unsigned n);

template <typename SUM_T, typename GRAD_T, int ITEMS_PER_THREAD>
__global__ void hist_sum_node(SUM_T *dst_sum, unsigned *dst_count,
                              const GRAD_T *__restrict__ values,
                              const unsigned *__restrict__ bin,
                              const unsigned end_bit, const unsigned segment,
                              const size_t n) {
  int warp_id = threadIdx.x / 32;
  int lane = threadIdx.x % 32;
  typedef cub::BlockRadixSort<unsigned short, HIST_SUM_BLOCK_DIM,
                              ITEMS_PER_THREAD, GRAD_T, 5, false,
                              cub ::BLOCK_SCAN_RAKING>
    BlockRadixSort;

  typedef cub::WarpScan<SUM_T> BlockScanSum;
  typedef cub::WarpScan<unsigned short> BlockScanCount;
  typedef cub::WarpScan<cub::KeyValuePair<unsigned short, unsigned short>>
    BlockScanMax;

  struct total {
    SUM_T sum[HIST_SUM_BLOCK_DIM];
    unsigned short count[HIST_SUM_BLOCK_DIM];
  };

  union U {
    U(){};
    typename BlockRadixSort::TempStorage sort;
    typename BlockScanSum::TempStorage scan_sum[HIST_SUM_BLOCK_DIM / 32];
    typename BlockScanCount::TempStorage scan_count[HIST_SUM_BLOCK_DIM / 32];
    typename BlockScanMax::TempStorage scan_max[HIST_SUM_BLOCK_DIM / 32];
    total total;
  };

  __shared__ U temp_storage;

  // Obtain a segment of consecutive items that are blocked across threads
  unsigned short thread_keys[ITEMS_PER_THREAD];
  GRAD_T thread_values[ITEMS_PER_THREAD];

#pragma unroll
  for (unsigned i = 0; i < ITEMS_PER_THREAD; ++i) {
    unsigned idx =
      blockDim.x * blockIdx.x * ITEMS_PER_THREAD + i * blockDim.x + threadIdx.x;
    if (idx < n) {
      thread_keys[i] = bin[idx];
      thread_values[i] = values[idx];
    } else {
      thread_keys[i] = HIST_SUM_NO_DATA;
    }
  }

  // Collectively sort the keys and values among block threads
  BlockRadixSort(temp_storage.sort)
    .Sort(thread_keys, thread_values, 0, end_bit);

  __syncthreads();

  unsigned short key = thread_keys[0];
  SUM_T sum_current = thread_values[0];
  unsigned short count_current = 1;

#pragma unroll
  for (unsigned i = 1; i < ITEMS_PER_THREAD; ++i) {
    if (key == thread_keys[i]) {
      sum_current += thread_values[i];
      count_current++;

    } else {
      atomicAdd(&dst_sum[key], sum_current);
      atomicAdd(&dst_count[key], count_current);
      key = thread_keys[i];
      sum_current = thread_values[i];
      count_current = 1;
    }
  }

  SUM_T zero;
  init(zero);
  SUM_T sum;
  unsigned short count;

  BlockScanSum(temp_storage.scan_sum[warp_id])
    .ExclusiveScan(sum_current, sum, zero, cub::Sum());

  BlockScanCount(temp_storage.scan_count[warp_id])
    .ExclusiveScan(count_current, count, 0, cub::Sum());

  // key - segment idx, value - segment value
  cub::KeyValuePair<unsigned short, unsigned short> segment_start(threadIdx.x,
                                                                  key);

  struct SegmentMaxAndMinIndex {
    __device__ __forceinline__ cub::KeyValuePair<unsigned short, unsigned short>
    operator()(
      const cub::KeyValuePair<unsigned short, unsigned short> &a,
      const cub::KeyValuePair<unsigned short, unsigned short> &b) const {
      if ((b.value > a.value) || ((a.value == b.value) && (b.key < a.key)))
        return b;
      return a;
    }
  };

  cub::KeyValuePair<unsigned short, unsigned short> initial(0, 0);

  BlockScanMax(temp_storage.scan_max[warp_id])
    .ExclusiveScan(segment_start, segment_start, initial,
                   SegmentMaxAndMinIndex());

  temp_storage.total.sum[threadIdx.x] = sum;
  temp_storage.total.count[threadIdx.x] = count;

  __syncthreads();

  // flush previous segment
  if (thread_keys[ITEMS_PER_THREAD - 1] != segment_start.value) {
    atomicAdd(&dst_sum[segment_start.value],
              sum - temp_storage.total.sum[segment_start.key]);
    atomicAdd(&dst_count[segment_start.value],
              count - temp_storage.total.count[segment_start.key]);
  }
  // last thread also need to handle it's own sum
  if (lane == 31 && thread_keys[ITEMS_PER_THREAD - 1] != HIST_SUM_NO_DATA) {
    // flush all collected data
    if (thread_keys[ITEMS_PER_THREAD - 1] == segment_start.value) {
      atomicAdd(&dst_sum[thread_keys[ITEMS_PER_THREAD - 1]],
                sum_current + sum - temp_storage.total.sum[segment_start.key]);
      atomicAdd(
        &dst_count[thread_keys[ITEMS_PER_THREAD - 1]],
        count_current + count - temp_storage.total.count[segment_start.key]);

    } else {  // only thread local sum
      atomicAdd(&dst_sum[thread_keys[ITEMS_PER_THREAD - 1]], sum_current);
      atomicAdd(&dst_count[thread_keys[ITEMS_PER_THREAD - 1]], count_current);
    }
  }
}

template __global__ void hist_sum_node<float, float, HIST_SUM_ITEMS_PER_THREAD>(
  float *dst_sum, unsigned *dst_count, const float *__restrict__ values,
  const unsigned *__restrict__ bin, const unsigned end_bit,
  const unsigned segment, const size_t n);

template __global__ void
hist_sum_node<double, float, HIST_SUM_ITEMS_PER_THREAD>(
  double *dst_sum, unsigned *dst_count, const float *__restrict__ values,
  const unsigned *__restrict__ bin, const unsigned end_bit,
  const unsigned segment, const size_t n);

template __global__ void
hist_sum_node<float2, float2, HIST_SUM_ITEMS_PER_THREAD>(
  float2 *dst_sum, unsigned *dst_count, const float2 *__restrict__ values,
  const unsigned *__restrict__ bin, const unsigned end_bit,
  const unsigned segment, const size_t n);

template __global__ void
hist_sum_node<mydouble2, float2, HIST_SUM_ITEMS_PER_THREAD>(
  mydouble2 *dst_sum, unsigned *dst_count, const float2 *__restrict__ values,
  const unsigned *__restrict__ bin, const unsigned end_bit,
  const unsigned segment, const size_t n);

template <typename SUM_T, typename GRAD_T>
__global__ void hist_sum_dynamic(
  SUM_T *dst_sum, unsigned *dst_count, const SUM_T *hist_sum_parent,
  const unsigned *hist_count_parent, const GRAD_T *__restrict__ values,
  const unsigned *__restrict__ parent_count_iter,
  const unsigned *__restrict__ bin, const unsigned hist_size,
  const unsigned hist_size_bits, const bool use_trick, const size_t n) {
  for (unsigned int i = blockDim.x * blockIdx.x + threadIdx.x; i < n;
       i += gridDim.x * blockDim.x) {
    constexpr unsigned threads = HIST_SUM_BLOCK_DIM;
    constexpr unsigned block_size = threads * HIST_SUM_ITEMS_PER_THREAD;
    const unsigned parent_idx = i / 2;
    const unsigned node_start = parent_count_iter[i];
    const unsigned node_size = parent_count_iter[i + 1] - parent_count_iter[i];
    const unsigned parent_size =
      parent_count_iter[parent_idx * 2 + 2] - parent_count_iter[parent_idx * 2];

    const bool invoke =
      !use_trick || ((node_size * 2 < parent_size) ||
                     (node_size * 2 == parent_size && i % 2 == 0));

    if (invoke && node_size != 0) {
      const unsigned grid_size = (node_size + block_size - 1) / block_size;
      cudaStream_t s;
      DEVICE_OK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));

      hist_sum_node<SUM_T, GRAD_T, HIST_SUM_ITEMS_PER_THREAD>
        <<<grid_size, threads, 0, s>>>(
          dst_sum + i * hist_size, dst_count + i * hist_size,
          values + node_start, bin + node_start, hist_size_bits, i, node_size);
      //   DEVICE_OK(cudaGetLastError());
      DEVICE_OK(cudaDeviceSynchronize());

      DEVICE_OK(cudaStreamDestroy(s));
    }

    if (use_trick && invoke) {
      unsigned other_id = i % 2 == 0 ? i + 1 : i - 1;
      for (unsigned j = 0; j < hist_size; ++j) {
        dst_sum[other_id * hist_size + j] =
          hist_sum_parent[parent_idx * hist_size + j] -
          dst_sum[i * hist_size + j];
        dst_count[other_id * hist_size + j] =
          hist_count_parent[parent_idx * hist_size + j] -
          dst_count[i * hist_size + j];
      }
    }
  }
}  // namespace core

template __global__ void hist_sum_dynamic<float, float>(
  float *dst_sum, unsigned *dst_count, const float *hist_sum_parent,
  const unsigned *hist_count_parent, const float *__restrict__ values,
  const unsigned *__restrict__ parent_count_iter,
  const unsigned *__restrict__ bin, const unsigned hist_size,
  const unsigned hist_size_bits, const bool use_trick, const size_t n);

template __global__ void hist_sum_dynamic<float2, float2>(
  float2 *dst_sum, unsigned *dst_count, const float2 *hist_sum_parent,
  const unsigned *hist_count_parent, const float2 *__restrict__ values,
  const unsigned *__restrict__ parent_count_iter,
  const unsigned *__restrict__ bin, const unsigned hist_size,
  const unsigned hist_size_bits, const bool use_trick, const size_t n);

template __global__ void hist_sum_dynamic<double, float>(
  double *dst_sum, unsigned *dst_count, const double *hist_sum_parent,
  const unsigned *hist_count_parent, const float *__restrict__ values,
  const unsigned *__restrict__ parent_count_iter,
  const unsigned *__restrict__ bin, const unsigned hist_size,
  const unsigned hist_size_bits, const bool use_trick, const size_t n);

template __global__ void hist_sum_dynamic<mydouble2, float2>(
  mydouble2 *dst_sum, unsigned *dst_count, const mydouble2 *hist_sum_parent,
  const unsigned *hist_count_parent, const float2 *__restrict__ values,
  const unsigned *__restrict__ parent_count_iter,
  const unsigned *__restrict__ bin, const unsigned hist_size,
  const unsigned hist_size_bits, const bool use_trick, const size_t n);

template <typename SUM_T, typename GRAD_T>
__global__ void hist_sum(SUM_T *dst_sum, unsigned *dst_count,
                         const GRAD_T *__restrict__ values,
                         const unsigned *__restrict__ parent_count_iter,
                         const unsigned *__restrict__ bin, const unsigned bins,
                         const size_t n) {
  for (unsigned int i = blockDim.x * blockIdx.x + threadIdx.x; i < n;
       i += gridDim.x * blockDim.x) {
    // TODO: Binary search?
    unsigned segment = 0;
    while ((i >= parent_count_iter[segment + 1])) {
      segment++;
    }

    unsigned idx = segment * bins + bin[i];
    SUM_T val = values[i];
    atomicAdd(&dst_sum[idx], val);
    atomicAdd(&dst_count[idx], 1);
  }
}

template __global__ void hist_sum<float, float>(
  float *dst_sum, unsigned *dst_count, const float *__restrict__ values,
  const unsigned *__restrict__ parent_count_iter,
  const unsigned *__restrict__ bin, const unsigned bins, const size_t n);

template __global__ void hist_sum<float2, float2>(
  float2 *dst_sum, unsigned *dst_count, const float2 *__restrict__ values,
  const unsigned *__restrict__ parent_count_iter,
  const unsigned *__restrict__ bin, const unsigned bins, const size_t n);

template __global__ void hist_sum<double, float>(
  double *dst_sum, unsigned *dst_count, const float *__restrict__ values,
  const unsigned *__restrict__ parent_count_iter,
  const unsigned *__restrict__ bin, const unsigned bins, const size_t n);

template __global__ void hist_sum<mydouble2, float2>(
  mydouble2 *dst_sum, unsigned *dst_count, const float2 *__restrict__ values,
  const unsigned *__restrict__ parent_count_iter,
  const unsigned *__restrict__ bin, const unsigned bins, const size_t n);

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

template <typename SUM_T, typename NODE_VALUE_T>
__global__ void gain_kernel(
  const SUM_T *const __restrict__ left_sum,
  const NODE_VALUE_T *const __restrict__ segments_fvalues, const unsigned span,
  const SUM_T *const __restrict__ parent_sum_iter,
  const unsigned int *const __restrict__ parent_count_iter, const size_t n,
  const GainFunctionParameters parameters, my_atomics *res) {
  for (unsigned int i = blockDim.x * blockIdx.x + threadIdx.x; i < n;
       i += gridDim.x * blockDim.x) {
    if (i == 0) continue;

    const unsigned int fvalue = segments_fvalues[i];
    const unsigned int fvalue_prev = segments_fvalues[i - 1];
    if (fvalue != fvalue_prev) {
      // TODO: Binary search?
      unsigned segment = 0;
      while (i >= parent_count_iter[segment + 1]) {
        segment++;
      }

      if (i == parent_count_iter[segment + 1]) continue;
      const SUM_T left_sum_offset = parent_sum_iter[segment];
      const SUM_T left_sum_value = left_sum[i] - left_sum_offset;

      const size_t left_count_offset = parent_count_iter[segment];
      const size_t left_count_value = i - left_count_offset;

      const SUM_T total_sum = parent_sum_iter[segment + 1] - left_sum_offset;
      const size_t total_count =
        parent_count_iter[segment + 1] - left_count_offset;

      const float gain = gain_func(left_sum_value, total_sum, left_count_value,
                                   total_count, parameters);
      if (gain > 0.0) {
        updateAtomicMax(&(res[segment].ulong), gain, i);
      }
    }
  }
}

template __global__ void gain_kernel<float, unsigned char>(
  const float *const __restrict__ left_sum,
  const unsigned char *const __restrict__ segments_fvalues, const unsigned span,
  const float *const __restrict__ parent_sum_iter,
  const unsigned int *const __restrict__ parent_count_iter, const size_t n,
  const GainFunctionParameters parameters, my_atomics *res);

template __global__ void gain_kernel<float, unsigned short>(
  const float *const __restrict__ left_sum,
  const unsigned short *const __restrict__ segments_fvalues,
  const unsigned span, const float *const __restrict__ parent_sum_iter,
  const unsigned int *const __restrict__ parent_count_iter, const size_t n,
  const GainFunctionParameters parameters, my_atomics *res);

template __global__ void gain_kernel<float, unsigned int>(
  const float *const __restrict__ left_sum,
  const unsigned int *const __restrict__ segments_fvalues, const unsigned span,
  const float *const __restrict__ parent_sum_iter,
  const unsigned int *const __restrict__ parent_count_iter, const size_t n,
  const GainFunctionParameters parameters, my_atomics *res);

template __global__ void gain_kernel<float, unsigned long>(
  const float *const __restrict__ left_sum,
  const unsigned long *const __restrict__ segments_fvalues, const unsigned span,
  const float *const __restrict__ parent_sum_iter,
  const unsigned int *const __restrict__ parent_count_iter, const size_t n,
  const GainFunctionParameters parameters, my_atomics *res);

template __global__ void gain_kernel<float, unsigned long long>(
  const float *const __restrict__ left_sum,
  const unsigned long long *const __restrict__ segments_fvalues,
  const unsigned span, const float *const __restrict__ parent_sum_iter,
  const unsigned int *const __restrict__ parent_count_iter, const size_t n,
  const GainFunctionParameters parameters, my_atomics *res);

template __global__ void gain_kernel<double, unsigned char>(
  const double *const __restrict__ left_sum,
  const unsigned char *const __restrict__ segments_fvalues, const unsigned span,
  const double *const __restrict__ parent_sum_iter,
  const unsigned int *const __restrict__ parent_count_iter, const size_t n,
  const GainFunctionParameters parameters, my_atomics *res);

template __global__ void gain_kernel<double, unsigned short>(
  const double *const __restrict__ left_sum,
  const unsigned short *const __restrict__ segments_fvalues,
  const unsigned span, const double *const __restrict__ parent_sum_iter,
  const unsigned int *const __restrict__ parent_count_iter, const size_t n,
  const GainFunctionParameters parameters, my_atomics *res);

template __global__ void gain_kernel<double, unsigned int>(
  const double *const __restrict__ left_sum,
  const unsigned int *const __restrict__ segments_fvalues, const unsigned span,
  const double *const __restrict__ parent_sum_iter,
  const unsigned int *const __restrict__ parent_count_iter, const size_t n,
  const GainFunctionParameters parameters, my_atomics *res);

template __global__ void gain_kernel<double, unsigned long>(
  const double *const __restrict__ left_sum,
  const unsigned long *const __restrict__ segments_fvalues, const unsigned span,
  const double *const __restrict__ parent_sum_iter,
  const unsigned int *const __restrict__ parent_count_iter, const size_t n,
  const GainFunctionParameters parameters, my_atomics *res);

template __global__ void gain_kernel<double, unsigned long long>(
  const double *const __restrict__ left_sum,
  const unsigned long long *const __restrict__ segments_fvalues,
  const unsigned span, const double *const __restrict__ parent_sum_iter,
  const unsigned int *const __restrict__ parent_count_iter, const size_t n,
  const GainFunctionParameters parameters, my_atomics *res);

template __global__ void gain_kernel<float2, unsigned char>(
  const float2 *const __restrict__ left_sum,
  const unsigned char *const __restrict__ segments_fvalues, const unsigned span,
  const float2 *const __restrict__ parent_sum_iter,
  const unsigned int *const __restrict__ parent_count_iter, const size_t n,
  const GainFunctionParameters parameters, my_atomics *res);

template __global__ void gain_kernel<float2, unsigned short>(
  const float2 *const __restrict__ left_sum,
  const unsigned short *const __restrict__ segments_fvalues,
  const unsigned span, const float2 *const __restrict__ parent_sum_iter,
  const unsigned int *const __restrict__ parent_count_iter, const size_t n,
  const GainFunctionParameters parameters, my_atomics *res);

template __global__ void gain_kernel<float2, unsigned int>(
  const float2 *const __restrict__ left_sum,
  const unsigned int *const __restrict__ segments_fvalues, const unsigned span,
  const float2 *const __restrict__ parent_sum_iter,
  const unsigned int *const __restrict__ parent_count_iter, const size_t n,
  const GainFunctionParameters parameters, my_atomics *res);

template __global__ void gain_kernel<float2, unsigned long>(
  const float2 *const __restrict__ left_sum,
  const unsigned long *const __restrict__ segments_fvalues, const unsigned span,
  const float2 *const __restrict__ parent_sum_iter,
  const unsigned int *const __restrict__ parent_count_iter, const size_t n,
  const GainFunctionParameters parameters, my_atomics *res);

template __global__ void gain_kernel<float2, unsigned long long>(
  const float2 *const __restrict__ left_sum,
  const unsigned long long *const __restrict__ segments_fvalues,
  const unsigned span, const float2 *const __restrict__ parent_sum_iter,
  const unsigned int *const __restrict__ parent_count_iter, const size_t n,
  const GainFunctionParameters parameters, my_atomics *res);

template __global__ void gain_kernel<mydouble2, unsigned char>(
  const mydouble2 *const __restrict__ left_sum,
  const unsigned char *const __restrict__ segments_fvalues, const unsigned span,
  const mydouble2 *const __restrict__ parent_sum_iter,
  const unsigned int *const __restrict__ parent_count_iter, const size_t n,
  const GainFunctionParameters parameters, my_atomics *res);

template __global__ void gain_kernel<mydouble2, unsigned short>(
  const mydouble2 *const __restrict__ left_sum,
  const unsigned short *const __restrict__ segments_fvalues,
  const unsigned span, const mydouble2 *const __restrict__ parent_sum_iter,
  const unsigned int *const __restrict__ parent_count_iter, const size_t n,
  const GainFunctionParameters parameters, my_atomics *res);

template __global__ void gain_kernel<mydouble2, unsigned int>(
  const mydouble2 *const __restrict__ left_sum,
  const unsigned int *const __restrict__ segments_fvalues, const unsigned span,
  const mydouble2 *const __restrict__ parent_sum_iter,
  const unsigned int *const __restrict__ parent_count_iter, const size_t n,
  const GainFunctionParameters parameters, my_atomics *res);

template __global__ void gain_kernel<mydouble2, unsigned long>(
  const mydouble2 *const __restrict__ left_sum,
  const unsigned long *const __restrict__ segments_fvalues, const unsigned span,
  const mydouble2 *const __restrict__ parent_sum_iter,
  const unsigned int *const __restrict__ parent_count_iter, const size_t n,
  const GainFunctionParameters parameters, my_atomics *res);

template __global__ void gain_kernel<mydouble2, unsigned long long>(
  const mydouble2 *const __restrict__ left_sum,
  const unsigned long long *const __restrict__ segments_fvalues,
  const unsigned span, const mydouble2 *const __restrict__ parent_sum_iter,
  const unsigned int *const __restrict__ parent_count_iter, const size_t n,
  const GainFunctionParameters parameters, my_atomics *res);

}  // namespace core
}  // namespace arboretum