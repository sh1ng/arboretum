#ifndef SRC_CORE_BUILDER_H
#define SRC_CORE_BUILDER_H
// #define CUB_STDERR
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/system/cuda/experimental/pinned_allocator.h>
#include "best_splits.h"
#include "common.h"
#include "cub/cub.cuh"
#include "cub/iterator/discard_output_iterator.cuh"
#include "cuda_helpers.h"
#include "cuda_runtime.h"
#include "histogram.h"
#include "param.h"

namespace arboretum {
namespace core {
using thrust::device_vector;
using thrust::host_vector;
using thrust::cuda::experimental::pinned_allocator;

#define HIST_SUM_NO_DATA (unsigned short)-1
#define HIST_SUM_BLOCK_DIM 128
#define HIST_SUM_ITEMS_PER_THREAD 10

template <typename SUM_T>
__global__ void update(SUM_T *sum_dst, unsigned *count_dst,
                       const SUM_T *parent_sum, const unsigned *parent_count,
                       const SUM_T *sum_src, const unsigned *count_src,
                       const unsigned n);

/**
 * @brief partitions binary tree's leaf to left of right
 *
 */
template <typename T>
struct PartitioningLeafs {
  __host__ __device__ __forceinline__ PartitioningLeafs(const unsigned level)
      : level(level) {}
  const unsigned level;
  __host__ __device__ __forceinline__ bool operator()(const T &a) const {
    return (a >> level) % 2 == 0;
  }
};

struct GainFunctionParameters {
  const unsigned int min_leaf_size;
  const float hess;
  const float gamma_absolute;
  const float gamma_relative;
  const float lambda;
  const float alpha;
  __host__ __device__ GainFunctionParameters(const unsigned int min_leaf_size,
                                             const float hess,
                                             const float gamma_absolute,
                                             const float gamma_relative,
                                             const float lambda,
                                             const float alpha)
      : min_leaf_size(min_leaf_size),
        hess(hess),
        gamma_absolute(gamma_absolute),
        gamma_relative(gamma_relative),
        lambda(lambda),
        alpha(alpha) {}
};

__forceinline__ __device__ unsigned long long int updateAtomicMax(
  unsigned long long int *address, float val1, unsigned int val2) {
  my_atomics loc, loctest;
  loc.floats[0] = val1;
  loc.ints[1] = val2;
  loctest.ulong = *address;
  while (loctest.floats[0] < val1)
    loctest.ulong = atomicCAS(address, loctest.ulong, loc.ulong);
  return loctest.ulong;
}

template <class T1, class T2>
__global__ void gather_kernel(const unsigned int *const __restrict__ position,
                              const T1 *const __restrict__ in1, T1 *out1,
                              const T2 *const __restrict__ in2, T2 *out2,
                              const size_t n) {
  for (size_t i = blockDim.x * blockIdx.x + threadIdx.x; i < n;
       i += gridDim.x * blockDim.x) {
    out1[i] = in1[position[i]];
    out2[i] = in2[position[i]];
  }
}

__forceinline__ __device__ __host__ float gain_func(
  const double2 left_sum, const double2 total_sum, const size_t left_count,
  const size_t total_count, const GainFunctionParameters &params) {
  const double2 right_sum = total_sum - left_sum;
  if (left_count >= params.min_leaf_size &&
      (total_count - left_count) >= params.min_leaf_size &&
      std::abs(left_sum.y) >= params.hess &&
      std::abs(right_sum.y) >= params.hess) {
    const float l = (left_sum.x * left_sum.x) / (left_sum.y + params.lambda);
    const float r = (right_sum.x * right_sum.x) / (right_sum.y + params.lambda);
    const float p = (total_sum.x * total_sum.x) / (total_sum.y + params.lambda);
    const float diff = l + r - p;
    return (diff > params.gamma_absolute && diff > params.gamma_relative * p) *
           diff;
  } else {
    return 0.0;
  }
}

__forceinline__ __device__ __host__ float gain_func(
  const float2 left_sum, const float2 total_sum, const size_t left_count,
  const size_t total_count, const GainFunctionParameters &params) {
  const float2 right_sum = total_sum - left_sum;
  if (left_count >= params.min_leaf_size &&
      (total_count - left_count) >= params.min_leaf_size &&
      std::abs(left_sum.y) >= params.hess &&
      std::abs(right_sum.y) >= params.hess) {
    const float l = (left_sum.x * left_sum.x) / (left_sum.y + params.lambda);
    const float r = (right_sum.x * right_sum.x) / (right_sum.y + params.lambda);
    const float p = (total_sum.x * total_sum.x) / (total_sum.y + params.lambda);
    const float diff = l + r - p;
    return (diff > params.gamma_absolute && diff > params.gamma_relative * p) *
           diff;
  } else {
    return 0.0;
  }
}

__forceinline__ __device__ __host__ float gain_func(
  const float left_sum, const float total_sum, const size_t left_count,
  const size_t total_count, const GainFunctionParameters &params) {
  const size_t right_count = total_count - left_count;
  if (left_count >= params.min_leaf_size &&
      right_count >= params.min_leaf_size) {
    const float right_sum = total_sum - left_sum;
    const float l = left_sum * left_sum / (left_count + params.lambda);
    const float r = right_sum * right_sum / (right_count + params.lambda);
    const float p = total_sum * total_sum / (total_count + params.lambda);
    const float diff = l + r - p;
    return (diff > params.gamma_absolute && diff > params.gamma_relative * p) *
           diff;
  } else {
    return 0.0;
  }
}

__forceinline__ __device__ __host__ float gain_func(
  const double left_sum, const double total_sum, const size_t left_count,
  const size_t total_count, const GainFunctionParameters &params) {
  const size_t right_count = total_count - left_count;
  if (left_count >= params.min_leaf_size &&
      right_count >= params.min_leaf_size) {
    const double right_sum = total_sum - left_sum;
    const double l = left_sum * left_sum / (left_count + params.lambda);
    const double r = right_sum * right_sum / (right_count + params.lambda);
    const double p = total_sum * total_sum / (total_count + params.lambda);
    const double diff = l + r - p;
    return (diff > params.gamma_absolute && diff > params.gamma_relative * p) *
           diff;
  } else {
    return 0.0;
  }
}

template <typename NODE_T, typename NODE_VALUE_T>
__global__ void assign_kernel(const unsigned int *const __restrict__ fvalue,
                              const NODE_T *const __restrict__ segments,
                              const unsigned char fvalue_size,
                              NODE_VALUE_T *out, const size_t n);

template <typename SUM_T, typename NODE_VALUE_T>
__global__ void gain_kernel(
  const SUM_T *const __restrict__ left_sum,
  const NODE_VALUE_T *const __restrict__ segments_fvalues, const unsigned span,
  const SUM_T *const __restrict__ parent_sum_iter,
  const unsigned int *const __restrict__ parent_count_iter, const size_t n,
  const GainFunctionParameters parameters, my_atomics *res);

template <typename SUM_T>
__global__ void hist_gain_kernel(
  const SUM_T *const __restrict__ hist_prefix_sum,
  const unsigned *const __restrict__ hist_prefix_count,
  const SUM_T *const __restrict__ parent_sum,
  const unsigned int *const __restrict__ parent_count, const unsigned hist_size,
  const size_t n, const GainFunctionParameters parameters, my_atomics *res) {
  for (unsigned int i = blockDim.x * blockIdx.x + threadIdx.x; i < n;
       i += gridDim.x * blockDim.x) {
    unsigned segment = i / hist_size;

    const SUM_T left_sum_offset = parent_sum[segment];
    const SUM_T left_sum_value = hist_prefix_sum[i] - left_sum_offset;

    const unsigned left_count_offset = parent_count[segment];
    const unsigned left_count_value = hist_prefix_count[i] - left_count_offset;

    const SUM_T total_sum = parent_sum[segment + 1] - left_sum_offset;
    const unsigned total_count = parent_count[segment + 1] - left_count_offset;

    const float gain = gain_func(left_sum_value, total_sum, left_count_value,
                                 total_count, parameters);
    if (gain > 0.0) {
      updateAtomicMax(&(res[segment].ulong), gain, i);
    }
  }
}

template <typename SUM_T, typename NODE_VALUE_T>
__global__ void gain_kernel_category(
  const SUM_T *const __restrict__ category_sum,
  const unsigned int *const __restrict__ category_count,
  const NODE_VALUE_T *const __restrict__ segments_fvalues,
  const unsigned char fvalue_size, const NODE_VALUE_T mask,
  const SUM_T *const __restrict__ parent_sum,
  const unsigned int *const __restrict__ parent_count,
  const unsigned int *const __restrict__ n,
  const GainFunctionParameters parameters, my_atomics *res) {
  for (unsigned int i = blockDim.x * blockIdx.x + threadIdx.x; i < n[0];
       i += gridDim.x * blockDim.x) {
    const NODE_VALUE_T fvalue_segment = segments_fvalues[i];

    const NODE_VALUE_T segment = fvalue_segment >> fvalue_size;

    const SUM_T left_sum_value = category_sum[i];

    const size_t left_count_value = category_count[i];

    const SUM_T total_sum = parent_sum[segment + 1] - parent_sum[segment];
    const size_t total_count =
      parent_count[segment + 1] - parent_count[segment];

    const float gain = gain_func(left_sum_value, total_sum, left_count_value,
                                 total_count, parameters);
    if (gain > 0.0) {
      updateAtomicMax(&(res[segment].ulong), gain, i);
    }
  }
}

template <typename NODE_T>
__global__ void apply_split(NODE_T *row2Node, const unsigned *fvalues,
                            const unsigned int threshold,
                            const unsigned int level, const unsigned n) {
  for (unsigned i = blockDim.x * blockIdx.x + threadIdx.x; i < n;
       i += gridDim.x * blockDim.x) {
    const unsigned right = unsigned(fvalues[i] >= threshold);

    unsigned int current = row2Node[i];
    // const unsigned tmp = current;
    // clear
    current &= ~(1ULL << (level));
    // set
    current |= right << level;
    row2Node[i] = current;
  }
}

template <typename NODE_T, typename SUM_T>
__global__ void filter_apply_candidates(
  float *gain, int *features, SUM_T *sum, unsigned *split, unsigned *count,
  unsigned *node_size_prefix_sum_next, SUM_T *node_sum_prefix_sum_next,
  const my_atomics *candidates, const SUM_T *split_sum, const unsigned *fvalue,
  const unsigned *fvalue_sorted, NODE_T *row2Node,
  const unsigned *node_size_prefix_sum, const SUM_T *node_sum_prefix_sum,
  const int feature, const unsigned level, const unsigned n) {
  for (unsigned i = blockDim.x * blockIdx.x + threadIdx.x; i < n;
       i += gridDim.x * blockDim.x) {
    const unsigned node_start = node_size_prefix_sum[i];
    const unsigned node_end = node_size_prefix_sum[i + 1];
    const unsigned node_size = node_end - node_start;
    const float gain_ = candidates[i].floats[0];
    const unsigned idx = candidates[i].ints[1];
    const SUM_T node_start_sum = node_sum_prefix_sum[i];
    const SUM_T node_end_sum = node_sum_prefix_sum[i + 1];
    if (node_size > 0) {
      if (gain[i] < gain_) {
        const SUM_T split_sum_value = split_sum[idx];
        gain[i] = gain_;
        features[i] = feature;
        sum[i] = split_sum_value - node_start_sum;
        count[i] = idx - node_start;
        unsigned threshold = fvalue_sorted[idx];
        split[i] = threshold;

        unsigned block_size = MAX_THREADS > node_size ? node_size : MAX_THREADS;
        unsigned grid_size =
          unsigned((node_size + block_size - 1) / block_size);
        cudaStream_t s;
        DEVICE_OK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
        apply_split<NODE_T><<<grid_size, block_size, 0, s>>>(
          row2Node + node_start, fvalue + node_start, threshold, level,
          node_size);
        DEVICE_OK(cudaDeviceSynchronize());
        DEVICE_OK(cudaStreamDestroy(s));
        node_size_prefix_sum_next[2 * i + 1] = idx;
        node_size_prefix_sum_next[2 * i + 2] = node_end;
        node_sum_prefix_sum_next[2 * i + 1] = split_sum_value;
        node_sum_prefix_sum_next[2 * i + 2] = node_end_sum;
      } else if (gain[i] == 0 && features[i] == -1) {
        sum[i] = node_end_sum - node_start_sum;
        split[i] = (unsigned)-1;
        count[i] = node_size;
        node_size_prefix_sum_next[2 * i + 1] =
          node_size_prefix_sum_next[2 * i + 2] = node_end;
        node_sum_prefix_sum_next[2 * i + 1] =
          node_sum_prefix_sum_next[2 * i + 2] = node_end_sum;
      }
    } else {
      node_size_prefix_sum_next[2 * i + 1] =
        node_size_prefix_sum_next[2 * i + 2] = node_end;
      node_sum_prefix_sum_next[2 * i + 1] =
        node_sum_prefix_sum_next[2 * i + 2] = node_end_sum;
    }
  }
}

template <typename NODE_T, typename SUM_T>
__global__ void hist_apply_candidates(
  float *gain, int *features, SUM_T *sum, unsigned *split, unsigned *count,
  unsigned *node_size_prefix_sum_next, SUM_T *node_sum_prefix_sum_next,
  const my_atomics *candidates, const SUM_T *split_sum,
  const unsigned *split_count, const unsigned *fvalue, NODE_T *row2Node,
  const unsigned *node_size_prefix_sum, const SUM_T *node_sum_prefix_sum,
  const int feature, const unsigned level, const unsigned hist_size,
  const unsigned n) {
  for (unsigned i = blockDim.x * blockIdx.x + threadIdx.x; i < n;
       i += gridDim.x * blockDim.x) {
    const unsigned node_start = node_size_prefix_sum[i];
    const unsigned node_end = node_size_prefix_sum[i + 1];
    const unsigned node_size = node_end - node_start;
    const float gain_ = candidates[i].floats[0];
    const unsigned idx = candidates[i].ints[1];
    const unsigned split_count_value = split_count[idx];
    const SUM_T node_start_sum = node_sum_prefix_sum[i];
    const SUM_T node_end_sum = node_sum_prefix_sum[i + 1];

    if (node_size > 0) {
      if (gain[i] < gain_) {
        gain[i] = gain_;
        features[i] = feature;
        sum[i] = split_sum[idx] - node_start_sum;
        count[i] = split_count_value - node_start;
        unsigned threshold = idx % hist_size;
        split[i] = threshold;

        unsigned block_size = MAX_THREADS > node_size ? node_size : MAX_THREADS;
        unsigned grid_size =
          unsigned((node_size + block_size - 1) / block_size);
        cudaStream_t s;
        DEVICE_OK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
        apply_split<NODE_T><<<grid_size, block_size, 0, s>>>(
          row2Node + node_start, fvalue + node_start, threshold + 1, level,
          node_size);
        DEVICE_OK(cudaDeviceSynchronize());
        DEVICE_OK(cudaStreamDestroy(s));
        node_size_prefix_sum_next[2 * i + 1] = split_count_value;
        node_size_prefix_sum_next[2 * i + 2] = node_end;
        node_sum_prefix_sum_next[2 * i + 1] = split_sum[idx];
        node_sum_prefix_sum_next[2 * i + 2] = node_end_sum;
      } else if (gain[i] == 0 && features[i] == -1) {  // no split, all to left
        sum[i] = node_end_sum - node_start_sum;
        split[i] = (unsigned)-1;
        count[i] = node_size;
        node_size_prefix_sum_next[2 * i + 1] =
          node_size_prefix_sum_next[2 * i + 2] = node_end;
        node_sum_prefix_sum_next[2 * i + 1] =
          node_sum_prefix_sum_next[2 * i + 2] = node_end_sum;
      }
      // ignore not-optimal splits
    } else {
      node_size_prefix_sum_next[2 * i + 1] =
        node_size_prefix_sum_next[2 * i + 2] = node_end;
      node_sum_prefix_sum_next[2 * i + 1] =
        node_sum_prefix_sum_next[2 * i + 2] = node_end_sum;
    }
  }
}

template <typename NODE_T, typename T, int OFFSET = 2>
__global__ void partition(T *dst, const NODE_T *row2Node, const T *src,
                          const unsigned *offsets, const unsigned level,
                          const size_t temp_bytes, void *temp_bytes_ptr,
                          const size_t size, const unsigned n) {
  for (unsigned i = blockDim.x * blockIdx.x + threadIdx.x; i < n;
       i += gridDim.x * blockDim.x) {
    assert(offsets[OFFSET * i + OFFSET] >= offsets[OFFSET * i]);
    const unsigned segment_size =
      offsets[OFFSET * i + OFFSET] - offsets[OFFSET * i];
    if (segment_size != 0) {
      cudaStream_t s;
      DEVICE_OK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));

      PartitioningLeafs<NODE_T> conversion_op(level);

      cub::DiscardOutputIterator<unsigned> discard_itr;

      cub::TransformInputIterator<bool, PartitioningLeafs<NODE_T>,
                                  const NODE_T *>
        partition_itr(row2Node + offsets[OFFSET * i], conversion_op);

      size_t offset = (temp_bytes * offsets[OFFSET * i]) / size;
      size_t aux_memory_size = (temp_bytes * segment_size) / size;
      char *p = ((char *)temp_bytes_ptr) + offset;
      size_t tmp = 0;
      DEVICE_OK(cub::DevicePartition::Flagged(
        NULL, tmp, src + offsets[OFFSET * i], partition_itr,
        dst + offsets[OFFSET * i], discard_itr, segment_size));

      if (tmp > aux_memory_size) {
        DEVICE_OK(cudaMalloc(&p, tmp));
        aux_memory_size = tmp;
        DEVICE_OK(cub::DevicePartition::Flagged(
          p, aux_memory_size, src + offsets[OFFSET * i], partition_itr,
          dst + offsets[OFFSET * i], discard_itr, segment_size, s));
        DEVICE_OK(cudaDeviceSynchronize());
        DEVICE_OK(cudaFree(p));
      } else {
        DEVICE_OK(cub::DevicePartition::Flagged(
          p, aux_memory_size, src + offsets[OFFSET * i], partition_itr,
          dst + offsets[OFFSET * i], discard_itr, segment_size, s));
        DEVICE_OK(cudaDeviceSynchronize());
      }
      DEVICE_OK(cudaStreamDestroy(s));
    }
  }
}

template <typename SUM_T, typename GRAD_T>
__global__ void hist_sum(SUM_T *dst_sum, unsigned *dst_count,
                         const GRAD_T *__restrict__ values,
                         const unsigned *__restrict__ parent_count_iter,
                         const unsigned *__restrict__ bin,
                         const unsigned end_bit, const size_t n);

template <typename SUM_T, typename GRAD_T,
          int ITEMS_PER_THREAD = HIST_SUM_ITEMS_PER_THREAD>
__global__ void hist_sum_node(SUM_T *dst_sum, unsigned *dst_count,
                              const GRAD_T *__restrict__ values,
                              const unsigned *__restrict__ bin,
                              const unsigned end_bit, const unsigned segment,
                              const size_t n);

template <typename SUM_T, typename GRAD_T>
__global__ void hist_sum_dynamic(
  SUM_T *dst_sum, unsigned *dst_count, const SUM_T *hist_sum_parent,
  const unsigned *hist_count_parent, const GRAD_T *__restrict__ values,
  const unsigned *__restrict__ parent_count_iter,
  const unsigned *__restrict__ bin, const unsigned hist_size,
  const unsigned hist_size_bits, const bool use_trick, const size_t n);

template <typename NODE_T, typename GRAD_T, typename SUM_T>
class BaseGrower {
 public:
  BaseGrower(const size_t size, const unsigned depth,
             const BestSplit<SUM_T> *best, Histogram<SUM_T> *features_histogram,
             const InternalConfiguration *config)
      : size(size),
        depth(depth),
        gridSizeGain(0),
        blockSizeGain(0),
        gridSizeGather(0),
        blockSizeGather(0),
        temp_bytes_allocated(0),
        best(best),
        features_histogram(features_histogram),
        config(config) {
    OK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    OK(cudaStreamCreateWithFlags(&copy_d2h_stream, cudaStreamNonBlocking));
    OK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));

    compute1DInvokeConfig(size, &gridSizeGain, &blockSizeGain,
                          gain_kernel<SUM_T, NODE_T>);

    compute1DInvokeConfig(size, &gridSizeGather, &blockSizeGather,
                          gather_kernel<NODE_T, float>);

    PartitioningLeafs<NODE_T> conversion_op(0);

    cub::TransformInputIterator<bool, PartitioningLeafs<NODE_T>, NODE_T *>
      partition_itr((NODE_T *)nullptr, conversion_op);

    cub::DiscardOutputIterator<unsigned> discard_itr;

    size_t temp_storage_bytes = 0;

    OK(cub::DevicePartition::Flagged(NULL, temp_storage_bytes,
                                     (GRAD_T *)nullptr, partition_itr,
                                     (GRAD_T *)nullptr, discard_itr, size));

    temp_bytes_allocated = std::max(temp_bytes_allocated, temp_storage_bytes);

    temp_storage_bytes = 0;

    OK(cub::DevicePartition::Flagged(NULL, temp_storage_bytes,
                                     (NODE_T *)nullptr, partition_itr,
                                     (NODE_T *)nullptr, discard_itr, size));

    temp_bytes_allocated = std::max(temp_bytes_allocated, temp_storage_bytes);

    temp_storage_bytes = 0;

    OK(cub::DeviceReduce::Sum(NULL, temp_storage_bytes, (GRAD_T *)nullptr,
                              (SUM_T *)nullptr, size));

    temp_bytes_allocated = std::max(temp_bytes_allocated, temp_storage_bytes);

    grad_sorted.resize(size);
    fvalue.resize(size);
    result_d.resize(1 << this->depth);
  }

  ~BaseGrower() {
    OK(cudaFree(temp_bytes));
    OK(cudaStreamDestroy(stream));
    OK(cudaStreamDestroy(copy_d2h_stream));
    OK(cudaEventDestroy(event));
  }

  template <typename T, int OFFSET = 2>
  void Partition(T *src, const NODE_T *row2Node,
                 const device_vector<unsigned> &parent_node_count,
                 const unsigned level, const unsigned depth) {
    if (level != 0) {
      const unsigned length = 1 << (level - 1);
      int gridSize = 0;
      int blockSize = 0;

      compute1DInvokeConfig(length, &gridSize, &blockSize, partition<NODE_T, T>,
                            0, 1);
      partition<NODE_T, T, OFFSET><<<gridSize, blockSize, 0, stream>>>(
        (T *)thrust::raw_pointer_cast(grad_sorted.data()), row2Node, src,
        thrust::raw_pointer_cast(parent_node_count.data()), depth - level - 1,
        temp_bytes_allocated, temp_bytes, this->size, length);

      OK(cudaMemcpyAsync(src, (T *)thrust::raw_pointer_cast(grad_sorted.data()),
                         size * sizeof(T), cudaMemcpyDeviceToDevice, stream));
    }
  }

  template <typename T, int OFFSET = 2>
  void Partition(T *dst, T *src, const NODE_T *row2Node,
                 const device_vector<unsigned> &parent_node_count,
                 const unsigned level, const unsigned depth) {
    const unsigned length = 1 << level;
    if (this->config->dynamic_parallelism) {
      int gridSize = 0;
      int blockSize = 0;

      compute1DInvokeConfig(length, &gridSize, &blockSize,
                            partition<T, unsigned>, 0, 1);

      partition<NODE_T, T, 2><<<gridSize, blockSize, 0, this->stream>>>(
        dst, row2Node, src, thrust::raw_pointer_cast(parent_node_count.data()),
        depth - level - 2, this->temp_bytes_allocated, this->temp_bytes,
        this->size, length);
    }
  }

  template <typename T, int OFFSET = 2>
  void Partition(T *dst, T *src, const NODE_T *row2Node,
                 const host_vector<unsigned> &parent_node_count,
                 const unsigned level, const unsigned depth) {
    const unsigned length = 1 << level;

    for (unsigned i = 0; i < length; ++i) {
      unsigned start = parent_node_count[i * OFFSET];
      unsigned size =
        parent_node_count[(i + 1) * OFFSET] - parent_node_count[i * OFFSET];
      if (size != 0) {
        PartitioningLeafs<NODE_T> conversion_op(depth - level - 2);

        cub::TransformInputIterator<bool, PartitioningLeafs<NODE_T>, NODE_T *>
          partition_itr((NODE_T *)row2Node + start, conversion_op);

        cub::DiscardOutputIterator<unsigned> discard_itr;

        OK(cub::DevicePartition::Flagged(
          this->temp_bytes, this->temp_bytes_allocated, src + start,
          partition_itr, dst + start, discard_itr, size, this->stream));
      }
    }
  }
  cudaStream_t stream;
  cudaStream_t copy_d2h_stream;
  cudaEvent_t event;
  device_vector<SUM_T> sum;
  device_vector<unsigned int> fvalue;
  device_vector<my_atomics> result_d;
  size_t temp_bytes_allocated;
  void *temp_bytes;
  const size_t size;
  const unsigned depth;

  int blockSizeGain;
  int gridSizeGain;

  int blockSizeGather;
  int gridSizeGather;
  device_vector<GRAD_T> grad_sorted;
  unsigned *d_fvalue_partitioned;
  const BestSplit<SUM_T> *best;
  Histogram<SUM_T> *features_histogram;
  const InternalConfiguration *config;
};

template <typename NODE_T, typename GRAD_T, typename SUM_T>
class HistTreeGrower : public BaseGrower<NODE_T, GRAD_T, SUM_T> {
 public:
  HistTreeGrower(const size_t size, const unsigned depth,
                 const unsigned hist_size, const BestSplit<SUM_T> *best,
                 Histogram<SUM_T> *features_histogram,
                 const InternalConfiguration *config)
      : BaseGrower<NODE_T, GRAD_T, SUM_T>(size, depth, best, features_histogram,
                                          config),
        hist_size(hist_size) {
    assert(hist_size > 0);
    unsigned index = hist_size;
    hist_size_bits = 1;
    while (index >>= 1) ++hist_size_bits;

    const size_t total_hist_size = hist_size * 2 * ((1 << depth) - 1);
    this->sum.resize(total_hist_size);
    hist_prefix_sum.resize(total_hist_size);
    hist_bin_count.resize(total_hist_size);
    hist_prefix_count.resize(total_hist_size);

    cudaFuncSetCacheConfig(hist_sum_dynamic<SUM_T, GRAD_T>,
                           cudaFuncCachePreferShared);

    cudaFuncSetCacheConfig(
      hist_sum_node<SUM_T, GRAD_T, HIST_SUM_ITEMS_PER_THREAD>,
      cudaFuncCachePreferShared);

    size_t temp_storage_bytes = 0;

    PartitioningLeafs<NODE_T> conversion_op(0);

    cub::TransformInputIterator<bool, PartitioningLeafs<NODE_T>, NODE_T *>
      partition_itr((NODE_T *)nullptr, conversion_op);

    cub::DiscardOutputIterator<unsigned> discard_itr;

    OK(cub::DevicePartition::Flagged(NULL, temp_storage_bytes,
                                     (GRAD_T *)nullptr, partition_itr,
                                     (GRAD_T *)nullptr, discard_itr, size));

    this->temp_bytes_allocated =
      std::max(this->temp_bytes_allocated, temp_storage_bytes);

    temp_storage_bytes = 0;

    OK(cub::DevicePartition::Flagged(NULL, temp_storage_bytes,
                                     (NODE_T *)nullptr, partition_itr,
                                     (NODE_T *)nullptr, discard_itr, size));

    this->temp_bytes_allocated =
      std::max(this->temp_bytes_allocated, temp_storage_bytes);

    temp_storage_bytes = 0;

    OK(cub::DevicePartition::Flagged(
      NULL, temp_storage_bytes, (GRAD_T *)nullptr, partition_itr,
      (GRAD_T *)nullptr, discard_itr, size / (1 << this->depth)));

    this->temp_bytes_allocated = std::max(
      this->temp_bytes_allocated, temp_storage_bytes * (1 << this->depth));

    temp_storage_bytes = 0;

    OK(cub::DevicePartition::Flagged(
      NULL, temp_storage_bytes, (NODE_T *)nullptr, partition_itr,
      (NODE_T *)nullptr, discard_itr, size / (1 << this->depth)));

    this->temp_bytes_allocated = std::max(
      this->temp_bytes_allocated, temp_storage_bytes * (1 << this->depth));

    temp_storage_bytes = 0;

    cub::Sum sum_op;

    OK(cub::DeviceScan::InclusiveScan(
      NULL, temp_storage_bytes, (SUM_T *)nullptr, (SUM_T *)nullptr, sum_op,
      (1 << this->depth) * this->hist_size));

    this->temp_bytes_allocated =
      std::max(this->temp_bytes_allocated, temp_storage_bytes);

    temp_storage_bytes = 0;

    OK(cub::DeviceScan::InclusiveSum(NULL, temp_storage_bytes,
                                     (unsigned *)nullptr, (unsigned *)nullptr,
                                     (1 << this->depth) * this->hist_size));

    this->temp_bytes_allocated =
      std::max(this->temp_bytes_allocated, temp_storage_bytes);

    OK(cudaMalloc(&this->temp_bytes, this->temp_bytes_allocated));
  }

  static void HistSum(SUM_T *sum, unsigned *bin_count,
                      const SUM_T *hist_sum_parent,
                      const unsigned *hist_count_parent, const GRAD_T *grad,
                      const unsigned *node_size, const unsigned *fvalue,
                      const unsigned hist_size_bits, const unsigned hist_size,
                      const unsigned size, const bool use_trick,
                      cudaStream_t stream = 0) {
    constexpr unsigned blockSize = 1;
    const unsigned gridSize = (size + blockSize - 1) / blockSize;

    hist_sum_dynamic<SUM_T, GRAD_T><<<gridSize, blockSize, 0, stream>>>(
      sum, bin_count, hist_sum_parent, hist_count_parent, grad, node_size,
      fvalue, hist_size, hist_size_bits, use_trick, size);
  }

  static void HistSumStatic(SUM_T *sum, unsigned *bin_count,
                            const SUM_T *hist_sum_parent,
                            const unsigned *hist_count_parent,
                            const GRAD_T *grad, const unsigned *node_size,
                            const unsigned *fvalue,
                            const unsigned hist_size_bits,
                            const unsigned hist_size, const unsigned size,
                            const bool use_trick, cudaStream_t stream = 0) {
    if (use_trick) {
      assert(size % 2 == 0);
      for (unsigned i = 0; i < size / 2; ++i) {
        unsigned left_segment_id = i * 2;
        unsigned right_segment_id = i * 2 + 1;
        unsigned smaller_segment_id = right_segment_id;
        unsigned larger_segment_id = left_segment_id;
        if (node_size[left_segment_id + 1] - node_size[left_segment_id] <=
            node_size[right_segment_id + 1] - node_size[right_segment_id]) {
          smaller_segment_id = left_segment_id;
          larger_segment_id = right_segment_id;
        }

        unsigned segment_start = node_size[smaller_segment_id];
        unsigned segment_size =
          node_size[smaller_segment_id + 1] - node_size[smaller_segment_id];
        if (segment_size != 0)
          HistTreeGrower<NODE_T, GRAD_T, SUM_T>::HistSumSingleNode(
            sum + smaller_segment_id * hist_size,
            bin_count + smaller_segment_id * hist_size, grad + segment_start,
            node_size + smaller_segment_id, fvalue + segment_start,
            hist_size_bits, segment_size, stream);

        const unsigned block_size = std::min(unsigned(1024), hist_size);
        const unsigned grid_size = (hist_size + block_size - 1) / block_size;

        update<SUM_T><<<grid_size, block_size, 0, stream>>>(
          sum + larger_segment_id * hist_size,
          bin_count + larger_segment_id * hist_size,
          hist_sum_parent + i * hist_size, hist_count_parent + i * hist_size,
          sum + smaller_segment_id * hist_size,
          bin_count + smaller_segment_id * hist_size, hist_size);
      }
    } else {
      for (unsigned i = 0; i < size; ++i) {
        unsigned segment_start = node_size[i];
        unsigned segment_size = node_size[i + 1] - node_size[i];
        if (segment_size != 0)
          HistTreeGrower<NODE_T, GRAD_T, SUM_T>::HistSumSingleNode(
            sum + i * hist_size, bin_count + i * hist_size,
            grad + segment_start, node_size + i, fvalue + segment_start,
            hist_size_bits, segment_size, stream);
      }
    }
  }

  static void HistSumSingleNode(SUM_T *sum, unsigned *bin_count,
                                const GRAD_T *grad, const unsigned *node_size,
                                const unsigned *fvalue,
                                const unsigned hist_size_bits,
                                const unsigned size, cudaStream_t stream = 0) {
    constexpr unsigned blockSize = HIST_SUM_BLOCK_DIM;
    const unsigned gridSize =
      (size + (blockSize * HIST_SUM_ITEMS_PER_THREAD) - 1) /
      (blockSize * HIST_SUM_ITEMS_PER_THREAD);

    hist_sum_node<SUM_T, GRAD_T><<<gridSize, blockSize, 0, stream>>>(
      sum, bin_count, grad, fvalue, hist_size_bits, 0, size);
  }

  static void HistSumNaive(SUM_T *sum, unsigned *bin_count, const GRAD_T *grad,
                           const unsigned *node_size, const unsigned *fvalue,
                           const unsigned hist_size, const unsigned size,
                           cudaStream_t stream = 0) {
    constexpr unsigned blockSize = 1024;
    const unsigned gridSize = (size + blockSize - 1) / blockSize;

    hist_sum<SUM_T, GRAD_T><<<gridSize, blockSize, 0, stream>>>(
      sum, bin_count, grad, node_size, fvalue, hist_size, size);
  }

  template <typename NODE_VALUE_T>
  inline void ProcessDenseFeature(
    const device_vector<NODE_T> &row2Node, const device_vector<GRAD_T> &grad_d,
    unsigned int *fvalue_d, unsigned int *fvalue_h,
    const device_vector<SUM_T> &parent_node_sum,
    const device_vector<unsigned int> &parent_node_count,
    const unsigned char fvalue_size, const unsigned level, const unsigned depth,
    const GainFunctionParameters gain_param, const bool partition_only,
    const int fid) {
    const unsigned lenght = 1 << level;

    OK(cudaMemsetAsync(thrust::raw_pointer_cast(this->result_d.data()), 0,
                       lenght * sizeof(my_atomics), this->stream));
    OK(cudaMemsetAsync(thrust::raw_pointer_cast(this->sum.data()), 0,
                       this->hist_size * lenght * sizeof(SUM_T), this->stream));
    OK(cudaMemsetAsync(thrust::raw_pointer_cast(this->hist_bin_count.data()), 0,
                       this->hist_size * lenght * sizeof(unsigned),
                       this->stream));

    unsigned int *fvalue_tmp = NULL;

    if (fvalue_d != nullptr) {
      fvalue_tmp = fvalue_d;
    } else {
      OK(cudaMemcpyAsync(thrust::raw_pointer_cast(this->fvalue.data()),
                         fvalue_h, this->size * sizeof(unsigned int),
                         cudaMemcpyHostToDevice, this->stream));
      fvalue_tmp = thrust::raw_pointer_cast(this->fvalue.data());
    }

    if (level != 0) {
      this->d_fvalue_partitioned =
        (unsigned *)thrust::raw_pointer_cast(this->grad_sorted.data());

      if (this->config->dynamic_parallelism)
        this->Partition(this->d_fvalue_partitioned, fvalue_tmp,
                        thrust::raw_pointer_cast(row2Node.data()),
                        parent_node_count, level - 1, depth);
      else
        this->Partition(this->d_fvalue_partitioned, fvalue_tmp,
                        thrust::raw_pointer_cast(row2Node.data()),
                        this->best->parent_node_count_h, level - 1, depth);

      OK(cudaEventRecord(this->event, this->stream));

      OK(cudaStreamWaitEvent(this->copy_d2h_stream, this->event, 0));
      if (fvalue_d == nullptr)
        OK(cudaMemcpyAsync(fvalue_h, this->d_fvalue_partitioned,
                           this->size * sizeof(unsigned int),
                           cudaMemcpyDeviceToHost, this->copy_d2h_stream));

      if (fvalue_d != nullptr) {
        OK(cudaMemcpyAsync(fvalue_d, this->d_fvalue_partitioned,
                           this->size * sizeof(unsigned int),
                           cudaMemcpyDeviceToDevice, this->copy_d2h_stream));
      }

    } else {
      this->d_fvalue_partitioned = fvalue_tmp;
    }

    if (partition_only) return;

    if (level != 0) {
      if (this->config->dynamic_parallelism)
        HistTreeGrower<NODE_T, GRAD_T, SUM_T>::HistSum(
          thrust::raw_pointer_cast(this->sum.data()),
          thrust::raw_pointer_cast(this->hist_bin_count.data()),
          thrust::raw_pointer_cast(
            this->features_histogram->grad_hist[fid].data()),
          thrust::raw_pointer_cast(
            this->features_histogram->count_hist[fid].data()),
          thrust::raw_pointer_cast(grad_d.data()),
          thrust::raw_pointer_cast(parent_node_count.data()),
          this->d_fvalue_partitioned, hist_size_bits, hist_size, lenght,
          this->features_histogram->CanUseTrick(fid, level), this->stream);
      else
        HistTreeGrower<NODE_T, GRAD_T, SUM_T>::HistSumStatic(
          thrust::raw_pointer_cast(this->sum.data()),
          thrust::raw_pointer_cast(this->hist_bin_count.data()),
          thrust::raw_pointer_cast(
            this->features_histogram->grad_hist[fid].data()),
          thrust::raw_pointer_cast(
            this->features_histogram->count_hist[fid].data()),
          thrust::raw_pointer_cast(grad_d.data()),
          thrust::raw_pointer_cast(this->best->parent_node_count_h.data()),
          this->d_fvalue_partitioned, hist_size_bits, hist_size, lenght,
          this->features_histogram->CanUseTrick(fid, level), this->stream);
    } else {
      HistTreeGrower<NODE_T, GRAD_T, SUM_T>::HistSumSingleNode(
        thrust::raw_pointer_cast(this->sum.data()),
        thrust::raw_pointer_cast(this->hist_bin_count.data()),
        thrust::raw_pointer_cast(grad_d.data()),
        thrust::raw_pointer_cast(parent_node_count.data()),
        this->d_fvalue_partitioned, hist_size_bits, this->size, this->stream);
    }
    cub::Sum sum_op;

    OK(cub::DeviceScan::InclusiveScan(
      this->temp_bytes, this->temp_bytes_allocated,
      thrust::raw_pointer_cast(this->sum.data()),
      thrust::raw_pointer_cast(this->hist_prefix_sum.data()), sum_op,
      lenght * this->hist_size, this->stream));

    OK(cub::DeviceScan::InclusiveSum(
      this->temp_bytes, this->temp_bytes_allocated,
      thrust::raw_pointer_cast(this->hist_bin_count.data()),
      thrust::raw_pointer_cast(this->hist_prefix_count.data()),
      lenght * this->hist_size, this->stream));
    int grid_size = 0;
    int block_size = 0;

    compute1DInvokeConfig(lenght * this->hist_size, &grid_size, &block_size,
                          hist_gain_kernel<SUM_T>, 0, 1024);

    hist_gain_kernel<SUM_T><<<grid_size, block_size, 0, this->stream>>>(
      thrust::raw_pointer_cast(this->hist_prefix_sum.data()),
      thrust::raw_pointer_cast(this->hist_prefix_count.data()),
      thrust::raw_pointer_cast(parent_node_sum.data()),
      thrust::raw_pointer_cast(parent_node_count.data()), this->hist_size,
      lenght * this->hist_size, gain_param,
      thrust::raw_pointer_cast(this->result_d.data()));
  }

  template <typename NODE_VALUE_T>
  inline void ProcessCategoryFeature(
    const device_vector<NODE_T> &row2Node, const device_vector<GRAD_T> &grad_d,
    const device_vector<unsigned int> &fvalue_d,
    const host_vector<unsigned int> &fvalue_h,
    const device_vector<SUM_T> &parent_node_sum,
    const device_vector<unsigned int> &parent_node_count,
    const unsigned char category_size, const size_t level,
    const GainFunctionParameters gain_param) {}

  void FindBest(BestSplit<SUM_T> &best, device_vector<NODE_T> &row2Node,
                const device_vector<SUM_T> &parent_node_sum,
                const device_vector<unsigned int> &parent_node_count,
                unsigned fid, const unsigned level, const unsigned depth,
                const unsigned size) {
    int gridSize = 0;
    int blockSize = 0;

    compute1DInvokeConfig(size, &gridSize, &blockSize,
                          hist_apply_candidates<NODE_T, SUM_T>);
    hist_apply_candidates<NODE_T, SUM_T>
      <<<gridSize, blockSize, 0, this->stream>>>(
        thrust::raw_pointer_cast(best.gain.data()),
        thrust::raw_pointer_cast(best.feature.data()),
        thrust::raw_pointer_cast(best.sum.data()),
        thrust::raw_pointer_cast(best.split_value.data()),
        thrust::raw_pointer_cast(best.count.data()),
        thrust::raw_pointer_cast(best.parent_node_count_next.data()),
        thrust::raw_pointer_cast(best.parent_node_sum_next.data()),
        thrust::raw_pointer_cast(this->result_d.data()),
        thrust::raw_pointer_cast(this->hist_prefix_sum.data()),
        thrust::raw_pointer_cast(this->hist_prefix_count.data()),
        this->d_fvalue_partitioned, thrust::raw_pointer_cast(row2Node.data()),
        thrust::raw_pointer_cast(parent_node_count.data()),
        thrust::raw_pointer_cast(parent_node_sum.data()), fid,
        depth - level - 2, this->hist_size, size);
    if (this->config->use_hist_subtraction_trick) {
      this->features_histogram->Update(this->sum, this->hist_bin_count, fid,
                                       level, this->stream);
    }
  }

  device_vector<SUM_T> hist_prefix_sum;
  device_vector<unsigned> hist_bin_count;
  device_vector<unsigned> hist_prefix_count;

  const unsigned hist_size;
  unsigned hist_size_bits;
};

template <typename NODE_T, typename GRAD_T, typename SUM_T>
class ContinuousTreeGrower : public BaseGrower<NODE_T, GRAD_T, SUM_T> {
 public:
  ContinuousTreeGrower(const size_t size, const unsigned depth,
                       const unsigned hist_size,
                       const BestSplit<SUM_T> *best = NULL,
                       Histogram<SUM_T> *features_histogram = NULL,
                       const InternalConfiguration *config = NULL)
      : BaseGrower<NODE_T, GRAD_T, SUM_T>(size, depth, best, features_histogram,
                                          config) {
    node_fvalue.resize(size);
    node_fvalue_sorted.resize(size);
    sum.resize(size);
    run_lenght.resize(1);

    size_t temp_storage_bytes = 0;

    OK(cub::DeviceSegmentedRadixSort::SortPairs(
      NULL, temp_storage_bytes, (NODE_T *)nullptr, (NODE_T *)nullptr,
      (GRAD_T *)nullptr, (GRAD_T *)nullptr, size, 1 << this->depth,
      (unsigned *)nullptr, (unsigned *)nullptr, 0, 1));

    this->temp_bytes_allocated =
      std::max(this->temp_bytes_allocated, temp_storage_bytes);

    temp_storage_bytes = 0;

    SUM_T initial_value;
    init(initial_value);
    cub::Sum sum_op;

    OK(cub::DeviceScan::ExclusiveScan(NULL, temp_storage_bytes,
                                      (GRAD_T *)nullptr, (SUM_T *)nullptr,
                                      sum_op, initial_value, size));

    this->temp_bytes_allocated =
      std::max(this->temp_bytes_allocated, temp_storage_bytes);

    temp_storage_bytes = 0;

    OK(cub::DeviceReduce::ReduceByKey(
      NULL, temp_storage_bytes, (NODE_T *)nullptr, (NODE_T *)nullptr,
      (GRAD_T *)nullptr, (SUM_T *)nullptr,
      thrust::raw_pointer_cast(run_lenght.data()), sum_op, size));

    this->temp_bytes_allocated =
      std::max(this->temp_bytes_allocated, temp_storage_bytes);

    temp_storage_bytes = 0;

    OK(cub::DeviceRunLengthEncode::Encode(
      NULL, temp_storage_bytes, (NODE_T *)nullptr, (NODE_T *)nullptr,
      (NODE_T *)nullptr, thrust::raw_pointer_cast(run_lenght.data()), size));

    this->temp_bytes_allocated =
      std::max(this->temp_bytes_allocated, temp_storage_bytes);

    OK(cudaMalloc(&this->temp_bytes, this->temp_bytes_allocated));
  }

  device_vector<NODE_T> node_fvalue;
  device_vector<NODE_T> node_fvalue_sorted;
  device_vector<SUM_T> sum;
  device_vector<unsigned int> run_lenght;
  NODE_T *best_split_h;

  inline void ApplySplit(NODE_T *row2Node, const unsigned level,
                         const unsigned threshold, size_t from, size_t to) {
    int gridSize;
    int blockSize;
    compute1DInvokeConfig(to - from, &gridSize, &blockSize,
                          apply_split<NODE_T>);

    apply_split<NODE_T><<<gridSize, blockSize, 0, this->stream>>>(
      row2Node + from,
      ((unsigned *)thrust::raw_pointer_cast(node_fvalue.data())) + from,
      threshold, level, to - from);
  }

  // FIXME: Use template parameter instead of unsigned
  template <typename NODE_VALUE_T>
  inline void ProcessDenseFeature(
    const device_vector<NODE_T> &row2Node, const device_vector<GRAD_T> &grad_d,
    unsigned int *fvalue_d, unsigned int *fvalue_h,
    const device_vector<SUM_T> &parent_node_sum,
    const device_vector<unsigned int> &parent_node_count,
    const unsigned char fvalue_size, const unsigned level, const unsigned depth,
    const GainFunctionParameters gain_param, const bool partition_only,
    const int fid = -1) {
    const unsigned lenght = 1 << level;

    OK(cudaMemsetAsync(thrust::raw_pointer_cast(this->result_d.data()), 0,
                       lenght * sizeof(my_atomics), this->stream));

    unsigned int *fvalue_tmp = NULL;

    if (fvalue_d != nullptr) {
      fvalue_tmp = fvalue_d;
    } else {
      OK(cudaMemcpyAsync(thrust::raw_pointer_cast(this->fvalue.data()),
                         fvalue_h, this->size * sizeof(unsigned int),
                         cudaMemcpyHostToDevice, this->stream));
      fvalue_tmp = thrust::raw_pointer_cast(this->fvalue.data());
    }

    if (level != 0) {
      const unsigned lenght = 1 << (level - 1);
      int gridSize = 0;
      int blockSize = 0;

      compute1DInvokeConfig(lenght, &gridSize, &blockSize,
                            partition<NODE_T, unsigned>, 0, 1);
      partition<NODE_T, unsigned, 2><<<gridSize, blockSize, 0, this->stream>>>(
        (unsigned *)thrust::raw_pointer_cast(node_fvalue.data()),
        thrust::raw_pointer_cast(row2Node.data()), fvalue_tmp,
        thrust::raw_pointer_cast(parent_node_count.data()), depth - level - 1,
        this->temp_bytes_allocated, this->temp_bytes, this->size, lenght);

      OK(cudaEventRecord(this->event, this->stream));

      OK(cudaStreamWaitEvent(this->copy_d2h_stream, this->event, 0));

      OK(cudaMemcpyAsync(fvalue_h, thrust::raw_pointer_cast(node_fvalue.data()),
                         this->size * sizeof(unsigned int),
                         cudaMemcpyDeviceToHost, this->copy_d2h_stream));

      if (fvalue_d != nullptr) {
        OK(cudaMemcpyAsync(fvalue_d,
                           thrust::raw_pointer_cast(node_fvalue.data()),
                           this->size * sizeof(unsigned int),
                           cudaMemcpyDeviceToDevice, this->copy_d2h_stream));
      }

      this->d_fvalue_partitioned =
        (unsigned *)thrust::raw_pointer_cast(node_fvalue.data());

    } else {
      this->d_fvalue_partitioned = fvalue_tmp;
    }

    if (partition_only) return;

    // FIXME: fvalue_size + 1 or just fvalue_size?
    CubDebugExit(cub::DeviceSegmentedRadixSort::SortPairs(
      this->temp_bytes, this->temp_bytes_allocated, this->d_fvalue_partitioned,
      (unsigned *)thrust::raw_pointer_cast(node_fvalue_sorted.data()),
      thrust::raw_pointer_cast(grad_d.data()),
      thrust::raw_pointer_cast(this->grad_sorted.data()), this->size,
      1 << level, thrust::raw_pointer_cast(parent_node_count.data()),
      thrust::raw_pointer_cast(parent_node_count.data()) + 1, 0,
      fvalue_size + 1, this->stream));

    SUM_T initial_value;
    init(initial_value);
    cub::Sum sum_op;

    OK(cub::DeviceScan::ExclusiveScan(
      this->temp_bytes, this->temp_bytes_allocated,
      thrust::raw_pointer_cast(this->grad_sorted.data()),
      thrust::raw_pointer_cast(sum.data()), sum_op, initial_value, this->size,
      this->stream));

    gain_kernel<<<this->gridSizeGain, this->blockSizeGain, 0, this->stream>>>(
      thrust::raw_pointer_cast(sum.data()),
      (unsigned *)thrust::raw_pointer_cast(node_fvalue_sorted.data()), lenght,
      thrust::raw_pointer_cast(parent_node_sum.data()),
      thrust::raw_pointer_cast(parent_node_count.data()), this->size,
      gain_param, thrust::raw_pointer_cast(this->result_d.data()));
  }

  template <typename NODE_VALUE_T>
  inline void ProcessCategoryFeature(
    const device_vector<NODE_T> &row2Node, const device_vector<GRAD_T> &grad_d,
    const device_vector<unsigned int> &fvalue_d,
    const host_vector<unsigned int> &fvalue_h,
    const device_vector<SUM_T> &parent_node_sum,
    const device_vector<unsigned int> &parent_node_count,
    const unsigned char category_size, const size_t level,
    const GainFunctionParameters gain_param) {
    size_t lenght = 1 << level;
    OK(cudaMemsetAsync(thrust::raw_pointer_cast(this->result_d.data()), 0,
                       lenght * sizeof(my_atomics), this->stream));

    device_vector<unsigned int> *fvalue_tmp = NULL;

    if (fvalue_d.size() > 0) {
      fvalue_tmp = const_cast<device_vector<unsigned int> *>(&(fvalue_d));
    } else {
      OK(cudaMemcpyAsync(thrust::raw_pointer_cast((this->fvalue.data())),
                         thrust::raw_pointer_cast(fvalue_h.data()),
                         this->size * sizeof(unsigned int),
                         cudaMemcpyHostToDevice, this->stream));
      fvalue_tmp = const_cast<device_vector<unsigned int> *>(&(this->fvalue));
    }

    assign_kernel<<<this->gridSizeGather, this->blockSizeGather, 0,
                    this->stream>>>(
      thrust::raw_pointer_cast(fvalue_tmp->data()),
      thrust::raw_pointer_cast(row2Node.data()), category_size,
      (NODE_VALUE_T *)thrust::raw_pointer_cast(node_fvalue.data()), this->size);

    OK(cub::DeviceRadixSort::SortPairs(
      this->temp_bytes, this->temp_bytes_allocated,
      (NODE_VALUE_T *)thrust::raw_pointer_cast(node_fvalue.data()),
      (NODE_VALUE_T *)thrust::raw_pointer_cast(node_fvalue_sorted.data()),
      thrust::raw_pointer_cast((grad_d.data())),
      thrust::raw_pointer_cast(this->grad_sorted.data()), this->size, 0,
      category_size + level + 1, this->stream));

    const NODE_VALUE_T mask = (1 << (category_size)) - 1;

    SUM_T initial_value;
    init(initial_value);
    cub::Sum sum_op;

    OK(cub::DeviceReduce::ReduceByKey(
      this->temp_bytes, this->temp_bytes_allocated,
      (NODE_VALUE_T *)thrust::raw_pointer_cast(node_fvalue_sorted.data()),
      (NODE_VALUE_T *)thrust::raw_pointer_cast(node_fvalue.data()),
      thrust::raw_pointer_cast(this->grad_sorted.data()),
      thrust::raw_pointer_cast(sum.data()),
      thrust::raw_pointer_cast(run_lenght.data()), sum_op, this->size,
      this->stream));

    OK(cub::DeviceRunLengthEncode::Encode(
      this->temp_bytes, this->temp_bytes_allocated,
      (NODE_VALUE_T *)thrust::raw_pointer_cast(node_fvalue_sorted.data()),
      (NODE_VALUE_T *)thrust::raw_pointer_cast(node_fvalue.data()),
      thrust::raw_pointer_cast(this->fvalue.data()),
      thrust::raw_pointer_cast(run_lenght.data()), this->size, this->stream));

    gain_kernel_category<<<this->gridSizeGain, this->blockSizeGain, 0,
                           this->stream>>>(
      thrust::raw_pointer_cast(sum.data()),
      thrust::raw_pointer_cast(this->fvalue.data()),
      (NODE_VALUE_T *)thrust::raw_pointer_cast(node_fvalue.data()),
      category_size, mask, thrust::raw_pointer_cast(parent_node_sum.data()),
      thrust::raw_pointer_cast(parent_node_count.data()),
      thrust::raw_pointer_cast(run_lenght.data()), gain_param,
      thrust::raw_pointer_cast(this->result_d.data()));
  }

  void FindBest(BestSplit<SUM_T> &best, device_vector<NODE_T> &row2Node,
                const device_vector<SUM_T> &parent_node_sum,
                const device_vector<unsigned int> &parent_node_count,
                const unsigned fid, const unsigned level, const unsigned depth,
                const unsigned size) {
    int gridSize = 0;
    int blockSize = 0;

    compute1DInvokeConfig(size, &gridSize, &blockSize,
                          filter_apply_candidates<NODE_T, SUM_T>);

    filter_apply_candidates<NODE_T, SUM_T>
      <<<gridSize, blockSize, 0, this->stream>>>(
        thrust::raw_pointer_cast(best.gain.data()),
        thrust::raw_pointer_cast(best.feature.data()),
        thrust::raw_pointer_cast(best.sum.data()),
        thrust::raw_pointer_cast(best.split_value.data()),
        thrust::raw_pointer_cast(best.count.data()),
        thrust::raw_pointer_cast(best.parent_node_count_next.data()),
        thrust::raw_pointer_cast(best.parent_node_sum_next.data()),
        thrust::raw_pointer_cast(this->result_d.data()),
        thrust::raw_pointer_cast(this->sum.data()), this->d_fvalue_partitioned,
        (unsigned *)thrust::raw_pointer_cast(this->node_fvalue_sorted.data()),
        thrust::raw_pointer_cast(row2Node.data()),
        thrust::raw_pointer_cast(parent_node_count.data()),
        thrust::raw_pointer_cast(parent_node_sum.data()), fid,
        depth - level - 2, size);
  }
};
}  // namespace core
}  // namespace arboretum

#endif