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
#include "partition.cuh"

namespace arboretum {
namespace core {
using thrust::device_vector;
using thrust::host_vector;

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

template <class T1>
__global__ void scatter_kernel(const unsigned int *const __restrict__ position,
                               const T1 *const __restrict__ in, T1 *out,
                               const size_t n) {
  for (size_t i = blockDim.x * blockIdx.x + threadIdx.x; i < n;
       i += gridDim.x * blockDim.x) {
    out[position[i]] = in[i];
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
__global__ void apply_split(NODE_T *row2Node, const unsigned short *fvalues,
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
                          scatter_kernel<NODE_T>);

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

    temp_storage_bytes = 0;

    const SegmentedInputIterator<const NODE_T *> input_iter((NODE_T *)nullptr,
                                                            0);
    PartitioningIterator<GRAD_T> out_iter(
      (unsigned *)nullptr, (GRAD_T *)nullptr, (GRAD_T *)nullptr, 0);

    OK(cub::DeviceScan::InclusiveScan(NULL, temp_storage_bytes, input_iter,
                                      out_iter, Position(0), this->size));

    temp_bytes_allocated = std::max(temp_bytes_allocated, temp_storage_bytes);

    grad_sorted.resize(size);
    fvalue.resize(size);
    fvalue_dst.resize(size);
    result_d.resize(1 << this->depth);
  }

  ~BaseGrower() {
    OK(cudaFree(temp_bytes));
    OK(cudaStreamDestroy(stream));
    OK(cudaStreamDestroy(copy_d2h_stream));
    OK(cudaEventDestroy(event));
  }

  template <typename T>
  void Partition(T *src, const device_vector<unsigned> &index) {
    this->PartitionByIndex((T *)thrust::raw_pointer_cast(grad_sorted.data()),
                           src, index);

    OK(cudaMemcpyAsync(src, (T *)thrust::raw_pointer_cast(grad_sorted.data()),
                       size * sizeof(T), cudaMemcpyDeviceToDevice, stream));
  }

  template <typename T, int OFFSET = 2>
  void Partition2(T *dst, T *src, const NODE_T *row2Node,
                  const device_vector<unsigned> &parent_node_count,
                  const unsigned level, const unsigned depth) {
    const SegmentedInputIterator<const NODE_T *> segmented(row2Node,
                                                           depth - level - 2);
    PartitioningIterator<T> out_iter(
      thrust::raw_pointer_cast(parent_node_count.data()), src, dst,
      depth - level - 2);

    OK(cub::DeviceScan::InclusiveScan(
      this->temp_bytes, this->temp_bytes_allocated, segmented, out_iter,
      Position(depth - level - 2), this->size, this->stream));
  }

  template <typename T>
  void PartitionByIndex(T *dst, const T *src,
                        const device_vector<unsigned> &index) {
    scatter_kernel<T><<<gridSizeGather, blockSizeGather, 0, stream>>>(
      thrust::raw_pointer_cast(index.data()), src, dst, index.size());
  }

  void CreatePartitioningIndexes(
    device_vector<unsigned> &indexes, const device_vector<NODE_T> &row2Node,
    const device_vector<unsigned> &parent_node_count, const unsigned level,
    const unsigned depth) {
    const SegmentedInputIterator<const NODE_T *> segmented(
      thrust::raw_pointer_cast(row2Node.data()), depth - level - 1);
    PartitioningIndexIterator out_iter(
      thrust::raw_pointer_cast(parent_node_count.data()),
      thrust::raw_pointer_cast(indexes.data()), depth - level - 1);

    OK(cub::DeviceScan::InclusiveScan(
      this->temp_bytes, this->temp_bytes_allocated, segmented, out_iter,
      Position(depth - level - 1), this->size, this->stream));
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
        PartitioningLeafs<NODE_T> conversion_op(depth - level - OFFSET);

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
  device_vector<unsigned short> fvalue;
  device_vector<unsigned short> fvalue_dst;
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
  unsigned short *d_fvalue_partitioned;
  const BestSplit<SUM_T> *best;
  Histogram<SUM_T> *features_histogram;
  const InternalConfiguration *config;
};

}  // namespace core
}  // namespace arboretum

#endif