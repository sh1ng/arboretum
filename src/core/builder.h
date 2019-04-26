#ifndef SRC_CORE_BUILDER_H
#define SRC_CORE_BUILDER_H
#define CUB_STDERR
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/system/cuda/experimental/pinned_allocator.h>
#include <cub/cub.cuh>
#include "cuda_helpers.h"
#include "cuda_runtime.h"

namespace arboretum {
namespace core {
using thrust::device_vector;
using thrust::host_vector;
using thrust::cuda::experimental::pinned_allocator;

union my_atomics {
  float floats[2];               // floats[0] = maxvalue
  unsigned int ints[2];          // ints[1] = maxindex
  unsigned long long int ulong;  // for atomic update
};

struct GainFunctionParameters {
  const unsigned int min_leaf_size;
  const float hess;
  const float gamma_absolute;
  const float gamma_relative;
  const float lambda;
  const float alpha;
  GainFunctionParameters(const unsigned int min_leaf_size, const float hess,
                         const float gamma_absolute, const float gamma_relative,
                         const float lambda, const float alpha)
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
                              NODE_VALUE_T *out, const size_t n) {
  for (unsigned int i = blockDim.x * blockIdx.x + threadIdx.x; i < n;
       i += gridDim.x * blockDim.x) {
    const NODE_VALUE_T node = segments[i];
    out[i] = (node << fvalue_size) | (NODE_VALUE_T)fvalue[i];
  }
}

template <typename SUM_T, typename NODE_VALUE_T>
__global__ void gain_kernel(
    const SUM_T *const __restrict__ left_sum,
    const NODE_VALUE_T *const __restrict__ segments_fvalues,
    const unsigned char fvalue_size, const NODE_VALUE_T mask,
    const SUM_T *const __restrict__ parent_sum_iter,
    const unsigned int *const __restrict__ parent_count_iter, const size_t n,
    const GainFunctionParameters parameters, my_atomics *res) {
  for (unsigned int i = blockDim.x * blockIdx.x + threadIdx.x; i < n;
       i += gridDim.x * blockDim.x) {
    if (i == 0) continue;

    const unsigned int fvalue_segment = segments_fvalues[i];
    const unsigned int fvalue_segment_prev = segments_fvalues[i - 1];

    const unsigned int fvalue = fvalue_segment & mask;
    const unsigned int fvalue_prev = fvalue_segment_prev & mask;
    if (fvalue != fvalue_prev) {
      const unsigned int segment = fvalue_segment_prev >> fvalue_size;

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

template <typename NODE_T, typename GRAD_T, typename SUM_T>
class ContinuousTreeGrower {
 public:
  ContinuousTreeGrower(const size_t size, const unsigned tree_span)
      : size(size),
        tree_span(tree_span),
        gridSizeGain(0),
        blockSizeGain(0),
        gridSizeGather(0),
        blockSizeGather(0) {
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

    compute1DInvokeConfig(size, &gridSizeGain, &blockSizeGain,
                          gain_kernel<SUM_T, NODE_T>);

    compute1DInvokeConfig(size, &gridSizeGather, &blockSizeGather,
                          gather_kernel<NODE_T, float>);

    fvalue.resize(size);
    node_fvalue.resize(size);
    node_fvalue_sorted.resize(size);
    grad_sorted.resize(size);
    sum.resize(size);
    run_lenght.resize(1);
    result_h.resize(tree_span);
    result_d.resize(tree_span);

    temp_bytes_allocated = 0;

    size_t temp_storage_bytes = 0;

    OK(cub::DeviceRadixSort::SortPairs(
        NULL, temp_storage_bytes, (NODE_T *)nullptr, (NODE_T *)nullptr,
        (GRAD_T *)nullptr, (GRAD_T *)nullptr, size, 0, 1));

    temp_bytes_allocated = std::max(temp_bytes_allocated, temp_storage_bytes);

    temp_storage_bytes = 0;

    SUM_T initial_value;
    init(initial_value);
    cub::Sum sum_op;

    OK(cub::DeviceScan::ExclusiveScan(NULL, temp_storage_bytes,
                                      (GRAD_T *)nullptr, (SUM_T *)nullptr,
                                      sum_op, initial_value, size));

    temp_bytes_allocated = std::max(temp_bytes_allocated, temp_storage_bytes);

    temp_storage_bytes = 0;

    OK(cub::DeviceReduce::ReduceByKey(
        NULL, temp_storage_bytes, (NODE_T *)nullptr, (NODE_T *)nullptr,
        (GRAD_T *)nullptr, (SUM_T *)nullptr,
        thrust::raw_pointer_cast(run_lenght.data()), sum_op, size));

    temp_bytes_allocated = std::max(temp_bytes_allocated, temp_storage_bytes);

    temp_storage_bytes = 0;

    OK(cub::DeviceRunLengthEncode::Encode(
        NULL, temp_storage_bytes, (NODE_T *)nullptr, (NODE_T *)nullptr,
        (NODE_T *)nullptr, thrust::raw_pointer_cast(run_lenght.data()), size));

    temp_bytes_allocated = std::max(temp_bytes_allocated, temp_storage_bytes);

    OK(cudaMalloc(&temp_bytes, temp_bytes_allocated));
  }

  ~ContinuousTreeGrower() {
    OK(cudaFree(temp_bytes));
    OK(cudaStreamDestroy(stream));
  }
  cudaStream_t stream;
  device_vector<unsigned int> fvalue;
  device_vector<NODE_T> node_fvalue;
  device_vector<NODE_T> node_fvalue_sorted;
  device_vector<GRAD_T> grad_sorted;
  device_vector<SUM_T> sum;
  device_vector<my_atomics> result_d;
  host_vector<my_atomics> result_h;
  device_vector<unsigned int> run_lenght;
  size_t temp_bytes_allocated;
  void *temp_bytes;
  NODE_T *best_split_h;
  const size_t size;
  const unsigned tree_span;

  int blockSizeGain;
  int gridSizeGain;

  int blockSizeGather;
  int gridSizeGather;

  template <typename NODE_VALUE_T>
  inline void ProcessDenseFeature(
      const device_vector<NODE_T> &row2Node, const GRAD_T *grad_d,
      const device_vector<unsigned int> &fvalue_d,
      const host_vector<unsigned int> &fvalue_h,
      const device_vector<SUM_T> &parent_node_sum,
      const device_vector<unsigned int> &parent_node_count,
      const unsigned char fvalue_size, const size_t level,
      const GainFunctionParameters gain_param) {
    size_t lenght = 1 << level;
    OK(cudaMemsetAsync(thrust::raw_pointer_cast(result_d.data()), 0,
                       lenght * sizeof(my_atomics), stream));

    device_vector<unsigned int> *fvalue_tmp = NULL;

    if (fvalue_d.size() > 0) {
      fvalue_tmp = const_cast<device_vector<unsigned int> *>(&(fvalue_d));
    } else {
      OK(cudaMemcpyAsync(thrust::raw_pointer_cast(fvalue.data()),
                         thrust::raw_pointer_cast(fvalue_h.data()),
                         size * sizeof(unsigned int), cudaMemcpyHostToDevice,
                         stream));
      OK(cudaStreamSynchronize(stream));
      fvalue_tmp = const_cast<device_vector<unsigned int> *>(&(fvalue));
    }

    assign_kernel<<<gridSizeGather, blockSizeGather, 0, stream>>>(
        thrust::raw_pointer_cast(fvalue_tmp->data()),
        thrust::raw_pointer_cast(row2Node.data()), fvalue_size,
        (NODE_VALUE_T *)thrust::raw_pointer_cast(node_fvalue.data()), size);

    OK(cub::DeviceRadixSort::SortPairs(
        temp_bytes, temp_bytes_allocated,
        (NODE_VALUE_T *)thrust::raw_pointer_cast(node_fvalue.data()),
        (NODE_VALUE_T *)thrust::raw_pointer_cast(node_fvalue_sorted.data()),
        grad_d, thrust::raw_pointer_cast(grad_sorted.data()), size, 0,
        fvalue_size + level + 1, stream));

    SUM_T initial_value;
    init(initial_value);
    cub::Sum sum_op;

    OK(cub::DeviceScan::ExclusiveScan(
        temp_bytes, temp_bytes_allocated,
        thrust::raw_pointer_cast(grad_sorted.data()),
        thrust::raw_pointer_cast(sum.data()), sum_op, initial_value, size,
        stream));

    const NODE_VALUE_T mask = (1 << (fvalue_size)) - 1;

    gain_kernel<<<gridSizeGain, blockSizeGain, 0, stream>>>(
        thrust::raw_pointer_cast(sum.data()),
        (NODE_VALUE_T *)thrust::raw_pointer_cast(node_fvalue_sorted.data()),
        fvalue_size, mask, thrust::raw_pointer_cast(parent_node_sum.data()),
        thrust::raw_pointer_cast(parent_node_count.data()), size, gain_param,
        thrust::raw_pointer_cast(result_d.data()));

    OK(cudaMemcpyAsync(thrust::raw_pointer_cast((result_h.data())),
                       thrust::raw_pointer_cast(result_d.data()),
                       lenght * sizeof(my_atomics), cudaMemcpyDeviceToHost,
                       stream));
  }

  template <typename NODE_VALUE_T>
  inline void ProcessCategoryFeature(
      const device_vector<NODE_T> &row2Node, const GRAD_T *grad_d,
      const device_vector<unsigned int> &fvalue_d,
      const host_vector<unsigned int> &fvalue_h,
      const device_vector<SUM_T> &parent_node_sum,
      const device_vector<unsigned int> &parent_node_count,
      const unsigned char category_size, const size_t level,
      const GainFunctionParameters gain_param) {
    size_t lenght = 1 << level;
    OK(cudaMemsetAsync(thrust::raw_pointer_cast(result_d.data()), 0,
                       lenght * sizeof(my_atomics), stream));

    device_vector<unsigned int> *fvalue_tmp = NULL;

    if (fvalue_d.size() > 0) {
      fvalue_tmp = const_cast<device_vector<unsigned int> *>(&(fvalue_d));
    } else {
      OK(cudaMemcpyAsync(thrust::raw_pointer_cast((fvalue.data())),
                         thrust::raw_pointer_cast(fvalue_h.data()),
                         size * sizeof(unsigned int), cudaMemcpyHostToDevice,
                         stream));
      OK(cudaStreamSynchronize(stream));
      fvalue_tmp = const_cast<device_vector<unsigned int> *>(&(fvalue));
    }

    assign_kernel<<<gridSizeGather, blockSizeGather, 0, stream>>>(
        thrust::raw_pointer_cast(fvalue_tmp->data()),
        thrust::raw_pointer_cast(row2Node.data()), category_size,
        (NODE_VALUE_T *)thrust::raw_pointer_cast(node_fvalue.data()), size);

    OK(cub::DeviceRadixSort::SortPairs(
        temp_bytes, temp_bytes_allocated,
        (NODE_VALUE_T *)thrust::raw_pointer_cast(node_fvalue.data()),
        (NODE_VALUE_T *)thrust::raw_pointer_cast(node_fvalue_sorted.data()),
        grad_d, thrust::raw_pointer_cast(grad_sorted.data()), size, 0,
        category_size + level + 1, stream));

    const NODE_VALUE_T mask = (1 << (category_size)) - 1;

    SUM_T initial_value;
    init(initial_value);
    cub::Sum sum_op;

    OK(cub::DeviceReduce::ReduceByKey(
        temp_bytes, temp_bytes_allocated,
        (NODE_VALUE_T *)thrust::raw_pointer_cast(node_fvalue_sorted.data()),
        (NODE_VALUE_T *)thrust::raw_pointer_cast(node_fvalue.data()),
        thrust::raw_pointer_cast(grad_sorted.data()),
        thrust::raw_pointer_cast(sum.data()),
        thrust::raw_pointer_cast(run_lenght.data()), sum_op, size, stream));

    OK(cub::DeviceRunLengthEncode::Encode(
        temp_bytes, temp_bytes_allocated,
        (NODE_VALUE_T *)thrust::raw_pointer_cast(node_fvalue_sorted.data()),
        (NODE_VALUE_T *)thrust::raw_pointer_cast(node_fvalue.data()),
        thrust::raw_pointer_cast(fvalue.data()),
        thrust::raw_pointer_cast(run_lenght.data()), size, stream));

    gain_kernel_category<<<gridSizeGain, blockSizeGain, 0, stream>>>(
        thrust::raw_pointer_cast(sum.data()),
        thrust::raw_pointer_cast(fvalue.data()),
        (NODE_VALUE_T *)thrust::raw_pointer_cast(node_fvalue.data()),
        category_size, mask, thrust::raw_pointer_cast(parent_node_sum.data()),
        thrust::raw_pointer_cast(parent_node_count.data()),
        thrust::raw_pointer_cast(run_lenght.data()), gain_param,
        thrust::raw_pointer_cast(result_d.data()));

    OK(cudaMemcpyAsync(thrust::raw_pointer_cast((result_h.data())),
                       thrust::raw_pointer_cast(result_d.data()),
                       lenght * sizeof(my_atomics), cudaMemcpyDeviceToHost,
                       stream));
  }
};
}  // namespace core
}  // namespace arboretum

#endif