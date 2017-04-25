#define CUB_STDERR

#include "cuda_helpers.h"
#include "cuda_runtime.h"
#include "garden.h"
#include "objective.h"
#include "param.h"
#include <algorithm>
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/util_allocator.cuh>
#include <limits>
#include <math.h>
#include <random>
#include <stdio.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/system/cuda/experimental/pinned_allocator.h>

namespace arboretum {
namespace core {
using namespace thrust;
using namespace thrust::cuda;
using thrust::host_vector;
using thrust::device_vector;
using thrust::cuda::experimental::pinned_allocator;

union my_atomics {
  float floats[2];              // floats[0] = maxvalue
  unsigned int ints[2];         // ints[1] = maxindex
  unsigned long long int ulong; // for atomic update
};

struct GainFunctionParameters {
  const unsigned int min_leaf_size;
  const float hess;
  const float gamma;
  const float lambda;
  const float alpha;
  GainFunctionParameters(const unsigned int min_leaf_size, const float hess,
                         const float gamma, const float lambda,
                         const float alpha)
      : min_leaf_size(min_leaf_size), hess(hess), gamma(gamma), lambda(lambda),
        alpha(alpha) {}
};

__forceinline__ __device__ unsigned long long int
updateAtomicMax(unsigned long long int *address, float val1,
                unsigned int val2) {
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

__forceinline__ __device__ __host__ float
gain_func(const double2 left_sum, const double2 total_sum,
          const size_t left_count, const size_t total_count,
          const GainFunctionParameters &params) {
  const double2 right_sum = total_sum - left_sum;
  if (left_count >= params.min_leaf_size &&
      (total_count - left_count) >= params.min_leaf_size &&
      std::abs(left_sum.y) >= params.hess &&
      std::abs(right_sum.y) >= params.hess) {
    const float l = (left_sum.x * left_sum.x) / (left_sum.y + params.lambda);
    const float r = (right_sum.x * right_sum.x) / (right_sum.y + params.lambda);
    const float p = (total_sum.x * total_sum.x) / (total_sum.y + params.lambda);
    return l + r - p;
  } else {
    return 0.0;
  }
}

__forceinline__ __device__ __host__ float
gain_func(const float2 left_sum, const float2 total_sum,
          const size_t left_count, const size_t total_count,
          const GainFunctionParameters &params) {
  const float2 right_sum = total_sum - left_sum;
  if (left_count >= params.min_leaf_size &&
      (total_count - left_count) >= params.min_leaf_size &&
      std::abs(left_sum.y) >= params.hess &&
      std::abs(right_sum.y) >= params.hess) {
    const float l = (left_sum.x * left_sum.x) / (left_sum.y + params.lambda);
    const float r = (right_sum.x * right_sum.x) / (right_sum.y + params.lambda);
    const float p = (total_sum.x * total_sum.x) / (total_sum.y + params.lambda);
    return l + r - p;
  } else {
    return 0.0;
  }
}

__forceinline__ __device__ __host__ float
gain_func(const float left_sum, const float total_sum, const size_t left_count,
          const size_t total_count, const GainFunctionParameters &params) {
  const size_t right_count = total_count - left_count;
  if (left_count >= params.min_leaf_size &&
      right_count >= params.min_leaf_size) {
    const float right_sum = total_sum - left_sum;
    const float l = left_sum * left_sum / (left_count + params.lambda);
    const float r = right_sum * right_sum / (right_count + params.lambda);
    const float p = total_sum * total_sum / (total_count + params.lambda);
    return l + r - p;
  } else {
    return 0.0;
  }
}

__forceinline__ __device__ __host__ float
gain_func(const double left_sum, const double total_sum,
          const size_t left_count, const size_t total_count,
          const GainFunctionParameters &params) {
  const size_t right_count = total_count - left_count;
  if (left_count >= params.min_leaf_size &&
      right_count >= params.min_leaf_size) {
    const double right_sum = total_sum - left_sum;
    const double l = left_sum * left_sum / (left_count + params.lambda);
    const double r = right_sum * right_sum / (right_count + params.lambda);
    const double p = total_sum * total_sum / (total_count + params.lambda);
    return l + r - p;
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
__global__ void
gain_kernel(const SUM_T *const __restrict__ left_sum,
            const NODE_VALUE_T *const __restrict__ segments_fvalues,
            const unsigned char fvalue_size, const NODE_VALUE_T mask,
            const SUM_T *const __restrict__ parent_sum_iter,
            const unsigned int *const __restrict__ parent_count_iter,
            const size_t n, const GainFunctionParameters parameters,
            my_atomics *res) {
  for (unsigned int i = blockDim.x * blockIdx.x + threadIdx.x; i < n;
       i += gridDim.x * blockDim.x) {
    if (i == 0)
      continue;

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
__global__ void
gain_kernel_category(const SUM_T *const __restrict__ category_sum,
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
class TaylorApproximationBuilder : public GardenBuilderBase {
public:
  TaylorApproximationBuilder(const TreeParam &param, const io::DataMatrix *data,
                             const InternalConfiguration &config,
                             const ApproximatedObjective<GRAD_T> *objective,
                             const bool verbose)
      : verbose(verbose), rnd(config.seed), overlap_depth(config.overlap),
        param(param), gain_param(param.min_leaf_size, param.min_child_weight,
                                 param.gamma, param.lambda, param.alpha),
        objective(objective) {

    grad_d.resize(data->rows);

    active_fids.resize(data->columns);

    const int lenght = 1 << param.depth;

    int minGridSize;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSizeGain,
                                       gain_kernel<SUM_T, NODE_T>, 0, 0);
    gridSizeGain = (data->rows + blockSizeGain - 1) / blockSizeGain;

    minGridSize = 0;

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSizeGather,
                                       gather_kernel<NODE_T, float>, 0, 0);
    gridSizeGather = (data->rows + blockSizeGather - 1) / blockSizeGather;

    row2Node.resize(data->rows);
    _rowIndex2Node.resize(data->rows, 0);
    _bestSplit.resize(1 << (param.depth - 2));
    _nodeStat.resize(1 << (param.depth - 2));

    parent_node_sum.resize(lenght + 1);
    parent_node_count.resize(lenght + 1);
    parent_node_sum_h.resize(lenght + 1);
    parent_node_count_h.resize(lenght + 1);

    for (size_t i = 0; i < overlap_depth; ++i) {
      cudaStream_t s;
      cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
      streams[i] = s;
      sum[i] = device_vector<SUM_T>(data->rows);
      fvalue[i] = device_vector<unsigned int>(data->rows);
      node_fvalue[i] = device_vector<NODE_T>(data->rows);
      node_fvalue_sorted[i] = device_vector<NODE_T>(data->rows);
      grad_sorted_sorted[i] = device_vector<GRAD_T>(data->rows);
      temp_bytes_allocated[i] = 0;
      CubDebugExit(cudaMalloc(&(results[i]), sizeof(my_atomics) * lenght));
      CubDebugExit(
          cudaMallocHost(&(results_h[i]), sizeof(my_atomics) * lenght));
      CubDebugExit(cudaMallocHost(&(best_split_h[i]), sizeof(NODE_T) * 2));
      CubDebugExit(cudaMalloc(&(run_lenght[i]), sizeof(unsigned int)));
    }

    {
      size_t max = 0;

      size_t temp_storage_bytes = 0;

      CubDebugExit(cub::DeviceRadixSort::SortPairs(
          NULL, temp_storage_bytes,
          thrust::raw_pointer_cast(node_fvalue[0].data()),
          thrust::raw_pointer_cast(node_fvalue_sorted[0].data()),
          thrust::raw_pointer_cast(grad_d.data()),
          thrust::raw_pointer_cast(grad_sorted_sorted[0].data()), data->rows, 0,
          1));

      max = std::max(max, temp_storage_bytes);

      temp_storage_bytes = 0;

      SUM_T initial_value;
      init(initial_value);
      cub::Sum sum_op;

      CubDebugExit(cub::DeviceScan::ExclusiveScan(
          NULL, temp_storage_bytes,
          thrust::raw_pointer_cast(grad_sorted_sorted[0].data()),
          thrust::raw_pointer_cast(sum[0].data()), sum_op, initial_value,
          data->rows));

      max = std::max(max, temp_storage_bytes);

      temp_storage_bytes = 0;

      CubDebugExit(cub::DeviceReduce::ReduceByKey(
          NULL, temp_storage_bytes,
          thrust::raw_pointer_cast(node_fvalue_sorted[0].data()),
          thrust::raw_pointer_cast(node_fvalue[0].data()),
          thrust::raw_pointer_cast(grad_sorted_sorted[0].data()),
          thrust::raw_pointer_cast(sum[0].data()),
          thrust::raw_pointer_cast(fvalue[0].data()) + data->rows - 1, sum_op,
          data->rows));

      max = std::max(max, temp_storage_bytes);

      temp_storage_bytes = 0;

      CubDebugExit(cub::DeviceRunLengthEncode::Encode(
          NULL, temp_storage_bytes,
          thrust::raw_pointer_cast(node_fvalue_sorted[0].data()),
          thrust::raw_pointer_cast(node_fvalue[0].data()),
          thrust::raw_pointer_cast(fvalue[0].data()),
          thrust::raw_pointer_cast(fvalue[0].data()) + data->rows - 1,
          data->rows));

      max = std::max(max, temp_storage_bytes);

      for (size_t i = 0; i < overlap_depth; ++i) {
        AllocateMemoryIfRequire(i, max);
      }

      temp_bytes_per_rec = temp_bytes_allocated[0] / data->rows;
    }
  }

  virtual ~TaylorApproximationBuilder() {
    for (auto i = 0; i < overlap_depth; ++i) {
      CubDebugExit(cudaFree(temp_bytes[i]));
      CubDebugExit(cudaFree(results[i]));
      CubDebugExit(cudaFreeHost(results_h[i]));
      CubDebugExit(cudaFreeHost(best_split_h[i]));
      CubDebugExit(cudaFree(run_lenght[i]));
      cudaStreamDestroy(streams[i]);
    }

    delete[] temp_bytes;
    delete[] results;
    delete[] results_h;
    delete[] best_split_h;
    delete[] run_lenght;
    delete[] streams;

    delete[] sum;
    delete[] fvalue;
    delete[] node_fvalue;
    delete[] node_fvalue_sorted;
    //    delete[] grad_sorted;
    delete[] grad_sorted_sorted;
  }

  virtual size_t MemoryRequirementsPerRecord() override {
    return sizeof(NODE_T) +        // node
           sizeof(GRAD_T) +        // grad
           (sizeof(SUM_T) +        // sum
            sizeof(unsigned int) + // fvalue
            sizeof(NODE_T) +       // node_fvalue
            sizeof(NODE_T) +       // node_fvalue_sorted
            sizeof(GRAD_T) +       // grad_sorted
            sizeof(GRAD_T) +       // grad_sorted_sorted
            temp_bytes_per_rec) *
               overlap_depth;
  }

  virtual void InitGrowingTree(const size_t columns) override {
    int take = (int)(param.colsample_bytree * columns);
    if (take == 0) {
      printf("colsample_bytree is too small %f for %ld columns \n",
             param.colsample_bytree, columns);
      throw "colsample_bytree is too small";
    }
    take = (int)(param.colsample_bytree * param.colsample_bylevel * columns);
    if (take == 0) {
      printf("colsample_bytree and colsample_bylevel are too small %f %f for "
             "%ld columns \n",
             param.colsample_bytree, param.colsample_bylevel, columns);
      throw "colsample_bytree and colsample_bylevel are too small";
    }

    for (size_t i = 0; i < columns; ++i) {
      active_fids[i] = i;
    }

    shuffle(active_fids.begin(), active_fids.end(), rnd);

    std::fill(_rowIndex2Node.begin(), _rowIndex2Node.end(), 0);
    for (size_t i = 0; i < _nodeStat.size(); ++i) {
      _nodeStat[i].Clean();
    }
    for (size_t i = 0; i < _bestSplit.size(); ++i) {
      _bestSplit[i].Clean();
    }
  }

  virtual void InitTreeLevel(const int level, const size_t columns) override {
    int take = (int)(param.colsample_bytree * columns);
    shuffle(active_fids.begin(), active_fids.begin() + take, rnd);
  }

  virtual void GrowTree(RegTree *tree, const io::DataMatrix *data,
                        const unsigned short label) override {

    cudaMemcpyAsync(
        thrust::raw_pointer_cast(grad_d.data()),
        thrust::raw_pointer_cast(objective->grad.data() + label * data->rows),
        data->rows * sizeof(GRAD_T), cudaMemcpyHostToDevice, streams[0]);

    grad_slice = const_cast<GRAD_T *>(
        thrust::raw_pointer_cast(objective->grad.data() + label * data->rows));

    InitGrowingTree(data->columns);

    for (unsigned int i = 0; (i + 1) < param.depth; ++i) {
      InitTreeLevel(i, data->columns);
      UpdateNodeStat(i, data, tree);
      FindBestSplits(i, data);
      UpdateTree(i, tree);
      UpdateNodeIndex(i, data, tree);
    }

    UpdateLeafWeight(tree);
  }

  virtual void PredictByGrownTree(RegTree *tree, io::DataMatrix *data,
                                  std::vector<float> &out) const override {
    tree->Predict(data, _rowIndex2Node, out);
  }

private:
  bool verbose;
  std::default_random_engine rnd;
  std::vector<unsigned int> active_fids;
  const unsigned short overlap_depth;
  const TreeParam param;
  const GainFunctionParameters gain_param;
  GRAD_T *grad_slice;
  const ApproximatedObjective<GRAD_T> *objective;
  host_vector<NODE_T, thrust::cuda::experimental::pinned_allocator<NODE_T>>
      _rowIndex2Node;
  std::vector<NodeStat<SUM_T>> _nodeStat;
  std::vector<Split<SUM_T>> _bestSplit;

  device_vector<SUM_T> *sum = new device_vector<SUM_T>[overlap_depth];
  device_vector<unsigned int> *fvalue =
      new device_vector<unsigned int>[overlap_depth];
  device_vector<NODE_T> *node_fvalue = new device_vector<NODE_T>[overlap_depth];
  device_vector<NODE_T> *node_fvalue_sorted =
      new device_vector<NODE_T>[overlap_depth];
  device_vector<GRAD_T> *grad_sorted_sorted =
      new device_vector<GRAD_T>[overlap_depth];
  cudaStream_t *streams = new cudaStream_t[overlap_depth];
  device_vector<GRAD_T> grad_d;
  device_vector<NODE_T> row2Node;
  device_vector<SUM_T> parent_node_sum;
  device_vector<unsigned int> parent_node_count;
  host_vector<SUM_T> parent_node_sum_h;
  host_vector<unsigned int> parent_node_count_h;
  size_t temp_bytes_per_rec = 0;
  size_t *temp_bytes_allocated = new size_t[overlap_depth];
  void **temp_bytes = new void *[overlap_depth];
  my_atomics **results = new my_atomics *[overlap_depth];
  my_atomics **results_h = new my_atomics *[overlap_depth];

  NODE_T **best_split_h = new NODE_T *[overlap_depth];
  unsigned int **run_lenght = new unsigned int *[overlap_depth];

  int blockSizeGain;
  int gridSizeGain;

  int blockSizeGather;
  int gridSizeGather;

  int blockSizeMax;
  int gridSizeMax;

  inline void AllocateMemoryIfRequire(const size_t circular_fid,
                                      const size_t bytes) {
    if (temp_bytes_allocated[circular_fid] == 0) {
      CubDebugExit(cudaMalloc(&(temp_bytes[circular_fid]), bytes));
      temp_bytes_allocated[circular_fid] = bytes;
    } else if (temp_bytes_allocated[circular_fid] < bytes) {
      CubDebugExit(cudaFree(temp_bytes[circular_fid]));
      CubDebugExit(cudaMalloc(&(temp_bytes[circular_fid]), bytes));
      temp_bytes_allocated[circular_fid] = bytes;
    }
  }

  void FindBestSplits(const unsigned int level, const io::DataMatrix *data) {

    cudaMemcpyAsync(thrust::raw_pointer_cast((row2Node.data())),
                    thrust::raw_pointer_cast(_rowIndex2Node.data()),
                    data->rows * sizeof(NODE_T), cudaMemcpyHostToDevice,
                    streams[0]);

    size_t lenght = 1 << level;

    {
      init(parent_node_sum_h[0]);
      parent_node_count_h[0] = 0;

      for (size_t i = 0; i < lenght; ++i) {
        parent_node_count_h[i + 1] =
            parent_node_count_h[i] + _nodeStat[i].count;
        parent_node_sum_h[i + 1] = parent_node_sum_h[i] + _nodeStat[i].sum_grad;
      }
      parent_node_sum = parent_node_sum_h;
      parent_node_count = parent_node_count_h;
    }

    unsigned int take = (unsigned int)(param.colsample_bylevel *
                                       param.colsample_bytree * data->columns);

    cudaStreamSynchronize(streams[0]);

    for (size_t j = 0; j < take; ++j) {

      for (size_t i = 0; i < overlap_depth && (j + i) < take; ++i) {

        if (j != 0 && (i + 1) < overlap_depth) {
          continue;
        }

        size_t active_fid = active_fids[j + i];
        size_t circular_fid = (j + i) % overlap_depth;

        if (active_fid < data->columns_dense) {
          if ((data->reduced_size[active_fid] + level) <
              sizeof(unsigned char) * CHAR_BIT) {
            ProcessDenseFeature<unsigned char>(active_fid, circular_fid, level,
                                               data);
          } else if ((data->reduced_size[active_fid] + level) <
                     sizeof(unsigned short) * CHAR_BIT) {
            ProcessDenseFeature<unsigned short>(active_fid, circular_fid, level,
                                                data);
          } else if ((data->reduced_size[active_fid] + level) <
                     sizeof(unsigned int) * CHAR_BIT) {
            ProcessDenseFeature<unsigned int>(active_fid, circular_fid, level,
                                              data);
          } else {
            ProcessDenseFeature<NODE_T>(active_fid, circular_fid, level, data);
          }
        } else {
          if ((data->category_size[active_fid - data->columns_dense] + level) <
              sizeof(unsigned char) * CHAR_BIT) {
            ProcessCategoryFeature<unsigned char>(
                active_fid - data->columns_dense, circular_fid, level, data);
          } else if ((data->category_size[active_fid - data->columns_dense] +
                      level) < sizeof(unsigned short) * CHAR_BIT) {
            ProcessCategoryFeature<unsigned short>(
                active_fid - data->columns_dense, circular_fid, level, data);
          } else if ((data->category_size[active_fid - data->columns_dense] +
                      level) < sizeof(unsigned int) * CHAR_BIT) {
            ProcessCategoryFeature<unsigned int>(
                active_fid - data->columns_dense, circular_fid, level, data);
          } else {
            ProcessCategoryFeature<NODE_T>(active_fid - data->columns_dense,
                                           circular_fid, level, data);
          }
        }
      }

      size_t circular_fid = j % overlap_depth;

      cudaStream_t s = streams[circular_fid];

      cudaStreamSynchronize(s);

      if (active_fids[j] < data->columns_dense) {
        if ((data->reduced_size[active_fids[j]] + level) <
            sizeof(unsigned char) * CHAR_BIT) {
          GetBestSplitForDenseFeature<unsigned char>(
              active_fids[j], circular_fid, lenght, data);
        } else if ((data->reduced_size[active_fids[j]] + level) <
                   sizeof(unsigned short) * CHAR_BIT) {
          GetBestSplitForDenseFeature<unsigned short>(
              active_fids[j], circular_fid, lenght, data);
        } else if ((data->reduced_size[active_fids[j]] + level) <
                   sizeof(unsigned int) * CHAR_BIT) {
          GetBestSplitForDenseFeature<unsigned int>(active_fids[j],
                                                    circular_fid, lenght, data);
        } else {
          GetBestSplitForDenseFeature<NODE_T>(active_fids[j], circular_fid,
                                              lenght, data);
        }
      } else {
        if ((data->category_size[active_fids[j] - data->columns_dense] +
             level) < sizeof(unsigned char) * CHAR_BIT) {
          GetBestSplitForCategoryFeature<unsigned char>(
              active_fids[j] - data->columns_dense, data->columns_dense,
              circular_fid, lenght, data);
        } else if ((data->category_size[active_fids[j] - data->columns_dense] +
                    level) < sizeof(unsigned short) * CHAR_BIT) {
          GetBestSplitForCategoryFeature<unsigned short>(
              active_fids[j] - data->columns_dense, data->columns_dense,
              circular_fid, lenght, data);
        } else if ((data->category_size[active_fids[j] - data->columns_dense] +
                    level) < sizeof(unsigned int) * CHAR_BIT) {
          GetBestSplitForCategoryFeature<unsigned int>(
              active_fids[j] - data->columns_dense, data->columns_dense,
              circular_fid, lenght, data);
        } else {
          GetBestSplitForCategoryFeature<NODE_T>(
              active_fids[j] - data->columns_dense, data->columns_dense,
              circular_fid, lenght, data);
        }
      }
    }

    for (size_t i = 0; i < lenght; ++i) {
      Split<SUM_T> &split = _bestSplit[i];

      if (split.fid < 0) {
        NodeStat<SUM_T> &node_stat = _nodeStat[i];
        _bestSplit[i].gain = 0.0;
        _bestSplit[i].fid = 0;
        _bestSplit[i].split_value = std::numeric_limits<float>::infinity();
        _bestSplit[i].count = node_stat.count;
        _bestSplit[i].sum_grad = node_stat.sum_grad;
      }
      }
  }

  template <typename NODE_VALUE_T>
  inline void
  GetBestSplitForDenseFeature(const int active_fid, const size_t circular_fid,
                              const size_t lenght, const io::DataMatrix *data) {
    for (size_t i = 0; i < lenght; ++i) {
      if (_nodeStat[i].count <= 0)
        continue;
      if (results_h[circular_fid][i].floats[0] > _bestSplit[i].gain) {
        const int index_value = results_h[circular_fid][i].ints[1];
        const SUM_T s = sum[circular_fid][index_value];
        if (!_isnan(s)) {

          cudaMemcpyAsync((NODE_VALUE_T *)best_split_h[circular_fid],
                          (NODE_VALUE_T *)thrust::raw_pointer_cast(
                              node_fvalue_sorted[circular_fid].data()) +
                              index_value - 1,
                          sizeof(NODE_VALUE_T), cudaMemcpyDeviceToHost,
                          streams[circular_fid]);

          cudaMemcpyAsync(((NODE_VALUE_T *)best_split_h[circular_fid]) + 1,
                          (NODE_VALUE_T *)thrust::raw_pointer_cast(
                              node_fvalue_sorted[circular_fid].data()) +
                              index_value,
                          sizeof(NODE_VALUE_T), cudaMemcpyDeviceToHost,
                          streams[circular_fid]);

          cudaStreamSynchronize(streams[circular_fid]);

          const NODE_VALUE_T segment_fvalue_prev_val =
              ((NODE_VALUE_T *)best_split_h[circular_fid])[0];
          const NODE_VALUE_T segment_fvalue_val =
              ((NODE_VALUE_T *)best_split_h[circular_fid])[1];

          const NODE_VALUE_T mask = (1 << (data->reduced_size[active_fid])) - 1;

          const NODE_VALUE_T fvalue_prev_val = segment_fvalue_prev_val & mask;
          const NODE_VALUE_T fvalue_val = segment_fvalue_val & mask;

          const size_t count_val =
              results_h[circular_fid][i].ints[1] - parent_node_count_h[i];

          const SUM_T sum_val = s - parent_node_sum_h[i];
          _bestSplit[i].fid = active_fid;
          _bestSplit[i].gain = results_h[circular_fid][i].floats[0];
          _bestSplit[i].split_value =
              (data->data_reduced_mapping[active_fid][fvalue_prev_val] +
               data->data_reduced_mapping[active_fid][fvalue_val]) *
              0.5;
          _bestSplit[i].count = count_val;
          _bestSplit[i].sum_grad = sum_val;
          _bestSplit[i].category = (unsigned int)-1;
        } else {
          if (verbose)
            printf("sum is nan(probably infinity), consider increasing the "
                   "accuracy \n");
        }
      }
    }
  }

  template <typename NODE_VALUE_T>
  inline void GetBestSplitForCategoryFeature(const int active_fid,
                                             const size_t columns_dense,
                                             const size_t circular_fid,
                                             const size_t lenght,
                                             const io::DataMatrix *data) {
    for (size_t i = 0; i < lenght; ++i) {
      if (_nodeStat[i].count <= 0)
        continue;
      if (results_h[circular_fid][i].floats[0] > _bestSplit[i].gain) {
        const int index_value = results_h[circular_fid][i].ints[1];
        const SUM_T sum_val = sum[circular_fid][index_value];
        if (!_isnan(sum_val)) {

          const unsigned int count_val = fvalue[circular_fid][index_value];
          if (count_val == 0)
            continue;

          cudaMemcpyAsync((NODE_VALUE_T *)best_split_h[circular_fid],
                          (NODE_VALUE_T *)thrust::raw_pointer_cast(
                              node_fvalue[circular_fid].data()) +
                              index_value,
                          sizeof(NODE_VALUE_T), cudaMemcpyDeviceToHost,
                          streams[circular_fid]);

          cudaStreamSynchronize(streams[circular_fid]);

          const NODE_VALUE_T segment_fvalues_val =
              ((NODE_VALUE_T *)best_split_h[circular_fid])[0];

          const NODE_VALUE_T mask =
              (1 << (data->category_size[active_fid])) - 1;

          const NODE_VALUE_T category_val = segment_fvalues_val & mask;

          _bestSplit[i].fid = active_fid + columns_dense;
          _bestSplit[i].gain = results_h[circular_fid][i].floats[0];
          _bestSplit[i].category = category_val;
          _bestSplit[i].count = count_val;
          _bestSplit[i].sum_grad = sum_val;
          _bestSplit[i].split_value = std::numeric_limits<float>::infinity();
        } else {
          if (verbose)
            printf("sum is nan(probably infinity), consider increasing the "
                   "accuracy \n");
        }
      }
    }
  }

  template <typename NODE_VALUE_T>
  inline void ProcessDenseFeature(const size_t active_fid,
                                  const size_t circular_fid, const size_t level,
                                  const io::DataMatrix *data) {

    size_t lenght = 1 << level;

    cudaStream_t s = streams[circular_fid];

    device_vector<unsigned int> *fvalue_tmp = NULL;

    cudaMemsetAsync(results[circular_fid], 0, lenght * sizeof(my_atomics), s);

    if (data->sorted_data_device[active_fid].size() > 0) {
      fvalue_tmp = const_cast<device_vector<unsigned int> *>(
          &(data->sorted_data_device[active_fid]));
    } else {
      cudaMemcpyAsync(
          thrust::raw_pointer_cast((fvalue[circular_fid].data())),
          thrust::raw_pointer_cast(data->data_reduced[active_fid].data()),
          data->rows * sizeof(unsigned int), cudaMemcpyHostToDevice, s);
      cudaStreamSynchronize(s);
      fvalue_tmp =
          const_cast<device_vector<unsigned int> *>(&(fvalue[circular_fid]));
    }

    assign_kernel<<<gridSizeGather, blockSizeGather, 0, s>>>(
        thrust::raw_pointer_cast(fvalue_tmp->data()),
        thrust::raw_pointer_cast(row2Node.data()),
        data->reduced_size[active_fid],
        (NODE_VALUE_T *)thrust::raw_pointer_cast(
            node_fvalue[circular_fid].data()),
        data->rows);

    CubDebugExit(cub::DeviceRadixSort::SortPairs(
        temp_bytes[circular_fid], temp_bytes_allocated[circular_fid],
        (NODE_VALUE_T *)thrust::raw_pointer_cast(
            node_fvalue[circular_fid].data()),
        (NODE_VALUE_T *)thrust::raw_pointer_cast(
            node_fvalue_sorted[circular_fid].data()),
        thrust::raw_pointer_cast(grad_d.data()),
        thrust::raw_pointer_cast(grad_sorted_sorted[circular_fid].data()),
        data->rows, 0, data->reduced_size[active_fid] + level + 1, s));

    SUM_T initial_value;
    init(initial_value);
    cub::Sum sum_op;

    CubDebugExit(cub::DeviceScan::ExclusiveScan(
        temp_bytes[circular_fid], temp_bytes_allocated[circular_fid],
        thrust::raw_pointer_cast(grad_sorted_sorted[circular_fid].data()),
        thrust::raw_pointer_cast(sum[circular_fid].data()), sum_op,
        initial_value, data->rows, s));

    const NODE_VALUE_T mask = (1 << (data->reduced_size[active_fid])) - 1;

    gain_kernel<<<gridSizeGain, blockSizeGain, 0, s>>>(
        thrust::raw_pointer_cast(sum[circular_fid].data()),
        (NODE_VALUE_T *)thrust::raw_pointer_cast(
            node_fvalue_sorted[circular_fid].data()),
        data->reduced_size[active_fid], mask,
        thrust::raw_pointer_cast(parent_node_sum.data()),
        thrust::raw_pointer_cast(parent_node_count.data()), data->rows,
        gain_param, results[circular_fid]);

    cudaMemcpyAsync(results_h[circular_fid], results[circular_fid],
                    lenght * sizeof(my_atomics), cudaMemcpyDeviceToHost, s);
  }

  template <typename NODE_VALUE_T>
  inline void
  ProcessCategoryFeature(const size_t active_fid, const size_t circular_fid,
                         const size_t level, const io::DataMatrix *data) {
    size_t lenght = 1 << level;

    cudaStream_t s = streams[circular_fid];

    device_vector<unsigned int> *fvalue_tmp = NULL;

    cudaMemsetAsync(results[circular_fid], 0, lenght * sizeof(my_atomics), s);

    if (data->sorted_data_device[active_fid].size() > 0) {
      fvalue_tmp = const_cast<device_vector<unsigned int> *>(
          &(data->data_category_device[active_fid]));
    } else {
      cudaMemcpyAsync(
          thrust::raw_pointer_cast((fvalue[circular_fid].data())),
          thrust::raw_pointer_cast(data->data_categories[active_fid].data()),
          data->rows * sizeof(unsigned int), cudaMemcpyHostToDevice, s);
      cudaStreamSynchronize(s);
      fvalue_tmp =
          const_cast<device_vector<unsigned int> *>(&(fvalue[circular_fid]));
    }

    assign_kernel<<<gridSizeGather, blockSizeGather, 0, s>>>(
        thrust::raw_pointer_cast(fvalue_tmp->data()),
        thrust::raw_pointer_cast(row2Node.data()),
        data->category_size[active_fid],
        (NODE_VALUE_T *)thrust::raw_pointer_cast(
            node_fvalue[circular_fid].data()),
        data->rows);

    CubDebugExit(cub::DeviceRadixSort::SortPairs(
        temp_bytes[circular_fid], temp_bytes_allocated[circular_fid],
        (NODE_VALUE_T *)thrust::raw_pointer_cast(
            node_fvalue[circular_fid].data()),
        (NODE_VALUE_T *)thrust::raw_pointer_cast(
            node_fvalue_sorted[circular_fid].data()),
        thrust::raw_pointer_cast(grad_d.data()),
        thrust::raw_pointer_cast(grad_sorted_sorted[circular_fid].data()),
        data->rows, 0, data->category_size[active_fid] + level + 1, s));

    const NODE_VALUE_T mask = (1 << (data->category_size[active_fid])) - 1;

    SUM_T initial_value;
    init(initial_value);
    cub::Sum sum_op;

    CubDebugExit(cub::DeviceReduce::ReduceByKey(
        temp_bytes[circular_fid], temp_bytes_allocated[circular_fid],
        (NODE_VALUE_T *)thrust::raw_pointer_cast(
            node_fvalue_sorted[circular_fid].data()),
        (NODE_VALUE_T *)thrust::raw_pointer_cast(
            node_fvalue[circular_fid].data()),
        thrust::raw_pointer_cast(grad_sorted_sorted[circular_fid].data()),
        thrust::raw_pointer_cast(sum[circular_fid].data()),
        run_lenght[circular_fid], sum_op, data->rows, s));

    CubDebugExit(cub::DeviceRunLengthEncode::Encode(
        temp_bytes[circular_fid], temp_bytes_allocated[circular_fid],
        (NODE_VALUE_T *)thrust::raw_pointer_cast(
            node_fvalue_sorted[circular_fid].data()),
        (NODE_VALUE_T *)thrust::raw_pointer_cast(
            node_fvalue[circular_fid].data()),
        thrust::raw_pointer_cast(fvalue[circular_fid].data()),
        run_lenght[circular_fid], data->rows, s));

    gain_kernel_category<<<gridSizeGain, blockSizeGain, 0, s>>>(
        thrust::raw_pointer_cast(sum[circular_fid].data()),
        thrust::raw_pointer_cast(fvalue[circular_fid].data()),
        (NODE_VALUE_T *)thrust::raw_pointer_cast(
            node_fvalue[circular_fid].data()),
        data->category_size[active_fid], mask,
        thrust::raw_pointer_cast(parent_node_sum.data()),
        thrust::raw_pointer_cast(parent_node_count.data()),
        run_lenght[circular_fid], gain_param, results[circular_fid]);

    cudaMemcpyAsync(results_h[circular_fid], results[circular_fid],
                    lenght * sizeof(my_atomics), cudaMemcpyDeviceToHost, s);
  }

  void UpdateNodeStat(const int level, const io::DataMatrix *data,
                      const RegTree *tree) {
    if (level != 0) {
      const unsigned int offset = Node::HeapOffset(level);
      const unsigned int offset_next = Node::HeapOffset(level + 1);
      std::vector<NodeStat<SUM_T>> tmp(_nodeStat.size());
      std::copy(_nodeStat.begin(), _nodeStat.end(), tmp.begin());

      size_t len = 1 << (level - 1);

      for (size_t i = 0; i < len; ++i) {
        _nodeStat[tree->ChildNode(i + offset, true) - offset_next].count =
            _bestSplit[i].count;
        _nodeStat[tree->ChildNode(i + offset, true) - offset_next].sum_grad =
            _bestSplit[i].sum_grad;

        _nodeStat[tree->ChildNode(i + offset, false) - offset_next].count =
            tmp[i].count - _bestSplit[i].count;

        _nodeStat[tree->ChildNode(i + offset, false) - offset_next].sum_grad =
            tmp[i].sum_grad - _bestSplit[i].sum_grad;
      }
    } else {
      _nodeStat[0].count = data->rows;

      SUM_T sum;
      init(sum);

#pragma omp parallel
      {
        SUM_T sum_thread;
        init(sum_thread);
#pragma omp for simd
        for (size_t i = 0; i < data->rows; ++i) {
          sum_thread += grad_slice[i];
        }
#pragma omp critical
        { sum += sum_thread; }
      }
      _nodeStat[0].sum_grad = sum;
    }

    size_t len = 1 << level;

    for (size_t i = 0; i < len; ++i) {
      _nodeStat[i].gain =
          0.0; // todo: gain_func(_nodeStat[i].count, _nodeStat[i].sum_grad);
      _bestSplit[i].Clean();
    }
  }

  void UpdateTree(const int level, RegTree *tree) const {
    unsigned int offset = Node::HeapOffset(level);

    const size_t len = 1 << level;

    for (size_t i = 0; i < len; ++i) {
      const Split<SUM_T> &best = _bestSplit[i];
      tree->nodes[i + offset].threshold = best.split_value;
      tree->nodes[i + offset].category = best.category;
      tree->nodes[i + offset].fid = best.fid < 0 ? 0 : best.fid;
    }
  }

  void UpdateNodeIndex(const unsigned int level, const io::DataMatrix *data,
                       RegTree *tree) {
    unsigned int const offset = Node::HeapOffset(level);
    unsigned int const offset_next = Node::HeapOffset(level + 1);

#pragma omp parallel for simd
    for (size_t i = 0; i < data->rows; ++i) {
      const unsigned int node = _rowIndex2Node[i];
      const auto &best = _bestSplit[node];
      const bool isLeft =
          (best.fid < (int)data->columns_dense &&
           data->data[best.fid][i] <= best.split_value) ||
          (best.fid >= (int)data->columns_dense &&
           data->data_categories[best.fid - data->columns_dense][i] ==
               best.category);
      _rowIndex2Node[i] = tree->ChildNode(node + offset, isLeft) - offset_next;
    }
  }

  void UpdateLeafWeight(RegTree *tree) const {
    const unsigned int offset_1 = Node::HeapOffset(tree->depth - 2);
    const unsigned int offset = Node::HeapOffset(tree->depth - 1);
    for (unsigned int i = 0, len = (1 << (tree->depth - 2)); i < len; ++i) {
      const Split<SUM_T> &best = _bestSplit[i];
      const NodeStat<SUM_T> &stat = _nodeStat[i];
      tree->leaf_level[tree->ChildNode(i + offset_1, true) - offset] =
          best.LeafWeight(param) * param.eta;
      tree->leaf_level[tree->ChildNode(i + offset_1, false) - offset] =
          best.LeafWeight(stat, param) * param.eta;
    }
  }
};

Garden::Garden(const TreeParam &param, const Verbose &verbose,
               const InternalConfiguration &cfg)
    : param(param), verbose(verbose), cfg(cfg), _init(false) {}

void Garden::GrowTree(io::DataMatrix *data, float *grad) {

  data->Init(verbose.data);

  if (!_init) {
    switch (param.objective) {
    case LinearRegression: {
      auto obj = new RegressionObjective(data, param.initial_y);

      if (data->max_feature_size + param.depth + 1 <=
          sizeof(unsigned char) * CHAR_BIT) {
        if (cfg.double_precision) {
          _builder =
              new TaylorApproximationBuilder<unsigned char, float, double>(
                  param, data, cfg, obj, verbose.booster);
        } else {
          _builder =
              new TaylorApproximationBuilder<unsigned char, float, float>(
                  param, data, cfg, obj, verbose.booster);
        }
      } else if (data->max_feature_size + param.depth + 1 <=
                 sizeof(unsigned short) * CHAR_BIT) {
        if (cfg.double_precision) {
          _builder =
              new TaylorApproximationBuilder<unsigned short, float, double>(
                  param, data, cfg, obj, verbose.booster);
        } else {
          _builder =
              new TaylorApproximationBuilder<unsigned short, float, float>(
                  param, data, cfg, obj, verbose.booster);
        }
      } else if (data->max_feature_size + param.depth + 1 <=
                 sizeof(unsigned int) * CHAR_BIT) {
        if (cfg.double_precision) {
          _builder =
              new TaylorApproximationBuilder<unsigned int, float, double>(
                  param, data, cfg, obj, verbose.booster);
        } else {
          _builder = new TaylorApproximationBuilder<unsigned int, float, float>(
              param, data, cfg, obj, verbose.booster);
        }
      } else if (data->max_feature_size + param.depth + 1 <=
                 sizeof(unsigned long) * CHAR_BIT) {
        if (cfg.double_precision) {
          _builder =
              new TaylorApproximationBuilder<unsigned long, float, double>(
                  param, data, cfg, obj, verbose.booster);
        } else {
          _builder =
              new TaylorApproximationBuilder<unsigned long, float, float>(
                  param, data, cfg, obj, verbose.booster);
        }
      } else if (data->max_feature_size + param.depth + 1 <=
                 sizeof(unsigned long long) * CHAR_BIT) {
        if (cfg.double_precision) {
          _builder =
              new TaylorApproximationBuilder<unsigned long long, float, double>(
                  param, data, cfg, obj, verbose.booster);
        } else {
          _builder =
              new TaylorApproximationBuilder<unsigned long long, float, float>(
                  param, data, cfg, obj, verbose.booster);
        }
      } else {
        throw "unsupported depth";
      }
      _objective = obj;
    }

    break;
    case LogisticRegression: {
      auto obj = new LogisticRegressionObjective(data, param.initial_y);

      if (data->max_feature_size + param.depth + 1 <=
          sizeof(unsigned char) * CHAR_BIT) {
        if (cfg.double_precision) {
          _builder =
              new TaylorApproximationBuilder<unsigned char, float2, mydouble2>(
                  param, data, cfg, obj, verbose.booster);
        } else {
          _builder =
              new TaylorApproximationBuilder<unsigned char, float2, float2>(
                  param, data, cfg, obj, verbose.booster);
        }
      } else if (data->max_feature_size + param.depth + 1 <=
                 sizeof(unsigned short) * CHAR_BIT) {
        if (cfg.double_precision) {
          _builder =
              new TaylorApproximationBuilder<unsigned short, float2, mydouble2>(
                  param, data, cfg, obj, verbose.booster);
        } else {
          _builder =
              new TaylorApproximationBuilder<unsigned short, float2, float2>(
                  param, data, cfg, obj, verbose.booster);
        }
      } else if (data->max_feature_size + param.depth + 1 <=
                 sizeof(unsigned int) * CHAR_BIT) {
        if (cfg.double_precision) {
          _builder =
              new TaylorApproximationBuilder<unsigned int, float2, mydouble2>(
                  param, data, cfg, obj, verbose.booster);
        } else {
          _builder =
              new TaylorApproximationBuilder<unsigned int, float2, float2>(
                  param, data, cfg, obj, verbose.booster);
        }
      } else if (data->max_feature_size + param.depth + 1 <=
                 sizeof(unsigned long) * CHAR_BIT) {
        if (cfg.double_precision) {
          _builder =
              new TaylorApproximationBuilder<unsigned long, float2, mydouble2>(
                  param, data, cfg, obj, verbose.booster);
        } else {
          _builder =
              new TaylorApproximationBuilder<unsigned long, float2, float2>(
                  param, data, cfg, obj, verbose.booster);
        }
      } else if (data->max_feature_size + param.depth + 1 <=
                 sizeof(unsigned long long) * CHAR_BIT) {
        if (cfg.double_precision) {
          _builder =
              new TaylorApproximationBuilder<unsigned long, float2, mydouble2>(
                  param, data, cfg, obj, verbose.booster);
        } else {
          _builder = new TaylorApproximationBuilder<unsigned long long, float2,
                                                    float2>(
              param, data, cfg, obj, verbose.booster);
        }
      } else {
        throw "unsupported depth";
      }
      _objective = obj;
    } break;
    case SoftMaxOneVsAll: {
      auto obj =
          new SoftMaxObjective(data, param.labels_count, param.initial_y);

      if (data->max_feature_size + param.depth + 1 <=
          sizeof(unsigned char) * CHAR_BIT) {
        if (cfg.double_precision) {
          _builder =
              new TaylorApproximationBuilder<unsigned char, float2, mydouble2>(
                  param, data, cfg, obj, verbose.booster);
        } else {
          _builder =
              new TaylorApproximationBuilder<unsigned char, float2, float2>(
                  param, data, cfg, obj, verbose.booster);
        }
      } else if (data->max_feature_size + param.depth + 1 <=
                 sizeof(unsigned short) * CHAR_BIT) {
        if (cfg.double_precision) {
          _builder =
              new TaylorApproximationBuilder<unsigned short, float2, mydouble2>(
                  param, data, cfg, obj, verbose.booster);
        } else {
          _builder =
              new TaylorApproximationBuilder<unsigned short, float2, float2>(
                  param, data, cfg, obj, verbose.booster);
        }
      } else if (data->max_feature_size + param.depth + 1 <=
                 sizeof(unsigned int) * CHAR_BIT) {
        if (cfg.double_precision) {
          _builder =
              new TaylorApproximationBuilder<unsigned int, float2, mydouble2>(
                  param, data, cfg, obj, verbose.booster);
        } else {
          _builder =
              new TaylorApproximationBuilder<unsigned int, float2, float2>(
                  param, data, cfg, obj, verbose.booster);
        }
      } else if (data->max_feature_size + param.depth + 1 <=
                 sizeof(unsigned long) * CHAR_BIT) {
        if (cfg.double_precision) {
          _builder =
              new TaylorApproximationBuilder<unsigned long, float2, mydouble2>(
                  param, data, cfg, obj, verbose.booster);
        } else {
          _builder =
              new TaylorApproximationBuilder<unsigned long, float2, float2>(
                  param, data, cfg, obj, verbose.booster);
        }
      } else if (data->max_feature_size + param.depth + 1 <=
                 sizeof(unsigned long long) * CHAR_BIT) {
        if (cfg.double_precision) {
          _builder = new TaylorApproximationBuilder<unsigned long long, float2,
                                                    mydouble2>(
              param, data, cfg, obj, verbose.booster);
        } else {
          _builder = new TaylorApproximationBuilder<unsigned long long, float2,
                                                    float2>(
              param, data, cfg, obj, verbose.booster);
        }
      } else {
        throw "unsupported depth";
      }
      _objective = obj;
    } break;
    default:
      throw "Unknown objective function " + param.objective;
    }

    auto mem_per_rec = _builder->MemoryRequirementsPerRecord();
    size_t total;
    size_t free;

    cudaMemGetInfo(&free, &total);

    if (verbose.gpu) {
      printf("Total bytes %ld avaliable %ld \n", total, free);
      printf("Memory usage estimation %ld per record %ld in total \n",
             mem_per_rec, mem_per_rec * data->rows);
    }

    data->TransferToGPU(free * 9 / 10, verbose.gpu);

    _init = true;
  }

  if (grad == NULL) {
    _objective->UpdateGrad();
  } else {
    //          todo: fix
    //          data->grad = std::vector<float>(grad, grad + data->rows);
  }

  for (unsigned short i = 0; i < param.labels_count; ++i) {
    RegTree *tree = new RegTree(param.depth, i);
    _builder->GrowTree(tree, data, i);
    _trees.push_back(tree);
    if (grad == NULL) {
      _builder->PredictByGrownTree(tree, data, data->y_internal);
    }
  }
}

void Garden::UpdateByLastTree(io::DataMatrix *data) {
  if (data->y_internal.size() == 0)
    data->y_internal.resize(data->rows * param.labels_count,
                            _objective->IntoInternal(param.initial_y));
  for (auto it = _trees.end() - param.labels_count; it != _trees.end(); ++it) {
    (*it)->Predict(data, data->y_internal);
  }
}

void Garden::GetY(arboretum::io::DataMatrix *data,
                  std::vector<float> &out) const {
  out.resize(data->y_internal.size());
  _objective->FromInternal(data->y_internal, out);
}

void Garden::Predict(const arboretum::io::DataMatrix *data,
                     std::vector<float> &out) const {
  out.resize(data->rows * param.labels_count);
  std::vector<float> tmp(data->rows * param.labels_count);

  std::fill(tmp.begin(), tmp.end(), _objective->IntoInternal(param.initial_y));
  for (size_t i = 0; i < _trees.size(); ++i) {
    _trees[i]->Predict(data, tmp);
  }

  _objective->FromInternal(tmp, out);
}
} // namespace core
} // namespace arboretum
