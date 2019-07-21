#include "continuous_tree_grower.h"
#include "cuda_helpers.h"

namespace arboretum {
namespace core {
using thrust::device_vector;
using thrust::host_vector;

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

template <typename NODE_T, typename GRAD_T, typename SUM_T>
ContinuousTreeGrower<NODE_T, GRAD_T, SUM_T>::ContinuousTreeGrower(
  const size_t size, const unsigned depth, const unsigned hist_size,
  const BestSplit<SUM_T> *best, Histogram<SUM_T> *features_histogram,
  const InternalConfiguration *config)
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

  OK(cub::DeviceScan::ExclusiveScan(NULL, temp_storage_bytes, (GRAD_T *)nullptr,
                                    (SUM_T *)nullptr, sum_op, initial_value,
                                    size));

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

template <typename NODE_T, typename GRAD_T, typename SUM_T>
inline void ContinuousTreeGrower<NODE_T, GRAD_T, SUM_T>::ApplySplit(
  NODE_T *row2Node, const unsigned level, const unsigned threshold, size_t from,
  size_t to) {
  int gridSize;
  int blockSize;
  compute1DInvokeConfig(to - from, &gridSize, &blockSize, apply_split<NODE_T>);

  apply_split<NODE_T><<<gridSize, blockSize, 0, this->stream>>>(
    row2Node + from,
    ((unsigned *)thrust::raw_pointer_cast(node_fvalue.data())) + from,
    threshold, level, to - from);
}

template <typename NODE_T, typename GRAD_T, typename SUM_T>
template <typename NODE_VALUE_T>
inline void ContinuousTreeGrower<NODE_T, GRAD_T, SUM_T>::ProcessDenseFeature(
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

  unsigned int *fvalue_tmp = NULL;

  if (fvalue_d != nullptr) {
    fvalue_tmp = fvalue_d;
  } else {
    OK(cudaMemcpyAsync(thrust::raw_pointer_cast(this->fvalue.data()), fvalue_h,
                       this->size * sizeof(unsigned int),
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
      OK(cudaMemcpyAsync(fvalue_d, thrust::raw_pointer_cast(node_fvalue.data()),
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
    thrust::raw_pointer_cast(this->grad_sorted.data()), this->size, 1 << level,
    thrust::raw_pointer_cast(parent_node_count.data()),
    thrust::raw_pointer_cast(parent_node_count.data()) + 1, 0, fvalue_size + 1,
    this->stream));

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
    thrust::raw_pointer_cast(parent_node_count.data()), this->size, gain_param,
    thrust::raw_pointer_cast(this->result_d.data()));
}

template <typename NODE_T, typename GRAD_T, typename SUM_T>
inline void ContinuousTreeGrower<NODE_T, GRAD_T, SUM_T>::FindBest(
  BestSplit<SUM_T> &best, device_vector<NODE_T> &row2Node,
  const device_vector<SUM_T> &parent_node_sum,
  const device_vector<unsigned int> &parent_node_count, const unsigned fid,
  const unsigned level, const unsigned depth, const unsigned size) {
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
      thrust::raw_pointer_cast(parent_node_sum.data()), fid, depth - level - 2,
      size);
}

template class ContinuousTreeGrower<unsigned, float, float>;

template void ContinuousTreeGrower<unsigned, float, float>::ProcessDenseFeature<
  unsigned>(const device_vector<unsigned> &row2Node,
            const device_vector<float> &grad_d, unsigned int *fvalue_d,
            unsigned int *fvalue_h, const device_vector<float> &parent_node_sum,
            const device_vector<unsigned int> &parent_node_count,
            const unsigned char fvalue_size, const unsigned level,
            const unsigned depth, const GainFunctionParameters gain_param,
            const bool partition_only, const int fid);

template void ContinuousTreeGrower<unsigned, float, float>::ProcessDenseFeature<
  unsigned long>(const device_vector<unsigned> &row2Node,
                 const device_vector<float> &grad_d, unsigned int *fvalue_d,
                 unsigned int *fvalue_h,
                 const device_vector<float> &parent_node_sum,
                 const device_vector<unsigned int> &parent_node_count,
                 const unsigned char fvalue_size, const unsigned level,
                 const unsigned depth, const GainFunctionParameters gain_param,
                 const bool partition_only, const int fid);

template class ContinuousTreeGrower<unsigned, float, double>;

template void
ContinuousTreeGrower<unsigned, float, double>::ProcessDenseFeature<unsigned>(
  const device_vector<unsigned> &row2Node, const device_vector<float> &grad_d,
  unsigned int *fvalue_d, unsigned int *fvalue_h,
  const device_vector<double> &parent_node_sum,
  const device_vector<unsigned int> &parent_node_count,
  const unsigned char fvalue_size, const unsigned level, const unsigned depth,
  const GainFunctionParameters gain_param, const bool partition_only,
  const int fid);

template void
ContinuousTreeGrower<unsigned, float, double>::ProcessDenseFeature<
  unsigned long>(const device_vector<unsigned> &row2Node,
                 const device_vector<float> &grad_d, unsigned int *fvalue_d,
                 unsigned int *fvalue_h,
                 const device_vector<double> &parent_node_sum,
                 const device_vector<unsigned int> &parent_node_count,
                 const unsigned char fvalue_size, const unsigned level,
                 const unsigned depth, const GainFunctionParameters gain_param,
                 const bool partition_only, const int fid);

template class ContinuousTreeGrower<unsigned, float2, float2>;

template void
ContinuousTreeGrower<unsigned, float2, float2>::ProcessDenseFeature<unsigned>(
  const device_vector<unsigned> &row2Node, const device_vector<float2> &grad_d,
  unsigned int *fvalue_d, unsigned int *fvalue_h,
  const device_vector<float2> &parent_node_sum,
  const device_vector<unsigned int> &parent_node_count,
  const unsigned char fvalue_size, const unsigned level, const unsigned depth,
  const GainFunctionParameters gain_param, const bool partition_only,
  const int fid);

template void
ContinuousTreeGrower<unsigned, float2, float2>::ProcessDenseFeature<
  unsigned long>(const device_vector<unsigned> &row2Node,
                 const device_vector<float2> &grad_d, unsigned int *fvalue_d,
                 unsigned int *fvalue_h,
                 const device_vector<float2> &parent_node_sum,
                 const device_vector<unsigned int> &parent_node_count,
                 const unsigned char fvalue_size, const unsigned level,
                 const unsigned depth, const GainFunctionParameters gain_param,
                 const bool partition_only, const int fid);

template class ContinuousTreeGrower<unsigned, float2, mydouble2>;

template void
ContinuousTreeGrower<unsigned, float2, mydouble2>::ProcessDenseFeature<
  unsigned>(const device_vector<unsigned> &row2Node,
            const device_vector<float2> &grad_d, unsigned int *fvalue_d,
            unsigned int *fvalue_h,
            const device_vector<mydouble2> &parent_node_sum,
            const device_vector<unsigned int> &parent_node_count,
            const unsigned char fvalue_size, const unsigned level,
            const unsigned depth, const GainFunctionParameters gain_param,
            const bool partition_only, const int fid);

template void
ContinuousTreeGrower<unsigned, float2, mydouble2>::ProcessDenseFeature<
  unsigned long>(const device_vector<unsigned> &row2Node,
                 const device_vector<float2> &grad_d, unsigned int *fvalue_d,
                 unsigned int *fvalue_h,
                 const device_vector<mydouble2> &parent_node_sum,
                 const device_vector<unsigned int> &parent_node_count,
                 const unsigned char fvalue_size, const unsigned level,
                 const unsigned depth, const GainFunctionParameters gain_param,
                 const bool partition_only, const int fid);

template class ContinuousTreeGrower<unsigned short, float, float>;
template class ContinuousTreeGrower<unsigned short, float, double>;
template class ContinuousTreeGrower<unsigned short, float2, float2>;
template class ContinuousTreeGrower<unsigned short, float2, mydouble2>;

template class ContinuousTreeGrower<unsigned char, float, float>;
template class ContinuousTreeGrower<unsigned char, float, double>;
template class ContinuousTreeGrower<unsigned char, float2, float2>;
template class ContinuousTreeGrower<unsigned char, float2, mydouble2>;

template class ContinuousTreeGrower<unsigned long, float, float>;

template void
ContinuousTreeGrower<unsigned long, float, float>::ProcessDenseFeature<
  unsigned>(const device_vector<unsigned long> &row2Node,
            const device_vector<float> &grad_d, unsigned int *fvalue_d,
            unsigned int *fvalue_h, const device_vector<float> &parent_node_sum,
            const device_vector<unsigned int> &parent_node_count,
            const unsigned char fvalue_size, const unsigned level,
            const unsigned depth, const GainFunctionParameters gain_param,
            const bool partition_only, const int fid);

template void
ContinuousTreeGrower<unsigned long, float, float>::ProcessDenseFeature<
  unsigned long>(const device_vector<unsigned long> &row2Node,
                 const device_vector<float> &grad_d, unsigned int *fvalue_d,
                 unsigned int *fvalue_h,
                 const device_vector<float> &parent_node_sum,
                 const device_vector<unsigned int> &parent_node_count,
                 const unsigned char fvalue_size, const unsigned level,
                 const unsigned depth, const GainFunctionParameters gain_param,
                 const bool partition_only, const int fid);

template class ContinuousTreeGrower<unsigned long, float, double>;

template void
ContinuousTreeGrower<unsigned long, float, double>::ProcessDenseFeature<
  unsigned>(const device_vector<unsigned long> &row2Node,
            const device_vector<float> &grad_d, unsigned int *fvalue_d,
            unsigned int *fvalue_h,
            const device_vector<double> &parent_node_sum,
            const device_vector<unsigned int> &parent_node_count,
            const unsigned char fvalue_size, const unsigned level,
            const unsigned depth, const GainFunctionParameters gain_param,
            const bool partition_only, const int fid);

template void
ContinuousTreeGrower<unsigned long, float, double>::ProcessDenseFeature<
  unsigned long>(const device_vector<unsigned long> &row2Node,
                 const device_vector<float> &grad_d, unsigned int *fvalue_d,
                 unsigned int *fvalue_h,
                 const device_vector<double> &parent_node_sum,
                 const device_vector<unsigned int> &parent_node_count,
                 const unsigned char fvalue_size, const unsigned level,
                 const unsigned depth, const GainFunctionParameters gain_param,
                 const bool partition_only, const int fid);

template class ContinuousTreeGrower<unsigned long, float2, float2>;

template void
ContinuousTreeGrower<unsigned long, float2, float2>::ProcessDenseFeature<
  unsigned>(const device_vector<unsigned long> &row2Node,
            const device_vector<float2> &grad_d, unsigned int *fvalue_d,
            unsigned int *fvalue_h,
            const device_vector<float2> &parent_node_sum,
            const device_vector<unsigned int> &parent_node_count,
            const unsigned char fvalue_size, const unsigned level,
            const unsigned depth, const GainFunctionParameters gain_param,
            const bool partition_only, const int fid);

template void
ContinuousTreeGrower<unsigned long, float2, float2>::ProcessDenseFeature<
  unsigned long>(const device_vector<unsigned long> &row2Node,
                 const device_vector<float2> &grad_d, unsigned int *fvalue_d,
                 unsigned int *fvalue_h,
                 const device_vector<float2> &parent_node_sum,
                 const device_vector<unsigned int> &parent_node_count,
                 const unsigned char fvalue_size, const unsigned level,
                 const unsigned depth, const GainFunctionParameters gain_param,
                 const bool partition_only, const int fid);

template class ContinuousTreeGrower<unsigned long, float2, mydouble2>;

template void
ContinuousTreeGrower<unsigned long, float2, mydouble2>::ProcessDenseFeature<
  unsigned>(const device_vector<unsigned long> &row2Node,
            const device_vector<float2> &grad_d, unsigned int *fvalue_d,
            unsigned int *fvalue_h,
            const device_vector<mydouble2> &parent_node_sum,
            const device_vector<unsigned int> &parent_node_count,
            const unsigned char fvalue_size, const unsigned level,
            const unsigned depth, const GainFunctionParameters gain_param,
            const bool partition_only, const int fid);

template void
ContinuousTreeGrower<unsigned long, float2, mydouble2>::ProcessDenseFeature<
  unsigned long>(const device_vector<unsigned long> &row2Node,
                 const device_vector<float2> &grad_d, unsigned int *fvalue_d,
                 unsigned int *fvalue_h,
                 const device_vector<mydouble2> &parent_node_sum,
                 const device_vector<unsigned int> &parent_node_count,
                 const unsigned char fvalue_size, const unsigned level,
                 const unsigned depth, const GainFunctionParameters gain_param,
                 const bool partition_only, const int fid);

}  // namespace core
}  // namespace arboretum
