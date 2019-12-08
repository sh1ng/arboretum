#include "builder.h"
#include "cuda_helpers.h"
#include "hist_grad_sum.cuh"
#include "hist_tree_grower.h"

namespace arboretum {
namespace core {

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

template <typename NODE_T, typename SUM_T>
__global__ void hist_apply_candidates(
  float *gain, int *features, SUM_T *sum, unsigned *split, unsigned *count,
  unsigned *node_size_prefix_sum_next, SUM_T *node_sum_prefix_sum_next,
  const my_atomics *candidates, const SUM_T *split_sum,
  const unsigned *split_count, const unsigned short *fvalue, NODE_T *row2Node,
  const unsigned *node_size_prefix_sum, const SUM_T *node_sum_prefix_sum,
  const int feature, const unsigned level, const unsigned hist_size,
  const unsigned n) {
  constexpr unsigned per = 1;
  const unsigned i = blockIdx.x / per;
  __shared__ unsigned node_start;
  __shared__ unsigned node_end;
  __shared__ unsigned node_size;
  __shared__ float gain_;
  __shared__ unsigned idx;
  __shared__ unsigned split_count_value;
  __shared__ SUM_T node_start_sum;
  __shared__ SUM_T node_end_sum;

  if (threadIdx.x == 0) {
    node_start = node_size_prefix_sum[i];
    node_end = node_size_prefix_sum[i + 1];
    node_size = node_end - node_start;
    gain_ = candidates[i].floats[0];
    idx = candidates[i].ints[1];
    split_count_value = split_count[idx];
    node_start_sum = node_sum_prefix_sum[i];
    node_end_sum = node_sum_prefix_sum[i + 1];
  }

  __syncthreads();

  if (node_size > 0) {
    if (gain[i] < gain_) {
      unsigned threshold = idx % hist_size;
      if (threadIdx.x == 0 && blockIdx.x % per == 0) {
        gain[i] = gain_;
        features[i] = feature;
        sum[i] = split_sum[idx] - node_start_sum;
        count[i] = split_count_value - node_start;
        split[i] = threshold;
      }

      for (unsigned j = threadIdx.x + (blockIdx.x % per) * blockDim.x;
           j < node_size; j += blockDim.x * per) {
        const unsigned right =
          unsigned(fvalue[j + node_start] >= (threshold + 1));

        unsigned int current = row2Node[j + node_start];
        // const unsigned tmp = current;
        // clear
        current &= ~(1ULL << (level));
        // set
        current |= right << level;
        row2Node[j + node_start] = current;
      }

      if (threadIdx.x == 0 && blockIdx.x % per == 0) {
        node_size_prefix_sum_next[2 * i + 1] = split_count_value;
        node_size_prefix_sum_next[2 * i + 2] = node_end;
        node_sum_prefix_sum_next[2 * i + 1] = split_sum[idx];
        node_sum_prefix_sum_next[2 * i + 2] = node_end_sum;
      }
    } else if (gain[i] == 0 && features[i] == -1 && threadIdx.x == 0 &&
               blockIdx.x % per == 0) {  // no split, all to left
      sum[i] = node_end_sum - node_start_sum;
      split[i] = (unsigned)-1;
      count[i] = node_size;
      node_size_prefix_sum_next[2 * i + 1] =
        node_size_prefix_sum_next[2 * i + 2] = node_end;
      node_sum_prefix_sum_next[2 * i + 1] =
        node_sum_prefix_sum_next[2 * i + 2] = node_end_sum;
    }
    // ignore not-optimal splits
  } else if (threadIdx.x == 0 && blockIdx.x % per == 0) {
    node_size_prefix_sum_next[2 * i + 1] =
      node_size_prefix_sum_next[2 * i + 2] = node_end;
    node_sum_prefix_sum_next[2 * i + 1] = node_sum_prefix_sum_next[2 * i + 2] =
      node_end_sum;
  }
}  // namespace core

template <typename NODE_T, typename GRAD_T, typename SUM_T>
HistTreeGrower<NODE_T, GRAD_T, SUM_T>::HistTreeGrower(
  const size_t size, const unsigned depth, const unsigned hist_size,
  const BestSplit<SUM_T> *best, Histogram<SUM_T> *features_histogram,
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

  //   cudaFuncSetCacheConfig(hist_sum_dynamic<SUM_T, GRAD_T>,
  //                          cudaFuncCachePreferShared);

  cudaFuncSetCacheConfig(hist_sum_node<SUM_T, GRAD_T>,
                         cudaFuncCachePreferShared);

  cudaFuncSetCacheConfig(hist_sum_multi_node<SUM_T, GRAD_T>,
                         cudaFuncCachePreferShared);

  size_t temp_storage_bytes = 0;

  PartitioningLeafs<NODE_T> conversion_op(0);

  cub::TransformInputIterator<bool, PartitioningLeafs<NODE_T>, NODE_T *>
    partition_itr((NODE_T *)nullptr, conversion_op);

  cub::DiscardOutputIterator<unsigned> discard_itr;

  OK(cub::DevicePartition::Flagged(NULL, temp_storage_bytes, (GRAD_T *)nullptr,
                                   partition_itr, (GRAD_T *)nullptr,
                                   discard_itr, size));

  this->temp_bytes_allocated =
    std::max(this->temp_bytes_allocated, temp_storage_bytes);

  temp_storage_bytes = 0;

  OK(cub::DevicePartition::Flagged(NULL, temp_storage_bytes, (NODE_T *)nullptr,
                                   partition_itr, (NODE_T *)nullptr,
                                   discard_itr, size));

  this->temp_bytes_allocated =
    std::max(this->temp_bytes_allocated, temp_storage_bytes);

  temp_storage_bytes = 0;

  OK(cub::DevicePartition::Flagged(NULL, temp_storage_bytes, (GRAD_T *)nullptr,
                                   partition_itr, (GRAD_T *)nullptr,
                                   discard_itr, size / (1 << this->depth)));

  this->temp_bytes_allocated = std::max(
    this->temp_bytes_allocated, temp_storage_bytes * (1 << this->depth));

  temp_storage_bytes = 0;

  OK(cub::DevicePartition::Flagged(NULL, temp_storage_bytes, (NODE_T *)nullptr,
                                   partition_itr, (NODE_T *)nullptr,
                                   discard_itr, size / (1 << this->depth)));

  this->temp_bytes_allocated = std::max(
    this->temp_bytes_allocated, temp_storage_bytes * (1 << this->depth));

  temp_storage_bytes = 0;

  cub::Sum sum_op;

  OK(cub::DeviceScan::InclusiveScan(NULL, temp_storage_bytes, (SUM_T *)nullptr,
                                    (SUM_T *)nullptr, sum_op,
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

template <typename NODE_T, typename GRAD_T, typename SUM_T>
void HistTreeGrower<NODE_T, GRAD_T, SUM_T>::HistSum(
  SUM_T *sum, unsigned *bin_count, const SUM_T *hist_sum_parent,
  const unsigned *hist_count_parent, const GRAD_T *grad,
  const unsigned *node_size, const unsigned short *fvalue,
  const unsigned hist_size_bits, const unsigned hist_size, const unsigned size,
  const bool use_trick, cudaStream_t stream) {
  unsigned min_grid_size = 256;
  int blocks_per_node = 4;

  while ((size << blocks_per_node) < min_grid_size) blocks_per_node++;

  constexpr unsigned blockSize = HIST_SUM_BLOCK_DIM;
  const unsigned gridSize = size * (1 << blocks_per_node);

  if (use_trick) {
    hist_sum_multi_node<SUM_T, GRAD_T><<<gridSize / 2, blockSize, 0, stream>>>(
      sum, bin_count, hist_sum_parent, hist_count_parent, grad, node_size,
      fvalue, hist_size, hist_size_bits, blocks_per_node, use_trick);

    const unsigned block_size = 1024;
    const unsigned grid_size =
      (hist_size * size / 2 + block_size - 1) / block_size;

    update_multi_node<SUM_T><<<grid_size, block_size, 0, stream>>>(
      sum, bin_count, hist_sum_parent, hist_count_parent, sum, bin_count,
      node_size, hist_size, hist_size * size / 2);

  } else {
    hist_sum_multi_node<SUM_T, GRAD_T><<<gridSize, blockSize, 0, stream>>>(
      sum, bin_count, hist_sum_parent, hist_count_parent, grad, node_size,
      fvalue, hist_size, hist_size_bits, blocks_per_node, use_trick);
  }
}

template <typename NODE_T, typename GRAD_T, typename SUM_T>
void HistTreeGrower<NODE_T, GRAD_T, SUM_T>::HistSumStatic(
  SUM_T *sum, unsigned *bin_count, const SUM_T *hist_sum_parent,
  const unsigned *hist_count_parent, const GRAD_T *grad,
  const unsigned *node_size, const unsigned short *fvalue,
  const unsigned hist_size_bits, const unsigned hist_size, const unsigned size,
  const bool use_trick, cudaStream_t stream) {
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
          sum + i * hist_size, bin_count + i * hist_size, grad + segment_start,
          node_size + i, fvalue + segment_start, hist_size_bits, segment_size,
          stream);
    }
  }
}

template <typename NODE_T, typename GRAD_T, typename SUM_T>
void HistTreeGrower<NODE_T, GRAD_T, SUM_T>::HistSumSingleNode(
  SUM_T *sum, unsigned *bin_count, const GRAD_T *grad,
  const unsigned *node_size, const unsigned short *fvalue,
  const unsigned hist_size_bits, const unsigned size, cudaStream_t stream) {
  constexpr unsigned blockSize = HIST_SUM_BLOCK_DIM;
  constexpr unsigned items_per_thread = ITEMS_PER_THREAD_FOR_TYPE<SUM_T>();
  const unsigned gridSize = (size + (blockSize * items_per_thread) - 1) /
                            (blockSize * items_per_thread);

  hist_sum_node<SUM_T, GRAD_T><<<gridSize, blockSize, 0, stream>>>(
    sum, bin_count, grad, fvalue, hist_size_bits, 0, size);
}

template <typename NODE_T, typename GRAD_T, typename SUM_T>
void HistTreeGrower<NODE_T, GRAD_T, SUM_T>::HistSumNaive(
  SUM_T *sum, unsigned *bin_count, const GRAD_T *grad,
  const unsigned *node_size, const unsigned short *fvalue,
  const unsigned hist_size, const unsigned size, cudaStream_t stream) {
  constexpr unsigned blockSize = 1024;
  const unsigned gridSize = (size + blockSize - 1) / blockSize;

  hist_sum<SUM_T, GRAD_T><<<gridSize, blockSize, 0, stream>>>(
    sum, bin_count, grad, node_size, fvalue, hist_size, size);
}

template <typename NODE_T, typename GRAD_T, typename SUM_T>
template <typename NODE_VALUE_T>
void HistTreeGrower<NODE_T, GRAD_T, SUM_T>::ProcessDenseFeature(
  const device_vector<unsigned> &partitioning_index,
  const device_vector<NODE_T> &row2Node, const device_vector<GRAD_T> &grad_d,
  device_vector<unsigned short> &fvalue_d, unsigned short *fvalue_h,
  const device_vector<SUM_T> &parent_node_sum,
  const device_vector<unsigned int> &parent_node_count,
  const unsigned char fvalue_size, const unsigned level, const unsigned depth,
  const GainFunctionParameters gain_param, const bool partition_only,
  const int fid) {
  const unsigned length = 1 << level;

  OK(cudaMemsetAsync(thrust::raw_pointer_cast(this->result_d.data()), 0,
                     length * sizeof(my_atomics), this->stream));
  OK(cudaMemsetAsync(thrust::raw_pointer_cast(this->sum.data()), 0,
                     this->hist_size * length * sizeof(SUM_T), this->stream));
  OK(cudaMemsetAsync(thrust::raw_pointer_cast(this->hist_bin_count.data()), 0,
                     this->hist_size * length * sizeof(unsigned),
                     this->stream));

  unsigned short *fvalue_tmp = NULL;

  if (!fvalue_d.empty()) {
    fvalue_tmp = thrust::raw_pointer_cast(fvalue_d.data());
  } else {
    OK(cudaMemcpyAsync(thrust::raw_pointer_cast(this->fvalue.data()), fvalue_h,
                       this->size * sizeof(unsigned short),
                       cudaMemcpyHostToDevice, this->stream));
    fvalue_tmp = thrust::raw_pointer_cast(this->fvalue.data());
  }

  if (level != 0) {
    this->PartitionByIndex(thrust::raw_pointer_cast(this->fvalue_dst.data()),
                           fvalue_tmp, partitioning_index);

    OK(cudaEventRecord(this->event, this->stream));

    OK(cudaStreamWaitEvent(this->copy_d2h_stream, this->event, 0));
    if (fvalue_d.empty()) {
      OK(cudaMemcpyAsync(fvalue_h,
                         thrust::raw_pointer_cast(this->fvalue_dst.data()),
                         this->size * sizeof(unsigned short),
                         cudaMemcpyDeviceToHost, this->copy_d2h_stream));
      this->d_fvalue_partitioned =
        thrust::raw_pointer_cast(this->fvalue_dst.data());
    } else {
      this->fvalue_dst.swap(fvalue_d);
      this->d_fvalue_partitioned = thrust::raw_pointer_cast(fvalue_d.data());
    }

  } else {
    this->d_fvalue_partitioned = fvalue_tmp;
  }

  if (partition_only) return;

  if (level != 0) {
    HistTreeGrower<NODE_T, GRAD_T, SUM_T>::HistSum(
      thrust::raw_pointer_cast(this->sum.data()),
      thrust::raw_pointer_cast(this->hist_bin_count.data()),
      thrust::raw_pointer_cast(this->features_histogram->grad_hist[fid].data()),
      thrust::raw_pointer_cast(
        this->features_histogram->count_hist[fid].data()),
      thrust::raw_pointer_cast(grad_d.data()),
      thrust::raw_pointer_cast(parent_node_count.data()),
      this->d_fvalue_partitioned, hist_size_bits, hist_size, length,
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
    length * this->hist_size, this->stream));

  OK(cub::DeviceScan::InclusiveSum(
    this->temp_bytes, this->temp_bytes_allocated,
    thrust::raw_pointer_cast(this->hist_bin_count.data()),
    thrust::raw_pointer_cast(this->hist_prefix_count.data()),
    length * this->hist_size, this->stream));
  int grid_size = 0;
  int block_size = 0;

  compute1DInvokeConfig(length * this->hist_size, &grid_size, &block_size,
                        hist_gain_kernel<SUM_T>, 0, 1024);

  hist_gain_kernel<SUM_T><<<grid_size, block_size, 0, this->stream>>>(
    thrust::raw_pointer_cast(this->hist_prefix_sum.data()),
    thrust::raw_pointer_cast(this->hist_prefix_count.data()),
    thrust::raw_pointer_cast(parent_node_sum.data()),
    thrust::raw_pointer_cast(parent_node_count.data()), this->hist_size,
    length * this->hist_size, gain_param,
    thrust::raw_pointer_cast(this->result_d.data()));
}

template <typename NODE_T, typename GRAD_T, typename SUM_T>
void HistTreeGrower<NODE_T, GRAD_T, SUM_T>::FindBest(
  BestSplit<SUM_T> &best, device_vector<NODE_T> &row2Node,
  const device_vector<SUM_T> &parent_node_sum,
  const device_vector<unsigned int> &parent_node_count, unsigned fid,
  const unsigned level, const unsigned depth, const unsigned size) {
  hist_apply_candidates<NODE_T, SUM_T><<<(1 << level), 1024, 0, this->stream>>>(
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
    thrust::raw_pointer_cast(parent_node_sum.data()), fid, depth - level - 2,
    this->hist_size, size);
  if (this->config->use_hist_subtraction_trick) {
    this->features_histogram->Update(this->sum, this->hist_bin_count, fid,
                                     level, this->stream);
  }
}

template class HistTreeGrower<unsigned, float, float>;

template void
HistTreeGrower<unsigned, float, float>::ProcessDenseFeature<unsigned>(
  const device_vector<unsigned> &partitioning_index,
  const device_vector<unsigned> &row2Node, const device_vector<float> &grad_d,
  device_vector<unsigned short> &fvalue_d, unsigned short *fvalue_h,
  const device_vector<float> &parent_node_sum,
  const device_vector<unsigned int> &parent_node_count,
  const unsigned char fvalue_size, const unsigned level, const unsigned depth,
  const GainFunctionParameters gain_param, const bool partition_only,
  const int fid);

template void
HistTreeGrower<unsigned, float, float>::ProcessDenseFeature<unsigned long>(
  const device_vector<unsigned> &partitioning_index,
  const device_vector<unsigned> &row2Node, const device_vector<float> &grad_d,
  device_vector<unsigned short> &fvalue_d, unsigned short *fvalue_h,
  const device_vector<float> &parent_node_sum,
  const device_vector<unsigned int> &parent_node_count,
  const unsigned char fvalue_size, const unsigned level, const unsigned depth,
  const GainFunctionParameters gain_param, const bool partition_only,
  const int fid);

template class HistTreeGrower<unsigned, float, double>;

template void
HistTreeGrower<unsigned, float, double>::ProcessDenseFeature<unsigned>(
  const device_vector<unsigned> &partitioning_index,
  const device_vector<unsigned> &row2Node, const device_vector<float> &grad_d,
  device_vector<unsigned short> &fvalue_d, unsigned short *fvalue_h,
  const device_vector<double> &parent_node_sum,
  const device_vector<unsigned int> &parent_node_count,
  const unsigned char fvalue_size, const unsigned level, const unsigned depth,
  const GainFunctionParameters gain_param, const bool partition_only,
  const int fid);

template void
HistTreeGrower<unsigned, float, double>::ProcessDenseFeature<unsigned long>(
  const device_vector<unsigned> &partitioning_index,
  const device_vector<unsigned> &row2Node, const device_vector<float> &grad_d,
  device_vector<unsigned short> &fvalue_d, unsigned short *fvalue_h,
  const device_vector<double> &parent_node_sum,
  const device_vector<unsigned int> &parent_node_count,
  const unsigned char fvalue_size, const unsigned level, const unsigned depth,
  const GainFunctionParameters gain_param, const bool partition_only,
  const int fid);

template class HistTreeGrower<unsigned, float2, float2>;

template void
HistTreeGrower<unsigned, float2, float2>::ProcessDenseFeature<unsigned>(
  const device_vector<unsigned> &partitioning_index,
  const device_vector<unsigned> &row2Node, const device_vector<float2> &grad_d,
  device_vector<unsigned short> &fvalue_d, unsigned short *fvalue_h,
  const device_vector<float2> &parent_node_sum,
  const device_vector<unsigned int> &parent_node_count,
  const unsigned char fvalue_size, const unsigned level, const unsigned depth,
  const GainFunctionParameters gain_param, const bool partition_only,
  const int fid);

template void
HistTreeGrower<unsigned, float2, float2>::ProcessDenseFeature<unsigned long>(
  const device_vector<unsigned> &partitioning_index,
  const device_vector<unsigned> &row2Node, const device_vector<float2> &grad_d,
  device_vector<unsigned short> &fvalue_d, unsigned short *fvalue_h,
  const device_vector<float2> &parent_node_sum,
  const device_vector<unsigned int> &parent_node_count,
  const unsigned char fvalue_size, const unsigned level, const unsigned depth,
  const GainFunctionParameters gain_param, const bool partition_only,
  const int fid);

template class HistTreeGrower<unsigned, float2, mydouble2>;

template void
HistTreeGrower<unsigned, float2, mydouble2>::ProcessDenseFeature<unsigned>(
  const device_vector<unsigned> &partitioning_index,
  const device_vector<unsigned> &row2Node, const device_vector<float2> &grad_d,
  device_vector<unsigned short> &fvalue_d, unsigned short *fvalue_h,
  const device_vector<mydouble2> &parent_node_sum,
  const device_vector<unsigned int> &parent_node_count,
  const unsigned char fvalue_size, const unsigned level, const unsigned depth,
  const GainFunctionParameters gain_param, const bool partition_only,
  const int fid);

template void
HistTreeGrower<unsigned, float2, mydouble2>::ProcessDenseFeature<unsigned long>(
  const device_vector<unsigned> &partitioning_index,
  const device_vector<unsigned> &row2Node, const device_vector<float2> &grad_d,
  device_vector<unsigned short> &fvalue_d, unsigned short *fvalue_h,
  const device_vector<mydouble2> &parent_node_sum,
  const device_vector<unsigned int> &parent_node_count,
  const unsigned char fvalue_size, const unsigned level, const unsigned depth,
  const GainFunctionParameters gain_param, const bool partition_only,
  const int fid);

template class HistTreeGrower<unsigned short, float, float>;
template class HistTreeGrower<unsigned short, float, double>;
template class HistTreeGrower<unsigned short, float2, float2>;
template class HistTreeGrower<unsigned short, float2, mydouble2>;

template class HistTreeGrower<unsigned char, float, float>;
template class HistTreeGrower<unsigned char, float, double>;
template class HistTreeGrower<unsigned char, float2, float2>;
template class HistTreeGrower<unsigned char, float2, mydouble2>;

template class HistTreeGrower<unsigned long, float, float>;

template void
HistTreeGrower<unsigned long, float, float>::ProcessDenseFeature<unsigned>(
  const device_vector<unsigned> &partitioning_index,
  const device_vector<unsigned long> &row2Node,
  const device_vector<float> &grad_d, device_vector<unsigned short> &fvalue_d,
  unsigned short *fvalue_h, const device_vector<float> &parent_node_sum,
  const device_vector<unsigned int> &parent_node_count,
  const unsigned char fvalue_size, const unsigned level, const unsigned depth,
  const GainFunctionParameters gain_param, const bool partition_only,
  const int fid);

template void
HistTreeGrower<unsigned long, float, float>::ProcessDenseFeature<unsigned long>(
  const device_vector<unsigned> &partitioning_index,
  const device_vector<unsigned long> &row2Node,
  const device_vector<float> &grad_d, device_vector<unsigned short> &fvalue_d,
  unsigned short *fvalue_h, const device_vector<float> &parent_node_sum,
  const device_vector<unsigned int> &parent_node_count,
  const unsigned char fvalue_size, const unsigned level, const unsigned depth,
  const GainFunctionParameters gain_param, const bool partition_only,
  const int fid);

template class HistTreeGrower<unsigned long, float, double>;

template void
HistTreeGrower<unsigned long, float, double>::ProcessDenseFeature<unsigned>(
  const device_vector<unsigned> &partitioning_index,
  const device_vector<unsigned long> &row2Node,
  const device_vector<float> &grad_d, device_vector<unsigned short> &fvalue_d,
  unsigned short *fvalue_h, const device_vector<double> &parent_node_sum,
  const device_vector<unsigned int> &parent_node_count,
  const unsigned char fvalue_size, const unsigned level, const unsigned depth,
  const GainFunctionParameters gain_param, const bool partition_only,
  const int fid);

template void HistTreeGrower<unsigned long, float, double>::ProcessDenseFeature<
  unsigned long>(const device_vector<unsigned> &partitioning_index,
                 const device_vector<unsigned long> &row2Node,
                 const device_vector<float> &grad_d,
                 device_vector<unsigned short> &fvalue_d,
                 unsigned short *fvalue_h,
                 const device_vector<double> &parent_node_sum,
                 const device_vector<unsigned int> &parent_node_count,
                 const unsigned char fvalue_size, const unsigned level,
                 const unsigned depth, const GainFunctionParameters gain_param,
                 const bool partition_only, const int fid);

template class HistTreeGrower<unsigned long, float2, float2>;

template void
HistTreeGrower<unsigned long, float2, float2>::ProcessDenseFeature<unsigned>(
  const device_vector<unsigned> &partitioning_index,
  const device_vector<unsigned long> &row2Node,
  const device_vector<float2> &grad_d, device_vector<unsigned short> &fvalue_d,
  unsigned short *fvalue_h, const device_vector<float2> &parent_node_sum,
  const device_vector<unsigned int> &parent_node_count,
  const unsigned char fvalue_size, const unsigned level, const unsigned depth,
  const GainFunctionParameters gain_param, const bool partition_only,
  const int fid);

template void
HistTreeGrower<unsigned long, float2, float2>::ProcessDenseFeature<
  unsigned long>(const device_vector<unsigned> &partitioning_index,
                 const device_vector<unsigned long> &row2Node,
                 const device_vector<float2> &grad_d,
                 device_vector<unsigned short> &fvalue_d,
                 unsigned short *fvalue_h,
                 const device_vector<float2> &parent_node_sum,
                 const device_vector<unsigned int> &parent_node_count,
                 const unsigned char fvalue_size, const unsigned level,
                 const unsigned depth, const GainFunctionParameters gain_param,
                 const bool partition_only, const int fid);

template class HistTreeGrower<unsigned long, float2, mydouble2>;

template void
HistTreeGrower<unsigned long, float2, mydouble2>::ProcessDenseFeature<unsigned>(
  const device_vector<unsigned> &partitioning_index,
  const device_vector<unsigned long> &row2Node,
  const device_vector<float2> &grad_d, device_vector<unsigned short> &fvalue_d,
  unsigned short *fvalue_h, const device_vector<mydouble2> &parent_node_sum,
  const device_vector<unsigned int> &parent_node_count,
  const unsigned char fvalue_size, const unsigned level, const unsigned depth,
  const GainFunctionParameters gain_param, const bool partition_only,
  const int fid);

template void
HistTreeGrower<unsigned long, float2, mydouble2>::ProcessDenseFeature<
  unsigned long>(const device_vector<unsigned> &partitioning_index,
                 const device_vector<unsigned long> &row2Node,
                 const device_vector<float2> &grad_d,
                 device_vector<unsigned short> &fvalue_d,
                 unsigned short *fvalue_h,
                 const device_vector<mydouble2> &parent_node_sum,
                 const device_vector<unsigned int> &parent_node_count,
                 const unsigned char fvalue_size, const unsigned level,
                 const unsigned depth, const GainFunctionParameters gain_param,
                 const bool partition_only, const int fid);

template class HistTreeGrower<unsigned long long, float, float>;

template void
HistTreeGrower<unsigned long long, float, float>::ProcessDenseFeature<unsigned>(
  const device_vector<unsigned> &partitioning_index,
  const device_vector<unsigned long long> &row2Node,
  const device_vector<float> &grad_d, device_vector<unsigned short> &fvalue_d,
  unsigned short *fvalue_h, const device_vector<float> &parent_node_sum,
  const device_vector<unsigned int> &parent_node_count,
  const unsigned char fvalue_size, const unsigned level, const unsigned depth,
  const GainFunctionParameters gain_param, const bool partition_only,
  const int fid);

template void HistTreeGrower<unsigned long long, float, float>::
  ProcessDenseFeature<unsigned long>(
    const device_vector<unsigned> &partitioning_index,
    const device_vector<unsigned long long> &row2Node,
    const device_vector<float> &grad_d, device_vector<unsigned short> &fvalue_d,
    unsigned short *fvalue_h, const device_vector<float> &parent_node_sum,
    const device_vector<unsigned int> &parent_node_count,
    const unsigned char fvalue_size, const unsigned level, const unsigned depth,
    const GainFunctionParameters gain_param, const bool partition_only,
    const int fid);

template void HistTreeGrower<unsigned long long, float, float>::
  ProcessDenseFeature<unsigned long long>(
    const device_vector<unsigned> &partitioning_index,
    const device_vector<unsigned long long> &row2Node,
    const device_vector<float> &grad_d, device_vector<unsigned short> &fvalue_d,
    unsigned short *fvalue_h, const device_vector<float> &parent_node_sum,
    const device_vector<unsigned int> &parent_node_count,
    const unsigned char fvalue_size, const unsigned level, const unsigned depth,
    const GainFunctionParameters gain_param, const bool partition_only,
    const int fid);

template class HistTreeGrower<unsigned long long, float, double>;

template void
HistTreeGrower<unsigned long long, float, double>::ProcessDenseFeature<
  unsigned>(const device_vector<unsigned> &partitioning_index,
            const device_vector<unsigned long long> &row2Node,
            const device_vector<float> &grad_d,
            device_vector<unsigned short> &fvalue_d, unsigned short *fvalue_h,
            const device_vector<double> &parent_node_sum,
            const device_vector<unsigned int> &parent_node_count,
            const unsigned char fvalue_size, const unsigned level,
            const unsigned depth, const GainFunctionParameters gain_param,
            const bool partition_only, const int fid);

template void HistTreeGrower<unsigned long long, float, double>::
  ProcessDenseFeature<unsigned long>(
    const device_vector<unsigned> &partitioning_index,
    const device_vector<unsigned long long> &row2Node,
    const device_vector<float> &grad_d, device_vector<unsigned short> &fvalue_d,
    unsigned short *fvalue_h, const device_vector<double> &parent_node_sum,
    const device_vector<unsigned int> &parent_node_count,
    const unsigned char fvalue_size, const unsigned level, const unsigned depth,
    const GainFunctionParameters gain_param, const bool partition_only,
    const int fid);

template void HistTreeGrower<unsigned long long, float, double>::
  ProcessDenseFeature<unsigned long long>(
    const device_vector<unsigned> &partitioning_index,
    const device_vector<unsigned long long> &row2Node,
    const device_vector<float> &grad_d, device_vector<unsigned short> &fvalue_d,
    unsigned short *fvalue_h, const device_vector<double> &parent_node_sum,
    const device_vector<unsigned int> &parent_node_count,
    const unsigned char fvalue_size, const unsigned level, const unsigned depth,
    const GainFunctionParameters gain_param, const bool partition_only,
    const int fid);

template class HistTreeGrower<unsigned long long, float2, float2>;

template void
HistTreeGrower<unsigned long long, float2, float2>::ProcessDenseFeature<
  unsigned>(const device_vector<unsigned> &partitioning_index,
            const device_vector<unsigned long long> &row2Node,
            const device_vector<float2> &grad_d,
            device_vector<unsigned short> &fvalue_d, unsigned short *fvalue_h,
            const device_vector<float2> &parent_node_sum,
            const device_vector<unsigned int> &parent_node_count,
            const unsigned char fvalue_size, const unsigned level,
            const unsigned depth, const GainFunctionParameters gain_param,
            const bool partition_only, const int fid);

template void
HistTreeGrower<unsigned long long, float2, float2>::ProcessDenseFeature<
  unsigned long>(const device_vector<unsigned> &partitioning_index,
                 const device_vector<unsigned long long> &row2Node,
                 const device_vector<float2> &grad_d,
                 device_vector<unsigned short> &fvalue_d,
                 unsigned short *fvalue_h,
                 const device_vector<float2> &parent_node_sum,
                 const device_vector<unsigned int> &parent_node_count,
                 const unsigned char fvalue_size, const unsigned level,
                 const unsigned depth, const GainFunctionParameters gain_param,
                 const bool partition_only, const int fid);

template void HistTreeGrower<unsigned long long, float2, float2>::
  ProcessDenseFeature<unsigned long long>(
    const device_vector<unsigned> &partitioning_index,
    const device_vector<unsigned long long> &row2Node,
    const device_vector<float2> &grad_d,
    device_vector<unsigned short> &fvalue_d, unsigned short *fvalue_h,
    const device_vector<float2> &parent_node_sum,
    const device_vector<unsigned int> &parent_node_count,
    const unsigned char fvalue_size, const unsigned level, const unsigned depth,
    const GainFunctionParameters gain_param, const bool partition_only,
    const int fid);

template class HistTreeGrower<unsigned long long, float2, mydouble2>;

template void
HistTreeGrower<unsigned long long, float2, mydouble2>::ProcessDenseFeature<
  unsigned>(const device_vector<unsigned> &partitioning_index,
            const device_vector<unsigned long long> &row2Node,
            const device_vector<float2> &grad_d,
            device_vector<unsigned short> &fvalue_d, unsigned short *fvalue_h,
            const device_vector<mydouble2> &parent_node_sum,
            const device_vector<unsigned int> &parent_node_count,
            const unsigned char fvalue_size, const unsigned level,
            const unsigned depth, const GainFunctionParameters gain_param,
            const bool partition_only, const int fid);

template void
HistTreeGrower<unsigned long long, float2, mydouble2>::ProcessDenseFeature<
  unsigned long>(const device_vector<unsigned> &partitioning_index,
                 const device_vector<unsigned long long> &row2Node,
                 const device_vector<float2> &grad_d,
                 device_vector<unsigned short> &fvalue_d,
                 unsigned short *fvalue_h,
                 const device_vector<mydouble2> &parent_node_sum,
                 const device_vector<unsigned int> &parent_node_count,
                 const unsigned char fvalue_size, const unsigned level,
                 const unsigned depth, const GainFunctionParameters gain_param,
                 const bool partition_only, const int fid);

template void HistTreeGrower<unsigned long long, float2, mydouble2>::
  ProcessDenseFeature<unsigned long long>(
    const device_vector<unsigned> &partitioning_index,
    const device_vector<unsigned long long> &row2Node,
    const device_vector<float2> &grad_d,
    device_vector<unsigned short> &fvalue_d, unsigned short *fvalue_h,
    const device_vector<mydouble2> &parent_node_sum,
    const device_vector<unsigned int> &parent_node_count,
    const unsigned char fvalue_size, const unsigned level, const unsigned depth,
    const GainFunctionParameters gain_param, const bool partition_only,
    const int fid);

}  // namespace core
}  // namespace arboretum
