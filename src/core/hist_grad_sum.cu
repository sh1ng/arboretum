#include "cub/cub.cuh"
#include "cuda_helpers.h"
#include "cuda_runtime.h"
#include "hist_grad_sum.cuh"

template <typename SUM_T>
__global__ void update_multi_node(
  SUM_T *sum_dst, unsigned *count_dst, const SUM_T *hist_sum_parent,
  const unsigned *hist_count_parent_count, const SUM_T *sum_src,
  const unsigned *count_src, const unsigned *__restrict__ parent_count_iter,
  const unsigned hist_size, const unsigned n) {
  for (unsigned int i = blockDim.x * blockIdx.x + threadIdx.x; i < n;
       i += gridDim.x * blockDim.x) {
    int parent_node_id = i / hist_size;
    int position = i % hist_size;
    int left_node_id = parent_node_id * 2;
    int right_node_id = parent_node_id * 2 + 1;
    unsigned left_size =
      parent_count_iter[left_node_id + 1] - parent_count_iter[left_node_id];
    unsigned right_size =
      parent_count_iter[right_node_id + 1] - parent_count_iter[right_node_id];

    int src_node = left_size <= right_size ? left_node_id : right_node_id;
    int affected_node = src_node == left_node_id ? right_node_id : left_node_id;

    sum_dst[affected_node * hist_size + position] =
      hist_sum_parent[parent_node_id * hist_size + position] -
      sum_src[src_node * hist_size + position];
    count_dst[affected_node * hist_size + position] =
      hist_count_parent_count[parent_node_id * hist_size + position] -
      count_src[src_node * hist_size + position];
  }
}

template __global__ void update_multi_node<float>(
  float *sum_dst, unsigned *count_dst, const float *hist_sum_parent,
  const unsigned *hist_count_parent_count, const float *sum_src,
  const unsigned *count_src, const unsigned *__restrict__ parent_count_iter,
  const unsigned hist_size, const unsigned n);

template __global__ void update_multi_node<double>(
  double *sum_dst, unsigned *count_dst, const double *hist_sum_parent,
  const unsigned *hist_count_parent_count, const double *sum_src,
  const unsigned *count_src, const unsigned *__restrict__ parent_count_iter,
  const unsigned hist_size, const unsigned n);

template __global__ void update_multi_node<float2>(
  float2 *sum_dst, unsigned *count_dst, const float2 *hist_sum_parent,
  const unsigned *hist_count_parent_count, const float2 *sum_src,
  const unsigned *count_src, const unsigned *__restrict__ parent_count_iter,
  const unsigned hist_size, const unsigned n);

template __global__ void update_multi_node<mydouble2>(
  mydouble2 *sum_dst, unsigned *count_dst, const mydouble2 *hist_sum_parent,
  const unsigned *hist_count_parent_count, const mydouble2 *sum_src,
  const unsigned *count_src, const unsigned *__restrict__ parent_count_iter,
  const unsigned hist_size, const unsigned n);

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

template <typename SUM_T, typename GRAD_T, int ITEMS_PER_THREAD>
__global__ void hist_sum_multi_node(
  SUM_T *dst_sum, unsigned *dst_count, const SUM_T *hist_sum_parent,
  const unsigned *hist_count_parent, const GRAD_T *__restrict__ values,
  const unsigned *__restrict__ parent_count_iter,
  const unsigned *__restrict__ bin, const unsigned hist_size,
  const unsigned end_bit, const int blocks_per_node, const bool use_trick) {
  //
  int node_id = blockIdx.x >> blocks_per_node;

  if (use_trick) {
    int node_left = node_id * 2;
    int node_right = node_id * 2 + 1;
    const unsigned left_size =
      parent_count_iter[node_left + 1] - parent_count_iter[node_left];
    const unsigned right_size =
      parent_count_iter[node_right + 1] - parent_count_iter[node_right];

    node_id = left_size > right_size ? node_right : node_left;
  }

  const unsigned node_start = parent_count_iter[node_id];
  const unsigned node_size =
    parent_count_iter[node_id + 1] - parent_count_iter[node_id];

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

  for (int block_offset =
         blockIdx.x - (blockIdx.x >> blocks_per_node) * (1 << blocks_per_node);
       block_offset * ITEMS_PER_THREAD * HIST_SUM_BLOCK_DIM < node_size;
       block_offset += (1 << blocks_per_node)) {
#pragma unroll
    for (unsigned i = 0; i < ITEMS_PER_THREAD; ++i) {
      unsigned idx = block_offset * ITEMS_PER_THREAD * HIST_SUM_BLOCK_DIM +
                     i * HIST_SUM_BLOCK_DIM + threadIdx.x;

      if (idx < node_size) {
        thread_keys[i] = bin[idx + node_start];
        thread_values[i] = values[idx + node_start];
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
        atomicAdd(&dst_sum[key + node_id * hist_size], sum_current);
        atomicAdd(&dst_count[key + node_id * hist_size], count_current);
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
      __device__ __forceinline__
        cub::KeyValuePair<unsigned short, unsigned short>
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
      atomicAdd(&dst_sum[segment_start.value + node_id * hist_size],
                sum - temp_storage.total.sum[segment_start.key]);
      atomicAdd(&dst_count[segment_start.value + node_id * hist_size],
                count - temp_storage.total.count[segment_start.key]);
    }
    // last thread also need to handle it's own sum
    if (lane == 31 && thread_keys[ITEMS_PER_THREAD - 1] != HIST_SUM_NO_DATA) {
      // flush all collected data
      if (thread_keys[ITEMS_PER_THREAD - 1] == segment_start.value) {
        atomicAdd(
          &dst_sum[thread_keys[ITEMS_PER_THREAD - 1] + node_id * hist_size],
          sum_current + sum - temp_storage.total.sum[segment_start.key]);
        atomicAdd(
          &dst_count[thread_keys[ITEMS_PER_THREAD - 1] + node_id * hist_size],
          count_current + count - temp_storage.total.count[segment_start.key]);

      } else {  // only thread local sum
        atomicAdd(
          &dst_sum[thread_keys[ITEMS_PER_THREAD - 1] + node_id * hist_size],
          sum_current);
        atomicAdd(
          &dst_count[thread_keys[ITEMS_PER_THREAD - 1] + node_id * hist_size],
          count_current);
      }
    }
    __syncthreads();
  }
}

template __global__ void hist_sum_multi_node<float, float>(
  float *dst_sum, unsigned *dst_count, const float *hist_sum_parent,
  const unsigned *hist_count_parent, const float *__restrict__ values,
  const unsigned *__restrict__ parent_count_iter,
  const unsigned *__restrict__ bin, const unsigned hist_size,
  const unsigned end_bit, const int blocks_per_node, const bool use_trick);

template __global__ void hist_sum_multi_node<float2, float2>(
  float2 *dst_sum, unsigned *dst_count, const float2 *hist_sum_parent,
  const unsigned *hist_count_parent, const float2 *__restrict__ values,
  const unsigned *__restrict__ parent_count_iter,
  const unsigned *__restrict__ bin, const unsigned hist_size,
  const unsigned end_bit, const int blocks_per_node, const bool use_trick);

template __global__ void hist_sum_multi_node<double, float>(
  double *dst_sum, unsigned *dst_count, const double *hist_sum_parent,
  const unsigned *hist_count_parent, const float *__restrict__ values,
  const unsigned *__restrict__ parent_count_iter,
  const unsigned *__restrict__ bin, const unsigned hist_size,
  const unsigned end_bit, const int blocks_per_node, const bool use_trick);

template __global__ void hist_sum_multi_node<mydouble2, float2>(
  mydouble2 *dst_sum, unsigned *dst_count, const mydouble2 *hist_sum_parent,
  const unsigned *hist_count_parent, const float2 *__restrict__ values,
  const unsigned *__restrict__ parent_count_iter,
  const unsigned *__restrict__ bin, const unsigned hist_size,
  const unsigned end_bit, const int blocks_per_node, const bool use_trick);

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
