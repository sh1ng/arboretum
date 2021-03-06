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

/*[[[cog
import cog
sum_types = ['float', 'double', 'float2', 'mydouble2']
cog.outl("// clang-format off")
for t in sum_types:
    cog.outl("""template __global__ void update_multi_node<{0}>(
  {0} *sum_dst, unsigned *count_dst, const {0} *hist_sum_parent,
  const unsigned *hist_count_parent_count, const {0} *sum_src,
  const unsigned *count_src, const unsigned *__restrict__ parent_count_iter,
  const unsigned hist_size, const unsigned n);""".format(t))
cog.outl("// clang-format on")
]]]*/
// clang-format off
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
// clang-format on
//[[[end]]] (checksum: 885ef776f06fb9f79c9d2ee3dd93ff40)

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

/*[[[cog
import cog
sum_types = ['float', 'double', 'float2', 'mydouble2']
cog.outl("// clang-format off")
for t in sum_types:
    cog.outl("""template __global__ void update<{0}>({0} *sum_dst, unsigned
*count_dst, const {0} *parent_sum, const unsigned *parent_count, const {0}
*sum_src, const unsigned *count_src, const unsigned n);""".format(t))
cog.outl("// clang-format on")
]]]*/
// clang-format off
template __global__ void update<float>(float *sum_dst, unsigned
*count_dst, const float *parent_sum, const unsigned *parent_count, const float
*sum_src, const unsigned *count_src, const unsigned n);
template __global__ void update<double>(double *sum_dst, unsigned
*count_dst, const double *parent_sum, const unsigned *parent_count, const double
*sum_src, const unsigned *count_src, const unsigned n);
template __global__ void update<float2>(float2 *sum_dst, unsigned
*count_dst, const float2 *parent_sum, const unsigned *parent_count, const float2
*sum_src, const unsigned *count_src, const unsigned n);
template __global__ void update<mydouble2>(mydouble2 *sum_dst, unsigned
*count_dst, const mydouble2 *parent_sum, const unsigned *parent_count, const mydouble2
*sum_src, const unsigned *count_src, const unsigned n);
// clang-format on
//[[[end]]] (checksum: 3a2af9afc26c66fcda0fa6541b4a2370)

template <typename SUM_T, typename GRAD_T, typename BIN_TYPE,
          int ITEMS_PER_THREAD>
__global__ void hist_sum_node(SUM_T *dst_sum, unsigned *dst_count,
                              const GRAD_T *__restrict__ values,
                              const BIN_TYPE *__restrict__ bin,
                              const unsigned end_bit, const unsigned segment,
                              const size_t n) {
  constexpr BIN_TYPE NO_DATA = BIN_TYPE(-1);
  const int warp_id = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  typedef cub::BlockRadixSort<BIN_TYPE, HIST_SUM_BLOCK_DIM, ITEMS_PER_THREAD,
                              GRAD_T, 4, false, cub ::BLOCK_SCAN_RAKING>
    BlockRadixSort;

  typedef cub::WarpScan<
    cub::KeyValuePair<BIN_TYPE, cub::KeyValuePair<unsigned short, SUM_T>>>
    WarpSum;

  __shared__ typename WarpSum::TempStorage temp_scan[HIST_SUM_BLOCK_DIM / 32];
  __shared__ typename BlockRadixSort::TempStorage temp_sort;

  // Obtain a segment of consecutive items that are blocked across threads
  BIN_TYPE thread_keys[ITEMS_PER_THREAD];
  GRAD_T thread_values[ITEMS_PER_THREAD];

#pragma unroll
  for (unsigned i = 0; i < ITEMS_PER_THREAD; ++i) {
    unsigned idx =
      blockDim.x * blockIdx.x * ITEMS_PER_THREAD + i * blockDim.x + threadIdx.x;
    if (idx < n) {
      thread_keys[i] = bin[idx];
      thread_values[i] = values[idx];
    } else {
      thread_keys[i] = NO_DATA;
    }
  }

  // Collectively sort the keys and values among block threads
  BlockRadixSort(temp_sort).Sort(thread_keys, thread_values, 0, end_bit);

  SUM_T sum_current = thread_values[0];
  unsigned short count_current = 1;

#pragma unroll
  for (unsigned i = 1; i < ITEMS_PER_THREAD; ++i) {
    if (thread_keys[i - 1] == thread_keys[i]) {
      sum_current += thread_values[i];
      count_current++;
    } else {
      atomicAdd(&dst_sum[thread_keys[i - 1]], sum_current);
      atomicAdd(&dst_count[thread_keys[i - 1]], count_current);
      sum_current = thread_values[i];
      count_current = 1;
    }
  }

  SUM_T zero;
  init(zero);

  struct SegmentSum {
    __device__ __forceinline__
      cub::KeyValuePair<BIN_TYPE, cub::KeyValuePair<unsigned short, SUM_T>>
      operator()(
        const cub::KeyValuePair<BIN_TYPE,
                                cub::KeyValuePair<unsigned short, SUM_T>> &a,
        const cub::KeyValuePair<
          BIN_TYPE, cub::KeyValuePair<unsigned short, SUM_T>> &b) const {
      if (b.key > a.key) return b;
      cub::KeyValuePair<unsigned short, SUM_T> sum(
        a.value.key + b.value.key, a.value.value + b.value.value);
      cub::KeyValuePair<BIN_TYPE, cub::KeyValuePair<unsigned short, SUM_T>> v(
        a.key, sum);
      return v;
    }
  };

  cub::KeyValuePair<unsigned short, SUM_T> initial_sum(count_current,
                                                       sum_current);
  cub::KeyValuePair<BIN_TYPE, cub::KeyValuePair<unsigned short, SUM_T>> initial(
    thread_keys[ITEMS_PER_THREAD - 1], initial_sum);

  cub::KeyValuePair<unsigned short, SUM_T> zero_sum_(0, zero);
  cub::KeyValuePair<BIN_TYPE, cub::KeyValuePair<unsigned short, SUM_T>> zero_(
    0, zero_sum_);

  WarpSum(temp_scan[warp_id])
    .ExclusiveScan(initial, initial, zero_, SegmentSum());

  // flush previous segment
  if (thread_keys[ITEMS_PER_THREAD - 1] != initial.key) {
    atomicAdd(&dst_sum[initial.key], initial.value.value);
    atomicAdd(&dst_count[initial.key], initial.value.key);
  }
  //   last thread also need to handle it's own sum
  if (lane == 31 && thread_keys[ITEMS_PER_THREAD - 1] != NO_DATA) {
    // flush all collected data
    if (thread_keys[ITEMS_PER_THREAD - 1] == initial.key) {
      atomicAdd(&dst_sum[thread_keys[ITEMS_PER_THREAD - 1]],
                sum_current + initial.value.value);
      atomicAdd(&dst_count[thread_keys[ITEMS_PER_THREAD - 1]],
                count_current + initial.value.key);
    } else {  // only thread local sum
      atomicAdd(&dst_sum[thread_keys[ITEMS_PER_THREAD - 1]], sum_current);
      atomicAdd(&dst_count[thread_keys[ITEMS_PER_THREAD - 1]], count_current);
    }
  }
}
// clang-format off
/*[[[cog
import cog
for t in [('float', 'float'), ('float', 'double'), ('float2', 'float2'), ('float2', 'mydouble2')]:
    for bin_type in ['unsigned short', 'unsigned char']:
        cog.outl("""template __global__ void hist_sum_node<{0}, {1}, {2}>(
  {0} *dst_sum, unsigned *dst_count, const {1} *__restrict__ values,
  const {2} *__restrict__ bin, const unsigned end_bit,
  const unsigned segment, const size_t n);""".format(
      t[1], t[0], bin_type))
]]]*/
template __global__ void hist_sum_node<float, float, unsigned short>(
  float *dst_sum, unsigned *dst_count, const float *__restrict__ values,
  const unsigned short *__restrict__ bin, const unsigned end_bit,
  const unsigned segment, const size_t n);
template __global__ void hist_sum_node<float, float, unsigned char>(
  float *dst_sum, unsigned *dst_count, const float *__restrict__ values,
  const unsigned char *__restrict__ bin, const unsigned end_bit,
  const unsigned segment, const size_t n);
template __global__ void hist_sum_node<double, float, unsigned short>(
  double *dst_sum, unsigned *dst_count, const float *__restrict__ values,
  const unsigned short *__restrict__ bin, const unsigned end_bit,
  const unsigned segment, const size_t n);
template __global__ void hist_sum_node<double, float, unsigned char>(
  double *dst_sum, unsigned *dst_count, const float *__restrict__ values,
  const unsigned char *__restrict__ bin, const unsigned end_bit,
  const unsigned segment, const size_t n);
template __global__ void hist_sum_node<float2, float2, unsigned short>(
  float2 *dst_sum, unsigned *dst_count, const float2 *__restrict__ values,
  const unsigned short *__restrict__ bin, const unsigned end_bit,
  const unsigned segment, const size_t n);
template __global__ void hist_sum_node<float2, float2, unsigned char>(
  float2 *dst_sum, unsigned *dst_count, const float2 *__restrict__ values,
  const unsigned char *__restrict__ bin, const unsigned end_bit,
  const unsigned segment, const size_t n);
template __global__ void hist_sum_node<mydouble2, float2, unsigned short>(
  mydouble2 *dst_sum, unsigned *dst_count, const float2 *__restrict__ values,
  const unsigned short *__restrict__ bin, const unsigned end_bit,
  const unsigned segment, const size_t n);
template __global__ void hist_sum_node<mydouble2, float2, unsigned char>(
  mydouble2 *dst_sum, unsigned *dst_count, const float2 *__restrict__ values,
  const unsigned char *__restrict__ bin, const unsigned end_bit,
  const unsigned segment, const size_t n);
//[[[end]]] (checksum: 6858952bec46e367f32f00e13445c7b9)
// clang-format on

template <typename SUM_T, typename GRAD_T, typename BIN_TYPE, bool USE_TRICK,
          int ITEMS_PER_THREAD>
__global__ void hist_sum_multi_node(
  SUM_T *dst_sum, unsigned *dst_count, const SUM_T *hist_sum_parent,
  const unsigned *hist_count_parent, const GRAD_T *__restrict__ values,
  const unsigned *__restrict__ parent_count_iter,
  const BIN_TYPE *__restrict__ bin, const unsigned hist_size,
  const unsigned end_bit, const int blocks_per_node) {
  constexpr BIN_TYPE NO_DATA = BIN_TYPE(-1);
  const int warp_id = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int blocks_per_node_size = (1 << blocks_per_node);

  int node_id = blockIdx.x >> blocks_per_node;
  unsigned node_start;
  unsigned node_size;

  if (USE_TRICK) {
    int node_left = node_id * 2;
    int node_right = node_id * 2 + 1;
    const unsigned x = parent_count_iter[node_left];
    const unsigned y = parent_count_iter[node_right];
    const unsigned z = parent_count_iter[node_right + 1];
    const unsigned left_size = y - x;
    const unsigned right_size = z - y;

    node_id = left_size > right_size ? node_right : node_left;
    node_start = left_size > right_size ? y : x;
    node_size = left_size > right_size ? right_size : left_size;
  } else {
    node_start = parent_count_iter[node_id];
    node_size = parent_count_iter[node_id + 1] - node_start;
  }

  typedef cub::BlockRadixSort<BIN_TYPE, HIST_SUM_BLOCK_DIM, ITEMS_PER_THREAD,
                              GRAD_T, 4, false, cub ::BLOCK_SCAN_RAKING>
    BlockRadixSort;

  typedef cub::WarpScan<
    cub::KeyValuePair<BIN_TYPE, cub::KeyValuePair<unsigned short, SUM_T>>>
    WarpSum;

  __shared__ typename WarpSum::TempStorage temp_scan[HIST_SUM_BLOCK_DIM / 32];
  __shared__ typename BlockRadixSort::TempStorage temp_sort;

  // Obtain a segment of consecutive items that are blocked across threads
  BIN_TYPE thread_keys[ITEMS_PER_THREAD];
  GRAD_T thread_values[ITEMS_PER_THREAD];

  for (int block_offset =
         blockIdx.x - (blockIdx.x >> blocks_per_node) * blocks_per_node_size;
       block_offset * ITEMS_PER_THREAD * HIST_SUM_BLOCK_DIM < node_size;
       block_offset += blocks_per_node_size) {
#pragma unroll
    for (unsigned i = 0; i < ITEMS_PER_THREAD; ++i) {
      unsigned idx = block_offset * ITEMS_PER_THREAD * HIST_SUM_BLOCK_DIM +
                     i * HIST_SUM_BLOCK_DIM + threadIdx.x;
      if (idx < node_size) {
        thread_keys[i] = bin[idx + node_start];
        thread_values[i] = values[idx + node_start];
      } else {
        thread_keys[i] = NO_DATA;
      }
    }

    // Collectively sort the keys and values among block threads
    BlockRadixSort(temp_sort).Sort(thread_keys, thread_values, 0, end_bit);
    SUM_T sum_current = thread_values[0];
    unsigned short count_current = 1;

#pragma unroll
    for (unsigned i = 1; i < ITEMS_PER_THREAD; ++i) {
      if (thread_keys[i - 1] == thread_keys[i]) {
        sum_current += thread_values[i];
        count_current++;
      } else {
        atomicAdd(&dst_sum[thread_keys[i - 1] + node_id * hist_size],
                  sum_current);
        atomicAdd(&dst_count[thread_keys[i - 1] + node_id * hist_size],
                  count_current);
        sum_current = thread_values[i];
        count_current = 1;
      }
    }

    SUM_T zero;
    init(zero);

    struct SegmentSum {
      __device__ __forceinline__
        cub::KeyValuePair<BIN_TYPE, cub::KeyValuePair<unsigned short, SUM_T>>
        operator()(
          const cub::KeyValuePair<BIN_TYPE,
                                  cub::KeyValuePair<unsigned short, SUM_T>> &a,
          const cub::KeyValuePair<
            BIN_TYPE, cub::KeyValuePair<unsigned short, SUM_T>> &b) const {
        if (b.key > a.key) return b;
        cub::KeyValuePair<unsigned short, SUM_T> sum(
          a.value.key + b.value.key, a.value.value + b.value.value);
        cub::KeyValuePair<BIN_TYPE, cub::KeyValuePair<unsigned short, SUM_T>> v(
          a.key, sum);
        return v;
      }
    };

    cub::KeyValuePair<unsigned short, SUM_T> initial_sum(count_current,
                                                         sum_current);
    cub::KeyValuePair<BIN_TYPE, cub::KeyValuePair<unsigned short, SUM_T>>
      initial(thread_keys[ITEMS_PER_THREAD - 1], initial_sum);

    cub::KeyValuePair<unsigned short, SUM_T> zero_sum_(0, zero);
    cub::KeyValuePair<BIN_TYPE, cub::KeyValuePair<unsigned short, SUM_T>> zero_(
      0, zero_sum_);

    WarpSum(temp_scan[warp_id])
      .ExclusiveScan(initial, initial, zero_, SegmentSum());

    // flush previous segment
    if (thread_keys[ITEMS_PER_THREAD - 1] != initial.key) {
      atomicAdd(&dst_sum[initial.key + node_id * hist_size],
                initial.value.value);
      atomicAdd(&dst_count[initial.key + node_id * hist_size],
                initial.value.key);
    }
    // last thread also need to handle it's own sum
    if (lane == 31 && thread_keys[ITEMS_PER_THREAD - 1] != NO_DATA) {
      // flush all collected data
      if (thread_keys[ITEMS_PER_THREAD - 1] == initial.key) {
        atomicAdd(
          &dst_sum[thread_keys[ITEMS_PER_THREAD - 1] + node_id * hist_size],
          sum_current + initial.value.value);
        atomicAdd(
          &dst_count[thread_keys[ITEMS_PER_THREAD - 1] + node_id * hist_size],
          count_current + initial.value.key);
      } else {  // only thread local sum
        atomicAdd(
          &dst_sum[thread_keys[ITEMS_PER_THREAD - 1] + node_id * hist_size],
          sum_current);
        atomicAdd(
          &dst_count[thread_keys[ITEMS_PER_THREAD - 1] + node_id * hist_size],
          count_current);
      }
    }
  }
}

// clang-format off
/*[[[cog
import cog
for t in [('float', 'float'), ('float', 'double'), ('float2', 'float2'), ('float2', 'mydouble2')]:
    for bin_type in ['unsigned short', 'unsigned char']:
        cog.outl("""template __global__ void
hist_sum_multi_node<{0}, {1}, {2}, false>(
  {0} *dst_sum, unsigned *dst_count, const {0} *hist_sum_parent,
  const unsigned *hist_count_parent, const {1} *__restrict__ values,
  const unsigned *__restrict__ parent_count_iter,
  const {2} *__restrict__ bin, const unsigned hist_size,
  const unsigned end_bit, const int blocks_per_node);""".format(
      t[1], t[0], bin_type))
        cog.outl("""template __global__ void
hist_sum_multi_node<{0}, {1}, {2}, true>(
  {0} *dst_sum, unsigned *dst_count, const {0} *hist_sum_parent,
  const unsigned *hist_count_parent, const {1} *__restrict__ values,
  const unsigned *__restrict__ parent_count_iter,
  const {2} *__restrict__ bin, const unsigned hist_size,
  const unsigned end_bit, const int blocks_per_node);""".format(
      t[1], t[0], bin_type))
]]]*/
template __global__ void
hist_sum_multi_node<float, float, unsigned short, false>(
  float *dst_sum, unsigned *dst_count, const float *hist_sum_parent,
  const unsigned *hist_count_parent, const float *__restrict__ values,
  const unsigned *__restrict__ parent_count_iter,
  const unsigned short *__restrict__ bin, const unsigned hist_size,
  const unsigned end_bit, const int blocks_per_node);
template __global__ void
hist_sum_multi_node<float, float, unsigned short, true>(
  float *dst_sum, unsigned *dst_count, const float *hist_sum_parent,
  const unsigned *hist_count_parent, const float *__restrict__ values,
  const unsigned *__restrict__ parent_count_iter,
  const unsigned short *__restrict__ bin, const unsigned hist_size,
  const unsigned end_bit, const int blocks_per_node);
template __global__ void
hist_sum_multi_node<float, float, unsigned char, false>(
  float *dst_sum, unsigned *dst_count, const float *hist_sum_parent,
  const unsigned *hist_count_parent, const float *__restrict__ values,
  const unsigned *__restrict__ parent_count_iter,
  const unsigned char *__restrict__ bin, const unsigned hist_size,
  const unsigned end_bit, const int blocks_per_node);
template __global__ void
hist_sum_multi_node<float, float, unsigned char, true>(
  float *dst_sum, unsigned *dst_count, const float *hist_sum_parent,
  const unsigned *hist_count_parent, const float *__restrict__ values,
  const unsigned *__restrict__ parent_count_iter,
  const unsigned char *__restrict__ bin, const unsigned hist_size,
  const unsigned end_bit, const int blocks_per_node);
template __global__ void
hist_sum_multi_node<double, float, unsigned short, false>(
  double *dst_sum, unsigned *dst_count, const double *hist_sum_parent,
  const unsigned *hist_count_parent, const float *__restrict__ values,
  const unsigned *__restrict__ parent_count_iter,
  const unsigned short *__restrict__ bin, const unsigned hist_size,
  const unsigned end_bit, const int blocks_per_node);
template __global__ void
hist_sum_multi_node<double, float, unsigned short, true>(
  double *dst_sum, unsigned *dst_count, const double *hist_sum_parent,
  const unsigned *hist_count_parent, const float *__restrict__ values,
  const unsigned *__restrict__ parent_count_iter,
  const unsigned short *__restrict__ bin, const unsigned hist_size,
  const unsigned end_bit, const int blocks_per_node);
template __global__ void
hist_sum_multi_node<double, float, unsigned char, false>(
  double *dst_sum, unsigned *dst_count, const double *hist_sum_parent,
  const unsigned *hist_count_parent, const float *__restrict__ values,
  const unsigned *__restrict__ parent_count_iter,
  const unsigned char *__restrict__ bin, const unsigned hist_size,
  const unsigned end_bit, const int blocks_per_node);
template __global__ void
hist_sum_multi_node<double, float, unsigned char, true>(
  double *dst_sum, unsigned *dst_count, const double *hist_sum_parent,
  const unsigned *hist_count_parent, const float *__restrict__ values,
  const unsigned *__restrict__ parent_count_iter,
  const unsigned char *__restrict__ bin, const unsigned hist_size,
  const unsigned end_bit, const int blocks_per_node);
template __global__ void
hist_sum_multi_node<float2, float2, unsigned short, false>(
  float2 *dst_sum, unsigned *dst_count, const float2 *hist_sum_parent,
  const unsigned *hist_count_parent, const float2 *__restrict__ values,
  const unsigned *__restrict__ parent_count_iter,
  const unsigned short *__restrict__ bin, const unsigned hist_size,
  const unsigned end_bit, const int blocks_per_node);
template __global__ void
hist_sum_multi_node<float2, float2, unsigned short, true>(
  float2 *dst_sum, unsigned *dst_count, const float2 *hist_sum_parent,
  const unsigned *hist_count_parent, const float2 *__restrict__ values,
  const unsigned *__restrict__ parent_count_iter,
  const unsigned short *__restrict__ bin, const unsigned hist_size,
  const unsigned end_bit, const int blocks_per_node);
template __global__ void
hist_sum_multi_node<float2, float2, unsigned char, false>(
  float2 *dst_sum, unsigned *dst_count, const float2 *hist_sum_parent,
  const unsigned *hist_count_parent, const float2 *__restrict__ values,
  const unsigned *__restrict__ parent_count_iter,
  const unsigned char *__restrict__ bin, const unsigned hist_size,
  const unsigned end_bit, const int blocks_per_node);
template __global__ void
hist_sum_multi_node<float2, float2, unsigned char, true>(
  float2 *dst_sum, unsigned *dst_count, const float2 *hist_sum_parent,
  const unsigned *hist_count_parent, const float2 *__restrict__ values,
  const unsigned *__restrict__ parent_count_iter,
  const unsigned char *__restrict__ bin, const unsigned hist_size,
  const unsigned end_bit, const int blocks_per_node);
template __global__ void
hist_sum_multi_node<mydouble2, float2, unsigned short, false>(
  mydouble2 *dst_sum, unsigned *dst_count, const mydouble2 *hist_sum_parent,
  const unsigned *hist_count_parent, const float2 *__restrict__ values,
  const unsigned *__restrict__ parent_count_iter,
  const unsigned short *__restrict__ bin, const unsigned hist_size,
  const unsigned end_bit, const int blocks_per_node);
template __global__ void
hist_sum_multi_node<mydouble2, float2, unsigned short, true>(
  mydouble2 *dst_sum, unsigned *dst_count, const mydouble2 *hist_sum_parent,
  const unsigned *hist_count_parent, const float2 *__restrict__ values,
  const unsigned *__restrict__ parent_count_iter,
  const unsigned short *__restrict__ bin, const unsigned hist_size,
  const unsigned end_bit, const int blocks_per_node);
template __global__ void
hist_sum_multi_node<mydouble2, float2, unsigned char, false>(
  mydouble2 *dst_sum, unsigned *dst_count, const mydouble2 *hist_sum_parent,
  const unsigned *hist_count_parent, const float2 *__restrict__ values,
  const unsigned *__restrict__ parent_count_iter,
  const unsigned char *__restrict__ bin, const unsigned hist_size,
  const unsigned end_bit, const int blocks_per_node);
template __global__ void
hist_sum_multi_node<mydouble2, float2, unsigned char, true>(
  mydouble2 *dst_sum, unsigned *dst_count, const mydouble2 *hist_sum_parent,
  const unsigned *hist_count_parent, const float2 *__restrict__ values,
  const unsigned *__restrict__ parent_count_iter,
  const unsigned char *__restrict__ bin, const unsigned hist_size,
  const unsigned end_bit, const int blocks_per_node);
//[[[end]]] (checksum: 25c0c64a105cac1fa2bd85171d8543e1)
// clang-format on

template <typename SUM_T, typename GRAD_T, typename BIN_T>
__global__ void hist_sum(SUM_T *dst_sum, unsigned *dst_count,
                         const GRAD_T *__restrict__ values,
                         const unsigned *__restrict__ parent_count_iter,
                         const BIN_T *__restrict__ bin, const unsigned bins,
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

// clang-format off
/*[[[cog
import cog
for t in [('float', 'float'), ('float', 'double'), ('float2', 'float2'), ('float2', 'mydouble2')]:
    for bin_type in ['unsigned short', 'unsigned char']:
        cog.outl("""template __global__ void hist_sum<{0}, {1}, {2}>(
  {0} *dst_sum, unsigned *dst_count, const {1} *__restrict__ values,
  const unsigned *__restrict__ parent_count_iter,
  const {2} *__restrict__ bin, const unsigned bins, const size_t n);""".format(
      t[1], t[0], bin_type))
]]]*/
template __global__ void hist_sum<float, float, unsigned short>(
  float *dst_sum, unsigned *dst_count, const float *__restrict__ values,
  const unsigned *__restrict__ parent_count_iter,
  const unsigned short *__restrict__ bin, const unsigned bins, const size_t n);
template __global__ void hist_sum<float, float, unsigned char>(
  float *dst_sum, unsigned *dst_count, const float *__restrict__ values,
  const unsigned *__restrict__ parent_count_iter,
  const unsigned char *__restrict__ bin, const unsigned bins, const size_t n);
template __global__ void hist_sum<double, float, unsigned short>(
  double *dst_sum, unsigned *dst_count, const float *__restrict__ values,
  const unsigned *__restrict__ parent_count_iter,
  const unsigned short *__restrict__ bin, const unsigned bins, const size_t n);
template __global__ void hist_sum<double, float, unsigned char>(
  double *dst_sum, unsigned *dst_count, const float *__restrict__ values,
  const unsigned *__restrict__ parent_count_iter,
  const unsigned char *__restrict__ bin, const unsigned bins, const size_t n);
template __global__ void hist_sum<float2, float2, unsigned short>(
  float2 *dst_sum, unsigned *dst_count, const float2 *__restrict__ values,
  const unsigned *__restrict__ parent_count_iter,
  const unsigned short *__restrict__ bin, const unsigned bins, const size_t n);
template __global__ void hist_sum<float2, float2, unsigned char>(
  float2 *dst_sum, unsigned *dst_count, const float2 *__restrict__ values,
  const unsigned *__restrict__ parent_count_iter,
  const unsigned char *__restrict__ bin, const unsigned bins, const size_t n);
template __global__ void hist_sum<mydouble2, float2, unsigned short>(
  mydouble2 *dst_sum, unsigned *dst_count, const float2 *__restrict__ values,
  const unsigned *__restrict__ parent_count_iter,
  const unsigned short *__restrict__ bin, const unsigned bins, const size_t n);
template __global__ void hist_sum<mydouble2, float2, unsigned char>(
  mydouble2 *dst_sum, unsigned *dst_count, const float2 *__restrict__ values,
  const unsigned *__restrict__ parent_count_iter,
  const unsigned char *__restrict__ bin, const unsigned bins, const size_t n);
//[[[end]]] (checksum: ccde6b65791a97b2a0a90e1c9694abda)
// clang-format on
