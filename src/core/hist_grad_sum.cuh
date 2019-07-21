#ifndef SRC_CORE_HIST_GRAD_SUM_CUH
#define SRC_CORE_HIST_GRAD_SUM_CUH

#include "cuda_runtime.h"

#define HIST_SUM_NO_DATA (unsigned short)-1
#define HIST_SUM_BLOCK_DIM 128
#define HIST_SUM_ITEMS_PER_THREAD 10

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

template <typename SUM_T>
__global__ void update(SUM_T *sum_dst, unsigned *count_dst,
                       const SUM_T *parent_sum, const unsigned *parent_count,
                       const SUM_T *sum_src, const unsigned *count_src,
                       const unsigned n);

#endif