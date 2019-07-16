#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "best_splits.h"
#include "common.h"
#include "cuda_helpers.h"

namespace arboretum {
namespace core {
using namespace thrust;
using thrust::device_vector;
using thrust::host_vector;

// template<typename T>
// __global__ void update_parent(T* grad_next)

// template <typename SUM_T>
// __global__ void update_next_level(SUM_T* parent  const unsigned n) {}

template <typename SUM_T>
BestSplit<SUM_T>::BestSplit(const unsigned length, const unsigned hist_size)
    : length(length), hist_size(hist_size) {
  gain.resize(length, 0);
  feature.resize(length, -1);
  SUM_T zero;
  init(zero);
  sum.resize(length, zero);
  count.resize(length, 0);
  split_value.resize(length, unsigned(-1));
  gain_h.resize(length);
  feature_h.resize(length);
  sum_h.resize(length);
  count_h.resize(length);
  parent_node_sum.resize(length + 1, zero);
  parent_node_sum_next.resize(length + 1, zero);
  parent_node_count.resize(length + 1, 0);
  parent_node_count_next.resize(length + 1, 0);

  parent_node_sum_h.resize(length + 1, zero);
  parent_node_count_h.resize(length + 1, 0);

  split_value_h.resize(length);
}

template <typename SUM_T>
void BestSplit<SUM_T>::Clear(const unsigned size) {
  thrust::fill_n(gain.begin(), size, 0.0);
  thrust::fill_n(feature.begin(), size, -1);
  thrust::fill_n(count.begin(), size, 0);
  thrust::fill_n(split_value.begin(), size, (unsigned)-1);
  SUM_T zero;
  init(zero);
  thrust::fill_n(sum.begin(), size, zero);
}

template <typename SUM_T>
void BestSplit<SUM_T>::Sync(const unsigned size) {
  thrust::copy(gain.begin(), gain.begin() + size, gain_h.begin());
  thrust::copy(feature.begin(), feature.begin() + size, feature_h.begin());
  thrust::copy(sum.begin(), sum.begin() + size, sum_h.begin());
  thrust::copy(count.begin(), count.begin() + size, count_h.begin());
  thrust::copy(split_value.begin(), split_value.begin() + size,
               split_value_h.begin());

  thrust::copy(parent_node_count.begin() + 1,
               parent_node_count.begin() + 1 + size,
               parent_node_count_h.begin() + 1);

  thrust::copy(parent_node_sum.begin() + 1, parent_node_sum.begin() + 1 + size,
               parent_node_sum_h.begin() + 1);
}

template <typename SUM_T>
void BestSplit<SUM_T>::NextLevel(const unsigned size) {
  thrust::copy(parent_node_count_next.begin() + 1,
               parent_node_count_next.begin() + 1 + size,
               parent_node_count.begin() + 1);

  thrust::copy(parent_node_sum_next.begin() + 1,
               parent_node_sum_next.begin() + 1 + size,
               parent_node_sum.begin() + 1);

  thrust::copy(parent_node_count.begin() + 1,
               parent_node_count.begin() + 1 + size,
               parent_node_count_h.begin() + 1);

  thrust::copy(parent_node_sum.begin() + 1, parent_node_sum.begin() + 1 + size,
               parent_node_sum_h.begin() + 1);
}

template class BestSplit<float>;
template class BestSplit<double>;
template class BestSplit<float2>;
template class BestSplit<mydouble2>;

}  // namespace core
}  // namespace arboretum