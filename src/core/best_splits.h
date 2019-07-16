#ifndef SRC_CORE_BEST_SPLITS_H
#define SRC_CORE_BEST_SPLITS_H

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

namespace arboretum {
namespace core {
using namespace thrust;
using thrust::device_vector;
using thrust::host_vector;

template <typename SUM_T>
struct BestSplit {
 public:
  BestSplit(const unsigned length, const unsigned hist_size);
  void Clear(const unsigned size);
  void Sync(const unsigned size);
  void NextLevel(const unsigned size);
  const unsigned length;
  const unsigned hist_size;
  device_vector<float> gain;
  device_vector<int> feature;
  device_vector<SUM_T> sum;
  device_vector<unsigned> count;
  device_vector<unsigned> split_value;
  device_vector<SUM_T> parent_node_sum;
  device_vector<SUM_T> parent_node_sum_next;
  device_vector<unsigned> parent_node_count;
  device_vector<unsigned> parent_node_count_next;
  host_vector<SUM_T> parent_node_sum_h;
  host_vector<unsigned> parent_node_count_h;

  host_vector<float> gain_h;
  host_vector<int> feature_h;
  host_vector<SUM_T> sum_h;
  host_vector<unsigned> count_h;
  host_vector<unsigned> split_value_h;
};

}  // namespace core
}  // namespace arboretum

#endif