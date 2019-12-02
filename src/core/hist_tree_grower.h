#ifndef SRC_CORE_HIST_TREE_GROWER_H
#define SRC_CORE_HIST_TREE_GROWER_H

#include "builder.h"

namespace arboretum {
namespace core {

template <typename NODE_T, typename GRAD_T, typename SUM_T>
class HistTreeGrower : public BaseGrower<NODE_T, GRAD_T, SUM_T> {
 public:
  HistTreeGrower(const size_t size, const unsigned depth,
                 const unsigned hist_size, const BestSplit<SUM_T> *best,
                 Histogram<SUM_T> *features_histogram,
                 const InternalConfiguration *config);

  static void HistSum(SUM_T *sum, unsigned *bin_count,
                      const SUM_T *hist_sum_parent,
                      const unsigned *hist_count_parent, const GRAD_T *grad,
                      const unsigned *node_size, const unsigned short *fvalue,
                      const unsigned hist_size_bits, const unsigned hist_size,
                      const unsigned size, const bool use_trick,
                      cudaStream_t stream = 0);

  static void HistSumStatic(SUM_T *sum, unsigned *bin_count,
                            const SUM_T *hist_sum_parent,
                            const unsigned *hist_count_parent,
                            const GRAD_T *grad, const unsigned *node_size,
                            const unsigned short *fvalue,
                            const unsigned hist_size_bits,
                            const unsigned hist_size, const unsigned size,
                            const bool use_trick, cudaStream_t stream = 0);

  static void HistSumSingleNode(SUM_T *sum, unsigned *bin_count,
                                const GRAD_T *grad, const unsigned *node_size,
                                const unsigned short *fvalue,
                                const unsigned hist_size_bits,
                                const unsigned size, cudaStream_t stream = 0);

  static void HistSumNaive(SUM_T *sum, unsigned *bin_count, const GRAD_T *grad,
                           const unsigned *node_size,
                           const unsigned short *fvalue,
                           const unsigned hist_size, const unsigned size,
                           cudaStream_t stream = 0);

  template <typename NODE_VALUE_T>
  void ProcessDenseFeature(const device_vector<unsigned> &partitioning_index,
                           const device_vector<NODE_T> &row2Node,
                           const device_vector<GRAD_T> &grad_d,
                           device_vector<unsigned short> &fvalue_d,
                           unsigned short *fvalue_h,
                           const device_vector<SUM_T> &parent_node_sum,
                           const device_vector<unsigned int> &parent_node_count,
                           const unsigned char fvalue_size,
                           const unsigned level, const unsigned depth,
                           const GainFunctionParameters gain_param,
                           const bool partition_only, const int fid);

  template <typename NODE_VALUE_T>
  inline void ProcessCategoryFeature(
    const device_vector<NODE_T> &row2Node, const device_vector<GRAD_T> &grad_d,
    const device_vector<unsigned short> &fvalue_d,
    const host_vector<unsigned short> &fvalue_h,
    const device_vector<SUM_T> &parent_node_sum,
    const device_vector<unsigned int> &parent_node_count,
    const unsigned char category_size, const size_t level,
    const GainFunctionParameters gain_param) {}

  void FindBest(BestSplit<SUM_T> &best, device_vector<NODE_T> &row2Node,
                const device_vector<SUM_T> &parent_node_sum,
                const device_vector<unsigned int> &parent_node_count,
                unsigned fid, const unsigned level, const unsigned depth,
                const unsigned size);

  device_vector<SUM_T> hist_prefix_sum;
  device_vector<unsigned> hist_bin_count;
  device_vector<unsigned> hist_prefix_count;

  const unsigned hist_size;
  unsigned hist_size_bits;
};

}  // namespace core
}  // namespace arboretum

#endif