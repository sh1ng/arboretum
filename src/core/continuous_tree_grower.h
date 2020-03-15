#ifndef SRC_CORE_CONTINOUS_TREE_GROWER_H
#define SRC_CORE_CONTINOUS_TREE_GROWER_H

#include "builder.h"

namespace arboretum {
namespace core {

template <typename NODE_T, typename BIN_T, typename GRAD_T, typename SUM_T>
class ContinuousTreeGrower : public BaseGrower<NODE_T, BIN_T, GRAD_T, SUM_T> {
 public:
  ContinuousTreeGrower(const size_t size, const unsigned depth,
                       const unsigned hist_size,
                       const BestSplit<SUM_T> *best = NULL,
                       Histogram<SUM_T> *features_histogram = NULL,
                       const InternalConfiguration *config = NULL);

  device_vector<BIN_T> node_fvalue;
  device_vector<BIN_T> node_fvalue_sorted;
  device_vector<SUM_T> sum;
  device_vector<unsigned int> run_lenght;
  NODE_T *best_split_h;

  void ApplySplit(NODE_T *row2Node, const unsigned level, const BIN_T threshold,
                  size_t from, size_t to);

  void ProcessDenseFeature(const device_vector<unsigned> &partitioning_index,
                           const device_vector<NODE_T> &row2Node,
                           const device_vector<GRAD_T> &grad_d,
                           device_vector<BIN_T> &fvalue_d, BIN_T *fvalue_h,
                           const device_vector<SUM_T> &parent_node_sum,
                           const device_vector<unsigned int> &parent_node_count,
                           const unsigned char fvalue_size,
                           const unsigned level, const unsigned depth,
                           const GainFunctionParameters gain_param,
                           const bool partition_only, const int fid = -1);

  template <typename NODE_VALUE_T>
  void ProcessCategoryFeature(
    const device_vector<NODE_T> &row2Node, const device_vector<GRAD_T> &grad_d,
    const device_vector<unsigned short> &fvalue_d,
    const host_vector<unsigned short> &fvalue_h,
    const device_vector<SUM_T> &parent_node_sum,
    const device_vector<unsigned int> &parent_node_count,
    const unsigned char category_size, const size_t level,
    const GainFunctionParameters gain_param) {
    // size_t lenght = 1 << level;
    // OK(cudaMemsetAsync(thrust::raw_pointer_cast(this->result_d.data()), 0,
    //                    lenght * sizeof(my_atomics), this->stream));

    // device_vector<unsigned int> *fvalue_tmp = NULL;

    // if (fvalue_d.size() > 0) {
    //   fvalue_tmp = const_cast<device_vector<unsigned short> *>(&(fvalue_d));
    // } else {
    //   OK(cudaMemcpyAsync(thrust::raw_pointer_cast((this->fvalue.data())),
    //                      thrust::raw_pointer_cast(fvalue_h.data()),
    //                      this->size * sizeof(unsigned short),
    //                      cudaMemcpyHostToDevice, this->stream));
    //   fvalue_tmp = const_cast<device_vector<unsigned short>
    //   *>(&(this->fvalue));
    // }

    // assign_kernel<<<this->gridSizeGather, this->blockSizeGather, 0,
    //                 this->stream>>>(
    //   thrust::raw_pointer_cast(fvalue_tmp->data()),
    //   thrust::raw_pointer_cast(row2Node.data()), category_size,
    //   (NODE_VALUE_T *)thrust::raw_pointer_cast(node_fvalue.data()),
    //   this->size);

    // OK(cub::DeviceRadixSort::SortPairs(
    //   this->temp_bytes, this->temp_bytes_allocated,
    //   (NODE_VALUE_T *)thrust::raw_pointer_cast(node_fvalue.data()),
    //   (NODE_VALUE_T *)thrust::raw_pointer_cast(node_fvalue_sorted.data()),
    //   thrust::raw_pointer_cast((grad_d.data())),
    //   thrust::raw_pointer_cast(this->grad_sorted.data()), this->size, 0,
    //   category_size + level + 1, this->stream));

    // const NODE_VALUE_T mask = (1 << (category_size)) - 1;

    // SUM_T initial_value;
    // init(initial_value);
    // cub::Sum sum_op;

    // OK(cub::DeviceReduce::ReduceByKey(
    //   this->temp_bytes, this->temp_bytes_allocated,
    //   (NODE_VALUE_T *)thrust::raw_pointer_cast(node_fvalue_sorted.data()),
    //   (NODE_VALUE_T *)thrust::raw_pointer_cast(node_fvalue.data()),
    //   thrust::raw_pointer_cast(this->grad_sorted.data()),
    //   thrust::raw_pointer_cast(sum.data()),
    //   thrust::raw_pointer_cast(run_lenght.data()), sum_op, this->size,
    //   this->stream));

    // OK(cub::DeviceRunLengthEncode::Encode(
    //   this->temp_bytes, this->temp_bytes_allocated,
    //   (NODE_VALUE_T *)thrust::raw_pointer_cast(node_fvalue_sorted.data()),
    //   (NODE_VALUE_T *)thrust::raw_pointer_cast(node_fvalue.data()),
    //   thrust::raw_pointer_cast(this->fvalue.data()),
    //   thrust::raw_pointer_cast(run_lenght.data()), this->size,
    //   this->stream));

    // gain_kernel_category<<<this->gridSizeGain, this->blockSizeGain, 0,
    //                        this->stream>>>(
    //   thrust::raw_pointer_cast(sum.data()),
    //   thrust::raw_pointer_cast(this->fvalue.data()),
    //   (NODE_VALUE_T *)thrust::raw_pointer_cast(node_fvalue.data()),
    //   category_size, mask, thrust::raw_pointer_cast(parent_node_sum.data()),
    //   thrust::raw_pointer_cast(parent_node_count.data()),
    //   thrust::raw_pointer_cast(run_lenght.data()), gain_param,
    //   thrust::raw_pointer_cast(this->result_d.data()));
  }

  void FindBest(BestSplit<SUM_T> &best, device_vector<NODE_T> &row2Node,
                const device_vector<SUM_T> &parent_node_sum,
                const device_vector<unsigned int> &parent_node_count,
                const unsigned fid, const unsigned level, const unsigned depth,
                const unsigned size);
};
}  // namespace core
}  // namespace arboretum

#endif