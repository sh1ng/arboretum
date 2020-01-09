#ifndef SRC_CORE_SPLIT_CUH
#define SRC_CORE_SPLIT_CUH

#include "param.h"

namespace arboretum {
namespace core {

template <typename GRAD_TYPE>
struct Split {
  float split_value;
  unsigned category;
  int fid;
  double gain;
  GRAD_TYPE sum_grad;
  unsigned count;
  unsigned quantized;
  Split();
  void Clean();
  float LeafWeight(const TreeParam &param) const;

  float LeafWeight(const unsigned parent_size, const GRAD_TYPE parent_sum,
                   const TreeParam &param) const;
};
}  // namespace core
}  // namespace arboretum

#endif