#include <limits>
#include "cuda_helpers.h"
#include "gain.cuh"
#include "param.h"
#include "split.cuh"

namespace arboretum {
namespace core {

template <class GRAD_TYPE>
Split<GRAD_TYPE>::Split() {
  Clean();
};

template <class GRAD_TYPE>
void Split<GRAD_TYPE>::Clean() {
  fid = category = (unsigned int)-1;
  gain = 0.0;
  init(sum_grad);
  count = 0;
  split_value = std::numeric_limits<float>::infinity();
  quantized = (unsigned)-1;
};

template <class GRAD_TYPE>
float Split<GRAD_TYPE>::LeafWeight(const TreeParam &param) const {
  return Weight(sum_grad, count, param);
};

template <class GRAD_TYPE>
float Split<GRAD_TYPE>::LeafWeight(const unsigned parent_size,
                                   const GRAD_TYPE parent_sum,
                                   const TreeParam &param) const {
  return Weight(parent_sum - sum_grad, parent_size - count, param);
};

/*[[[cog
import cog
types = ['float', 'double', 'float2', 'mydouble2']
cog.outl("// clang-format off")
for t in types:
    cog.outl("template class Split<%s>;" % t)
cog.outl("// clang-format on")
]]]*/
// clang-format off
template class Split<float>;
template class Split<double>;
template class Split<float2>;
template class Split<mydouble2>;
// clang-format on
//[[[end]]] (checksum: 9ae16fc3222eb9fed20370d2642626d0)

}  // namespace core
}  // namespace arboretum