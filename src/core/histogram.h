#ifndef SRC_CORE_HISTOGRAM_H
#define SRC_CORE_HISTOGRAM_H

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

namespace arboretum {
namespace core {
using namespace thrust;
using thrust::device_vector;
using thrust::host_vector;

template <typename SUM_T>
class Histogram {
 public:
  Histogram(const unsigned size, const unsigned hist_size,
            const unsigned features);
  void Update(const device_vector<SUM_T> &src_grad,
              const device_vector<unsigned> &src_count, const unsigned fid,
              const unsigned level, const cudaStream_t stream = 0);
  bool CanUseTrick(const unsigned fid, const unsigned level);
  void Clear();
  const unsigned size;
  const unsigned hist_size;
  const unsigned features;
  host_vector<device_vector<SUM_T>> grad_hist;
  host_vector<device_vector<unsigned>> count_hist;
  host_vector<int> at_level;
};

}  // namespace core
}  // namespace arboretum

#endif