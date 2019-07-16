#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "common.h"
#include "cuda_helpers.h"
#include "histogram.h"

namespace arboretum {
namespace core {
using namespace thrust;
using thrust::device_vector;
using thrust::host_vector;

template <typename SUM_T>
Histogram<SUM_T>::Histogram(const unsigned size, const unsigned hist_size,
                            const unsigned features)
    : size(size), hist_size(hist_size), features(features) {
  at_level.resize(features, -1);
  grad_hist.resize(features);
  count_hist.resize(features);
  SUM_T zero;
  init(zero);
  for (unsigned i = 0; i < features; ++i) {
    grad_hist[i].resize(size * hist_size, zero);
    count_hist[i].resize(size * hist_size, 0);
  }
}

template <typename SUM_T>
void Histogram<SUM_T>::Update(const device_vector<SUM_T> &src_grad,
                              const device_vector<unsigned> &src_count,
                              const unsigned fid, const unsigned level,
                              const cudaStream_t stream) {
  OK(cudaMemcpyAsync(thrust::raw_pointer_cast(this->grad_hist[fid].data()),
                     thrust::raw_pointer_cast(src_grad.data()),
                     sizeof(SUM_T) * (1 << level) * this->hist_size,
                     cudaMemcpyDeviceToDevice, stream));
  OK(cudaMemcpyAsync(thrust::raw_pointer_cast(this->count_hist[fid].data()),
                     thrust::raw_pointer_cast(src_count.data()),
                     sizeof(unsigned) * (1 << level) * this->hist_size,
                     cudaMemcpyDeviceToDevice, stream));
  this->at_level[fid] = level;
}

template <typename SUM_T>
bool Histogram<SUM_T>::CanUseTrick(const unsigned fid, const unsigned level) {
  return level != 0 && (this->at_level[fid] + 1 == level);
}

template <typename SUM_T>
void Histogram<SUM_T>::Clear() {
  thrust::fill_n(at_level.begin(), features, -1);
  for (unsigned i = 0; i < features; ++i) {
    SUM_T zero;
    init(zero);
    thrust::fill_n(grad_hist[i].begin(), size * hist_size, zero);
    thrust::fill_n(count_hist[i].begin(), size * hist_size, 0);
  }
}

template class Histogram<float>;
template class Histogram<double>;
template class Histogram<float2>;
template class Histogram<mydouble2>;

}  // namespace core
}  // namespace arboretum