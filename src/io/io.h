#ifndef SRC_IO_H
#define SRC_IO_H

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/system/cuda/experimental/pinned_allocator.h>
#include <functional>
#include <vector>

namespace arboretum {
namespace io {
using namespace thrust;

class DataMatrix {
 public:
  std::vector<thrust::host_vector<
    unsigned short,
    thrust::cuda::experimental::pinned_allocator<unsigned short>>>
    data_categories;

  std::vector<thrust::host_vector<float>> data;

  std::vector<std::vector<float>> data_reduced_mapping;

  std::vector<thrust::device_vector<unsigned short>> data_category_device;

  thrust::host_vector<float> y;
  thrust::host_vector<float> y_hat;
  std::vector<unsigned char> labels;
  std::vector<unsigned char> reduced_size;
  std::vector<unsigned char> category_size;
  std::vector<float> weights;
  unsigned char max_reduced_size;
  unsigned char max_category_size;
  unsigned char max_feature_size;
  const size_t rows;
  const size_t columns;
  const size_t columns_dense;
  const size_t columns_category;
  void InitExact(bool verbose);
  void InitHist(int hist_size, bool verbose);
  void UpdateGrad();
  void TransferToGPU(const size_t free, bool verbose);

  template <typename T>
  thrust::host_vector<T, thrust::cuda::experimental::pinned_allocator<T>>
    &GetHostData(int column);

  template <typename T>
  thrust::device_vector<T> &GetDeviceData(int column);

  DataMatrix(int rows, int columns, int columns_category);

 private:
  bool _init;
  std::vector<unsigned int> SortedIndex(int column);

  template <typename T>
  void InitHistInternal(int hist_size, bool verbose);

template <typename T>
  void TransferToGPUInternal(const size_t free, bool verbose);

  std::vector<thrust::host_vector<
    unsigned int, thrust::cuda::experimental::pinned_allocator<unsigned int>>>
    data_reduced_u32;

  std::vector<thrust::host_vector<
    unsigned short,
    thrust::cuda::experimental::pinned_allocator<unsigned short>>>
    data_reduced_u16;

  std::vector<thrust::host_vector<
    unsigned char, thrust::cuda::experimental::pinned_allocator<unsigned char>>>
    data_reduced_u8;

  std::vector<thrust::device_vector<unsigned int>> data_reduced_u32_device;

  std::vector<thrust::device_vector<unsigned short>> data_reduced_u16_device;

  std::vector<thrust::device_vector<unsigned char>> data_reduced_u8_device;
};
}  // namespace io
}  // namespace arboretum

#endif  // IO_H
