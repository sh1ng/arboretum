#ifndef IO_H
#define IO_H

#include <functional>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/system/cuda/experimental/pinned_allocator.h>
#include <vector>

namespace arboretum {
namespace io {
using namespace thrust;

class DataMatrix {
public:
  std::vector<thrust::host_vector<
      unsigned int, thrust::cuda::experimental::pinned_allocator<unsigned int>>>
      data_categories;

  std::vector<std::vector<float>> data;
  std::vector<thrust::host_vector<
      unsigned int, thrust::cuda::experimental::pinned_allocator<unsigned int>>>
      data_reduced;

  std::vector<std::vector<float>> data_reduced_mapping;

  std::vector<thrust::device_vector<unsigned int>> data_category_device;
  std::vector<thrust::device_vector<unsigned int>> sorted_data_device;

  std::vector<float> y_hat;
  std::vector<float> y_internal;
  std::vector<unsigned char> labels;
  std::vector<unsigned char> reduced_size;
  std::vector<unsigned char> category_size;
  unsigned char max_reduced_size;
  unsigned char max_category_size;
  unsigned char max_feature_size;
  size_t rows;
  size_t columns;
  size_t columns_dense;
  size_t columns_category;
  void Init(bool verbose);
  void UpdateGrad();
  void TransferToGPU(const size_t free, bool verbose);
  DataMatrix(int rows, int columns, int columns_category);

private:
  bool _init;
  std::vector<unsigned int> SortedIndex(int column);
};
} // namespace io
} // namespace arboretum

#endif // IO_H
