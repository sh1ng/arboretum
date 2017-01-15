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
      int, thrust::cuda::experimental::pinned_allocator<int>>>
      sparse_index;

  std::vector<std::vector<float>> data;
  std::vector<thrust::host_vector<
      unsigned int, thrust::cuda::experimental::pinned_allocator<unsigned int>>>
      data_reduced;

  std::vector<std::vector<float>> data_reduced_mapping;

  std::vector<thrust::device_vector<unsigned int>> index_device;
  std::vector<thrust::device_vector<unsigned int>> sorted_data_device;

  std::vector<thrust::host_vector<
      unsigned int, thrust::cuda::experimental::pinned_allocator<unsigned int>>>
      lil_column;

  std::vector<thrust::device_vector<unsigned int>> lil_column_device;

  std::vector<std::vector<unsigned int>> lil_row;

  std::vector<float> y_hat;
  std::vector<float> y_internal;
  std::vector<unsigned char> labels;
  std::vector<unsigned char> reduced_size;
  unsigned char max_reduced_size;
  size_t rows;
  size_t columns;
  size_t columns_dense;
  size_t columns_sparse;
  void Init(bool verbose);
  void UpdateGrad();
  void TransferToGPU(const size_t free, bool verbose);
  DataMatrix(int rows, int columns);

private:
  bool _init;
  std::vector<unsigned int> SortedIndex(int column);
};
}
}

#endif // IO_H
