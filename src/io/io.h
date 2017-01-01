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
      index;

  std::vector<thrust::host_vector<
      int, thrust::cuda::experimental::pinned_allocator<int>>>
      sparse_index;

  std::vector<std::vector<float>> data;

  std::vector<thrust::host_vector<
      float, thrust::cuda::experimental::pinned_allocator<float>>>
      sorted_data;

  std::vector<thrust::device_vector<unsigned int>> index_device;
  std::vector<thrust::device_vector<float>> sorted_data_device;

  std::vector<thrust::host_vector<
      unsigned int, thrust::cuda::experimental::pinned_allocator<unsigned int>>>
      lil_column;

  std::vector<thrust::device_vector<unsigned int>> lil_column_device;

  std::vector<std::vector<unsigned int>> lil_row;

  std::vector<float> y_hat;
  std::vector<float> y_internal;
  std::vector<unsigned char> labels;
  size_t rows;
  size_t columns;
  size_t columns_dense;
  size_t columns_sparse;
  void Init();
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
