#ifndef IO_H
#define IO_H

#include <vector>
#include <functional>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/system/cuda/experimental/pinned_allocator.h>

namespace arboretum {
  namespace io {
    using namespace thrust;

    class DataMatrix {
    public:
      std::vector<thrust::host_vector<int, thrust::cuda::experimental::pinned_allocator< int > > > index;
      std::vector<thrust::host_vector<float, thrust::cuda::experimental::pinned_allocator< float > > > data;
      std::vector<thrust::host_vector<float, thrust::cuda::experimental::pinned_allocator< float > > > sorted_data;

      std::vector<thrust::device_vector<int>> index_device;
      std::vector<thrust::device_vector<float>> sorted_data_device;

      std::vector<float> y_hat;
      std::vector<float> y;
      thrust::host_vector<float, thrust::cuda::experimental::pinned_allocator< float > > grad;
      size_t rows;
      size_t columns;
      void Init(const float initial_y, std::function<float(const float, const float)> func);
      void UpdateGrad();
      void TransferToGPU(const size_t free, bool verbose);
      DataMatrix(int rows, int columns);
    private:
      std::function<float(const float, const float)> _gradFunc;
      bool _init;
      std::vector<int> SortedIndex(int column);
    };
  }
}

#endif // IO_H
