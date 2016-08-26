#ifndef IO_H
#define IO_H

#include <vector>
#include <functional>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

namespace arboretum {
  namespace io {
    using namespace thrust;

    class DataMatrix {
    public:
      std::vector<thrust::host_vector<int> > index;
      std::vector<thrust::host_vector<float> > data;
      std::vector<thrust::host_vector<float> > sorted_data;
      std::vector<thrust::host_vector<float> > sorted_grad;
      std::vector<float> y_hat;
      std::vector<float> y;
      thrust::host_vector<float> grad;
      size_t rows;
      size_t columns;
      void Init(const float initial_y, std::function<float const(const float, const float)> func);
      void UpdateGrad();
      DataMatrix(int rows, int columns);
    private:
      std::function<float const(const float, const float)> _gradFunc;
      bool _init;
      std::vector<int> SortedIndex(int column);
    };
  }
}

#endif // IO_H
