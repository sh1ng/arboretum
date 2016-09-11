//#include <omp.h>
#include <vector>
#include <algorithm>
#include <functional>
#include <thrust/system/cuda/experimental/pinned_allocator.h>
#include "io.h"

namespace arboretum {
  namespace io {
    using namespace std;

    DataMatrix::DataMatrix(int rows, int columns) : rows(rows), columns(columns)
    {
      _init = false;
      data.resize(columns);
      sorted_data.resize(columns);
      index.resize(columns);
      grad.resize(rows);
      for(int i = 0; i < columns; ++i){
            data[i].resize(rows);
      }
    }

    void DataMatrix::Init(const float initial_y, std::function<float const(const float, const float)> func){
      if(!_init){
          _gradFunc = func;
          y.resize(y_hat.size(), initial_y);

          #pragma omp parallel for
          for(size_t i = 0; i < data.size(); ++i){
            index[i] = SortedIndex(i);
            std::vector<float> tmp(data[i].size());
            for(size_t j = 0; j < data[i].size(); ++j){
                tmp[j] = data[i][index[i][j]];
              }
            sorted_data[i] = tmp;
          }
          _init = true;
        }
    }

    std::vector<int> DataMatrix::SortedIndex(int column){
        auto &v = data[column];
        size_t size = v.size();
        std::vector<int> idx(size);
        for(size_t i = 0; i < size; i ++)
          idx[i] = i;

        sort(idx.begin(), idx.end(),
             [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});

        return idx;
    }

    void DataMatrix::UpdateGrad(){
      for(size_t i = 0; i < rows; ++i){
          grad[i] = _gradFunc(y[i], y_hat[i]);
        }
    }
  }
}
