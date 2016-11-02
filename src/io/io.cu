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
      index_device.resize(columns);
      sorted_data_device.resize(columns);

      for(int i = 0; i < columns; ++i){
            data[i].resize(rows);
      }
    }

    void DataMatrix::Init(){
      if(!_init){
          #pragma omp parallel for
          for(size_t i = 0; i < columns; ++i){
            index[i] = SortedIndex(i);
          }

          for(size_t i = 0; i < columns; ++i){
            std::vector<float> tmp(data[i].size());
            #pragma omp parallel for simd
            for(size_t j = 0; j < rows; ++j){
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

    }
    void DataMatrix::TransferToGPU(const size_t free, bool verbose){
      size_t index_size = sizeof(int) * rows;
      size_t data_size = sizeof(float) * rows;
      size_t copy_count = std::min(free / index_size, columns);
      for(size_t i = 0; i < copy_count; ++i){
          index_device[i] = index[i];
        }
      if(verbose)
        printf("copied index data %ld from %ld \n", copy_count, columns);

      copy_count = std::min((free - copy_count * sizeof(int) * rows)/data_size, columns);
      for(size_t i = 0; i < copy_count; ++i){
          sorted_data_device[i].resize(rows + 1);
          sorted_data_device[i][0] = -std::numeric_limits<float>::infinity();
          thrust::copy(sorted_data[i].begin(), sorted_data[i].end(), sorted_data_device[i].begin() + 1);
        }
      if(verbose)
        printf("copied features data %ld from %ld \n", copy_count, columns);
    }
  }
}
