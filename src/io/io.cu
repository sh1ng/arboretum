//#include <omp.h>
#include "io.h"
#include <algorithm>
#include <functional>
#include <thrust/system/cuda/experimental/pinned_allocator.h>
#include <unordered_set>
#include <vector>

namespace arboretum {
namespace io {
using namespace std;

DataMatrix::DataMatrix(int rows, int columns)
    : rows(rows), columns(columns), columns_dense(columns), columns_sparse(0) {
  _init = false;
  data.resize(columns);
  index_device.resize(columns);
  sorted_data_device.resize(columns);
  data_reduced.resize(columns);
  reduced_size.resize(columns);
  data_reduced_mapping.resize(columns);

  for (int i = 0; i < columns; ++i) {
    data[i].resize(rows);
  }
}

void DataMatrix::Init(bool verbose) {
  if (!_init) {
    lil_column_device.resize(columns_sparse);

#pragma omp parallel for
    for (size_t i = 0; i < columns_dense; ++i) {
      data_reduced[i].resize(rows);

      std::unordered_set<float> s;
      for (float v : data[i])
        s.insert(v);
      data_reduced_mapping[i].assign(s.begin(), s.end());
      std::sort(data_reduced_mapping[i].begin(), data_reduced_mapping[i].end());
      reduced_size[i] = 32 - __builtin_clz(data_reduced_mapping[i].size());

      for (size_t j = 0; j < rows; ++j) {
        vector<float>::iterator indx =
            std::lower_bound(data_reduced_mapping[i].begin(),
                             data_reduced_mapping[i].end(), data[i][j]);
        unsigned int idx = indx - data_reduced_mapping[i].begin();
        data_reduced[i][j] = idx;
      }
    }

    for (size_t i = 0; i < columns_dense && verbose; ++i) {
      printf("feature %lu has been reduced to %u bits \n", i, reduced_size[i]);
    }
    max_reduced_size =
        *std::max_element(reduced_size.begin(), reduced_size.end());
    if (verbose)
      printf("max feature size %u \n", max_reduced_size);

    _init = true;
  }
}

void DataMatrix::UpdateGrad() {}
void DataMatrix::TransferToGPU(size_t free, bool verbose) {
  size_t data_size = sizeof(float) * rows;
  size_t copy_count = std::min(free / data_size, columns_dense);
  for (size_t i = 0; i < copy_count; ++i) {
    sorted_data_device[i].resize(rows);
    thrust::copy(data_reduced[i].begin(), data_reduced[i].end(),
                 sorted_data_device[i].begin());
  }
  if (verbose)
    printf("copied features data %ld from %ld \n", copy_count, columns_dense);

  free -= copy_count * data_size;

  copy_count = 0;

  for (size_t i = 0; i < columns_sparse; ++i) {
    size_t size = lil_column[i].size();

    if (size * sizeof(unsigned int) < free) {
      copy_count++;
      lil_column_device[i] = lil_column[i];
      free -= size * sizeof(unsigned int);
    } else {
      break;
    }
  }
  if (verbose)
    printf("copied sparse features %ld from %ld \n", copy_count,
           columns_sparse);
}
}
}
