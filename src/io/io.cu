//#include <omp.h>
#include <thrust/sort.h>
#include <thrust/system/cuda/experimental/pinned_allocator.h>
#include <thrust/unique.h>
#include <algorithm>
#include <functional>
#include <unordered_set>
#include <vector>
#include "io.h"

namespace arboretum {
namespace io {
using namespace std;

DataMatrix::DataMatrix(int rows, int columns, int columns_category)
    : rows(rows),
      columns(columns + columns_category),
      columns_dense(columns),
      columns_category(columns_category) {
  _init = false;
  y_hat.reserve(rows);
  data.resize(columns);
  data_category_device.resize(columns_category);
  sorted_data_device.resize(columns);
  data_reduced.resize(columns);
  reduced_size.resize(columns);
  category_size.resize(columns_category);
  data_reduced_mapping.resize(columns);
  data_categories.resize(columns_category);

  for (int i = 0; i < columns; ++i) {
    data[i].resize(rows);
  }
  for (int i = 0; i < columns_category; ++i) {
    data_categories[i].resize(rows);
  }
}

void DataMatrix::InitHist(int hist_size, bool verbose) {
  if (!_init) {
    thrust::host_vector<thrust::host_vector<float>> thresholds(columns_dense);
    for (size_t i = 0; i < columns_dense; ++i) {
      thresholds[i].resize(hist_size, std::numeric_limits<float>::infinity());
      thrust::device_vector<float> d_data = data[i];
      thrust::sort(d_data.begin(), d_data.end());

      auto n = thrust::unique(d_data.begin(), d_data.end());
      int size = std::min(static_cast<int>(n - d_data.begin()), hist_size);
      reduced_size[i] = 32 - __builtin_clz(size);
      data_reduced_mapping[i].resize(size);
      for (auto j = 0; j < size - 1; ++j) {
        auto offset = (j + 1) * (n - d_data.begin()) / size;
        thresholds[i][j] = (d_data[offset] + d_data[offset - 1]) * 0.5;
      }

      data_reduced_mapping[i].assign(thresholds[i].begin(),
                                     thresholds[i].end());
    }
#pragma omp parallel for
    for (size_t i = 0; i < columns_dense; ++i) {
      data_reduced[i].resize(rows);
      for (size_t j = 0; j < rows; ++j) {
        vector<float>::iterator indx =
          std::lower_bound(data_reduced_mapping[i].begin(),
                           data_reduced_mapping[i].end(), data[i][j]);
        unsigned short idx = indx - data_reduced_mapping[i].begin();
        data_reduced[i][j] = idx;
      }
    }
    for (size_t i = 0; i < columns_dense && verbose; ++i) {
      printf("feature %lu has been reduced to %u bits \n", i, reduced_size[i]);
    }
    max_reduced_size = max_feature_size =
      *std::max_element(reduced_size.begin(), reduced_size.end());
    if (verbose) printf("max feature size %u \n", max_reduced_size);

    this->_init = true;
  }
}

void DataMatrix::InitExact(bool verbose) {
  if (!_init) {
#pragma omp parallel for
    for (size_t i = 0; i < columns_dense; ++i) {
      data_reduced[i].resize(rows);

      std::unordered_set<float> s;
      for (float v : data[i]) s.insert(v);
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

#pragma omp parallel for
    for (size_t i = 0; i < columns_category; ++i) {
      unsigned int m =
        *std::max_element(data_categories[i].begin(), data_categories[i].end());
      category_size[i] = 32 - __builtin_clz(m);
    }

    for (size_t i = 0; i < columns_dense && verbose; ++i) {
      printf("feature %lu has been reduced to %u bits \n", i, reduced_size[i]);
    }
    max_reduced_size =
      *std::max_element(reduced_size.begin(), reduced_size.end());
    if (verbose) printf("max feature size %u \n", max_reduced_size);

    if (columns_category == 0)
      max_feature_size = max_reduced_size;
    else {
      max_category_size =
        *std::max_element(category_size.begin(), category_size.end());
      max_feature_size = std::max(max_reduced_size, max_category_size);
    }
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

  for (size_t i = 0; i < columns_category; ++i) {
    if (rows * sizeof(unsigned int) < free) {
      copy_count++;
      data_category_device[i].resize(rows);
      thrust::copy(data_categories[i].begin(), data_categories[i].end(),
                   data_category_device[i].begin());
      free -= rows * sizeof(unsigned int);
    } else {
      break;
    }
  }
  if (verbose)
    printf("copied category features %ld from %ld \n", copy_count,
           columns_category);
}
}  // namespace io
}  // namespace arboretum
