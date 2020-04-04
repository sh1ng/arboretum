//#include <omp.h>
#include <thrust/sort.h>
#include <thrust/system/cuda/experimental/pinned_allocator.h>
#include <thrust/unique.h>

#include <algorithm>
#include <functional>
#include <unordered_set>
#include <vector>

#include "core/cuda_helpers.h"
#include "cub/cub.cuh"
#include "io.h"

namespace arboretum {
namespace io {
using namespace std;

#define ITEMS 8

struct ValueIndexSegment {
  bool border;
  float value;
  unsigned count;
};

class ValueSegmentIterator {
 public:
  // Required iterator traits
  typedef ValueSegmentIterator self_type;  ///< My own type
  typedef ptrdiff_t difference_type;       ///< Type to express the result of
                                      ///< subtracting one iterator from another
  typedef ValueIndexSegment
    value_type;  ///< The type of the element the iterator can point to
  typedef value_type *
    pointer;  ///< The type of a pointer to an element the iterator can point to
  typedef value_type reference;  ///< The type of a reference to an element the
                                 ///< iterator can point to

  typedef typename thrust::detail::iterator_facade_category<
    thrust::any_system_tag, thrust::random_access_traversal_tag, value_type,
    reference>::type iterator_category;  ///< The iterator category

 private:
  const unsigned segment_size;
  const float *itr;
  difference_type offset;

 public:
  /// Constructor
  __host__ __device__ __forceinline__ ValueSegmentIterator(
    const unsigned segment_size,
    const float *itr,            ///< Input iterator to wrap
    difference_type offset = 0)  ///< OffsetT (in items) from \p itr denoting
                                 ///< the position of the iterator
      : segment_size(segment_size), itr(itr), offset(offset) {}

  /// Postfix increment
  __host__ __device__ __forceinline__ self_type operator++(int) {
    self_type retval = *this;
    offset++;
    return retval;
  }

  /// Prefix increment
  __host__ __device__ __forceinline__ self_type operator++() {
    offset++;
    return *this;
  }

  /// Indirection
  __host__ __device__ __forceinline__ reference operator*() const {
    difference_type prev_offset = offset > 0 ? offset - 1 : 0;
    value_type retval;
    retval.value = (itr[offset] + itr[prev_offset]) * 0.5;
    retval.border = (itr[offset] != itr[prev_offset]);
    retval.count = retval.border;
    return retval;
  }

  /// Addition
  template <typename Distance>
  __host__ __device__ __forceinline__ self_type operator+(Distance n) const {
    self_type retval(segment_size, itr, offset + n);
    return retval;
  }

  /// Addition assignment
  template <typename Distance>
  __host__ __device__ __forceinline__ self_type &operator+=(Distance n) {
    offset += n;
    return *this;
  }

  /// Subtraction
  template <typename Distance>
  __host__ __device__ __forceinline__ self_type operator-(Distance n) const {
    self_type retval(itr, offset - n);
    return retval;
  }

  /// Subtraction assignment
  template <typename Distance>
  __host__ __device__ __forceinline__ self_type &operator-=(Distance n) {
    offset -= n;
    return *this;
  }

  /// Distance
  __host__ __device__ __forceinline__ difference_type
  operator-(self_type other) const {
    return offset - other.offset;
  }

  /// Array subscript
  template <typename Distance>
  __host__ __device__ __forceinline__ reference operator[](Distance n) const {
    self_type offset = (*this) + n;
    return *offset;
  }

  /// Equal to
  __host__ __device__ __forceinline__ bool operator==(const self_type &rhs) {
    return ((itr == rhs.itr) && (offset == rhs.offset));
  }

  /// Not equal to
  __host__ __device__ __forceinline__ bool operator!=(const self_type &rhs) {
    return ((itr != rhs.itr) || (offset != rhs.offset));
  }

  /// Normalize
  __host__ __device__ __forceinline__ void normalize() {
    itr += offset;
    offset = 0;
  }

  /// ostream operator
  friend std::ostream &operator<<(std::ostream &os, const self_type & /*itr*/) {
    return os;
  }
};

struct SegmentLength {
  __device__ __forceinline__ ValueIndexSegment
  operator()(const ValueIndexSegment &a, const ValueIndexSegment &b) const {
    ValueIndexSegment ret;
    ret.value = b.value;
    ret.count = a.count + b.count;
    ret.border = b.border;
    return ret;
  }
};

class ThresholdOutputIterator {
 public:
  // Required iterator traits
  typedef ThresholdOutputIterator self_type;  ///< My own type
  typedef ptrdiff_t difference_type;          ///< Type to express the result of
                                      ///< subtracting one iterator from another
  typedef void
    value_type;  ///< The type of the element the iterator can point to
  typedef void
    pointer;  ///< The type of a pointer to an element the iterator can point to
  typedef void reference;  ///< The type of a reference to an element the
                           ///< iterator can point to

  typedef typename thrust::detail::iterator_facade_category<
    thrust::any_system_tag, thrust::random_access_traversal_tag, value_type,
    reference>::type iterator_category;  ///< The iterator category

 private:
  float *thresholds;
  unsigned size;
  difference_type offset;

 public:
  /// Constructor
  __host__ __device__ __forceinline__
  ThresholdOutputIterator(float *thresholds, unsigned size,
                          difference_type offset = 0)  ///< Base offset
      : thresholds(thresholds), size(size), offset(offset) {}

  /// Postfix increment
  __host__ __device__ __forceinline__ self_type operator++(int) {
    self_type retval = *this;
    offset++;
    return retval;
  }

  /// Prefix increment
  __host__ __device__ __forceinline__ self_type operator++() {
    offset++;
    return *this;
  }

  /// Indirection
  __host__ __device__ __forceinline__ self_type &operator*() {
    // return self reference, which can be assigned to anything
    return *this;
  }

  /// Addition
  template <typename Distance>
  __host__ __device__ __forceinline__ self_type operator+(Distance n) const {
    self_type retval(thresholds, size, offset + n);
    return retval;
  }

  /// Addition assignment
  template <typename Distance>
  __host__ __device__ __forceinline__ self_type &operator+=(Distance n) {
    offset += n;
    return *this;
  }

  /// Subtraction
  template <typename Distance>
  __host__ __device__ __forceinline__ self_type operator-(Distance n) const {
    self_type retval(thresholds, size, offset - n);
    return retval;
  }

  /// Subtraction assignment
  template <typename Distance>
  __host__ __device__ __forceinline__ self_type &operator-=(Distance n) {
    offset -= n;
    return *this;
  }

  /// Distance
  __host__ __device__ __forceinline__ difference_type
  operator-(self_type other) const {
    return offset - other.offset;
  }

  /// Array subscript
  template <typename Distance>
  __host__ __device__ __forceinline__ self_type &operator[](Distance n) {
    // return self reference, which can be assigned to anything
    self_type retval(thresholds, size, offset + n);
    return *retval;
  }

  /// Structure dereference
  __host__ __device__ __forceinline__ pointer operator->() { return; }

  /// Assignment to self (no-op)
  __host__ __device__ __forceinline__ void operator=(self_type const &other) {
    offset = other.offset;
    size = other.size;
  }

  /// Assignment to anything else (no-op)
  __device__ __forceinline__ void operator=(ValueIndexSegment const &value) {
    if (value.border) {
      int segment = int(min(value.count + 1, unsigned(offset / size + 1))) - 2;
      if (segment >= 0) {
        union fi {
          float f;
          int i;
        };

        fi loc, test;
        loc.f = value.value;
        test.f = thresholds[segment];
        while (loc.f < test.f)
          test.i = atomicCAS(((int *)thresholds) + segment, test.i, loc.i);
      }
    }
  }

  /// Cast to void* operator
  __host__ __device__ __forceinline__ operator void *() const { return NULL; }

  /// Equal to
  __host__ __device__ __forceinline__ bool operator==(const self_type &rhs) {
    return (offset == rhs.offset);
  }

  /// Not equal to
  __host__ __device__ __forceinline__ bool operator!=(const self_type &rhs) {
    return (offset != rhs.offset);
  }

  /// ostream operator
  friend std::ostream &operator<<(std::ostream &os, const self_type &itr) {
    os << "[" << itr.offset << "]";
    return os;
  }
};

template <typename T, int ITEMS_PER_THREAD>
__global__ void build_histogram(T *bin, const float *threshold,
                                const float *fvalue, const int hist_size,
                                const int unique_size, const size_t n) {
  extern __shared__ float values[];
  const int size = min(unique_size, hist_size);
  if (threadIdx.x < hist_size) {
    values[threadIdx.x] = INFINITY;
    if (threadIdx.x < size - 1) values[threadIdx.x] = threshold[threadIdx.x];
  }

  __syncthreads();

#pragma unroll
  for (unsigned i = 0; i < ITEMS_PER_THREAD; ++i) {
    unsigned idx =
      blockDim.x * blockIdx.x * ITEMS_PER_THREAD + i * blockDim.x + threadIdx.x;
    if (idx < n) bin[idx] = lower_bound<float>(values, fvalue[idx], size);
  }
}

DataMatrix::DataMatrix(int rows, int columns, int columns_category)
    : rows(rows),
      columns(columns + columns_category),
      columns_dense(columns),
      columns_category(columns_category) {
  _init = false;
  data.resize(columns);
  data_category_device.resize(columns_category);
  data_reduced_u8.resize(columns);
  data_reduced_u16.resize(columns);
  data_reduced_u32.resize(columns);
  data_reduced_u8_device.resize(columns);
  data_reduced_u16_device.resize(columns);
  data_reduced_u32_device.resize(columns);

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
  if (hist_size < (1 << 8))
    this->InitHistInternal<unsigned char>(hist_size, verbose);
  else
    this->InitHistInternal<unsigned short>(hist_size, verbose);
}

template <typename T>
void DataMatrix::InitHistInternal(int hist_size, bool verbose) {
  if (!_init) {
    thrust::host_vector<thrust::host_vector<float>> thresholds(columns_dense);
    thrust::device_vector<float> d_data(rows);
    thrust::device_vector<float> d_data_sorted(rows);
    thrust::device_vector<T> bin(rows);
    thrust::device_vector<float> d_threshold(hist_size);

    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;

    cub::DeviceRadixSort::SortKeys(
      d_temp_storage, temp_storage_bytes,
      thrust::raw_pointer_cast(d_data.data()),
      thrust::raw_pointer_cast(d_data_sorted.data()), rows);

    size_t temp_storage_bytes_scan = 0;

    OK(cub::DeviceScan::InclusiveScan(
      NULL, temp_storage_bytes_scan, ValueSegmentIterator(0, nullptr, 0),
      ThresholdOutputIterator(nullptr, 0), SegmentLength(), rows));

    temp_storage_bytes = max(temp_storage_bytes_scan, temp_storage_bytes);

    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    unsigned segment_size = rows / hist_size;
    segment_size = max(1, segment_size);

    for (size_t i = 0; i < columns_dense; ++i) {
      thrust::copy(data[i].begin(), data[i].end(), d_data.begin());

      cub::DeviceRadixSort::SortKeys(
        d_temp_storage, temp_storage_bytes,
        thrust::raw_pointer_cast(d_data.data()),
        thrust::raw_pointer_cast(d_data_sorted.data()), rows);

      thrust::fill(d_threshold.begin(), d_threshold.end(), INFINITY);

      ValueSegmentIterator in(segment_size,
                              thrust::raw_pointer_cast(d_data_sorted.data()));
      ThresholdOutputIterator out(thrust::raw_pointer_cast(d_threshold.data()),
                                  segment_size);
      SegmentLength op;

      OK(cub::DeviceScan::InclusiveScan(d_temp_storage, temp_storage_bytes, in,
                                        out, op, rows));

      data_reduced_mapping[i].resize(hist_size);

      thrust::copy(d_threshold.begin(), d_threshold.end(),
                   data_reduced_mapping[i].begin());

      int unique_size = 0;
      for (unique_size = 0; unique_size < data_reduced_mapping[i].size() &&
                            !std::isinf(data_reduced_mapping[i][unique_size]);
           unique_size++) {
      }
      unique_size++;

      int size = std::min(unique_size, hist_size);
      reduced_size[i] = 32 - __builtin_clz(size);
      int grid_size = (rows + 1024 * ITEMS - 1) / (1024 * ITEMS);
      build_histogram<T, ITEMS><<<grid_size, 1024, hist_size * sizeof(float)>>>(
        thrust::raw_pointer_cast(bin.data()),
        thrust::raw_pointer_cast(d_threshold.data()),
        thrust::raw_pointer_cast(d_data.data()), hist_size, unique_size, rows);

      OK(cudaDeviceSynchronize());

      GetHostData<T>(i).resize(rows);
      thrust::copy(bin.begin(), bin.end(), GetHostData<T>(i).begin());
    }

    OK(cudaFree(d_temp_storage));

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
    data_reduced_u32.resize(columns);
#pragma omp parallel for
    for (size_t i = 0; i < columns_dense; ++i) {
      data_reduced_u32[i].resize(rows);

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
        data_reduced_u32[i][j] = idx;
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
  if (this->max_feature_size <= sizeof(unsigned char) * CHAR_BIT) {
    this->TransferToGPUInternal<unsigned char>(free, verbose);
  } else if (this->max_feature_size <= sizeof(unsigned short) * CHAR_BIT) {
    this->TransferToGPUInternal<unsigned short>(free, verbose);
  } else {
    this->TransferToGPUInternal<unsigned int>(free, verbose);
  }
}

template <typename T>
void DataMatrix::TransferToGPUInternal(size_t free, bool verbose) {
  size_t data_size = sizeof(T) * rows;
  size_t copy_count = std::min(free / data_size, columns_dense);
  for (size_t i = 0; i < copy_count; ++i) {
    this->GetDeviceData<T>(i).resize(rows);
    thrust::copy(this->GetHostData<T>(i).begin(), this->GetHostData<T>(i).end(),
                 this->GetDeviceData<T>(i).begin());
  }
  if (verbose)
    printf("copied features data %ld from %ld \n", copy_count, columns_dense);

  //   free -= copy_count * data_size;

  //   copy_count = 0;

  //   for (size_t i = 0; i < columns_category; ++i) {
  //     if (rows * sizeof(unsigned int) < free) {
  //       copy_count++;
  //       data_category_device[i].resize(rows);
  //       thrust::copy(data_categories[i].begin(), data_categories[i].end(),
  //                    data_category_device[i].begin());
  //       free -= rows * sizeof(unsigned int);
  //     } else {
  //       break;
  //     }
  //   }
  //   if (verbose)
  //     printf("copied category features %ld from %ld \n", copy_count,
  //            columns_category);
}

template <>
thrust::host_vector<unsigned int,
                    thrust::cuda::experimental::pinned_allocator<unsigned int>>
  &DataMatrix::GetHostData(int column) {
  return data_reduced_u32[column];
}

template <>
thrust::host_vector<
  unsigned short, thrust::cuda::experimental::pinned_allocator<unsigned short>>
  &DataMatrix::GetHostData(int column) {
  return data_reduced_u16[column];
}

template <>
thrust::host_vector<unsigned char,
                    thrust::cuda::experimental::pinned_allocator<unsigned char>>
  &DataMatrix::GetHostData(int column) {
  return data_reduced_u8[column];
}

template <>
thrust::device_vector<unsigned int> &DataMatrix::GetDeviceData(int column) {
  return data_reduced_u32_device[column];
}

template <>
thrust::device_vector<unsigned short> &DataMatrix::GetDeviceData(int column) {
  return data_reduced_u16_device[column];
}

template <>
thrust::device_vector<unsigned char> &DataMatrix::GetDeviceData(int column) {
  return data_reduced_u8_device[column];
}

}  // namespace io
}  // namespace arboretum
