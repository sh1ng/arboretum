#ifndef SRC_CORE_PARTITION_H
#define SRC_CORE_PARTITION_H
// #define CUB_STDERR
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/system/cuda/experimental/pinned_allocator.h>
#include "best_splits.h"
#include "common.h"
#include "cub/cub.cuh"
#include "cub/iterator/discard_output_iterator.cuh"
#include "cuda_helpers.h"
#include "cuda_runtime.h"
#include "histogram.h"
#include "param.h"

namespace arboretum {
namespace core {

struct Position {
  const unsigned position;
  Position(const unsigned position) : position(position) {}
  __host__ __device__ __forceinline__ uint2 operator()(const uint2 &a,
                                                       const uint2 &b) const {
    unsigned segment_a = a.y >> (position + 2);
    segment_a = segment_a << 1;
    unsigned segment_b = b.y >> (position + 2);
    segment_b = segment_b << 1;
    if (segment_a == segment_b) {
      return make_uint2(a.x + b.x, b.y);
    }
    return b;
  }
};

class PartitioningIndexIterator {
 public:
  using self_type = PartitioningIndexIterator;
  using difference_type = unsigned;
  using value_type = void;
  using pointer = value_type *;
  using reference = unsigned;
  using iterator_category = typename thrust::detail::iterator_facade_category<
    thrust::any_system_tag, thrust::random_access_traversal_tag, value_type,
    reference>::type;

 private:
  difference_type offset_;
  const unsigned *node_size;
  unsigned *index;
  const int position;

 public:
  __host__ __device__ __forceinline__
  PartitioningIndexIterator(const unsigned *node_size, unsigned *index,
                            const int position, difference_type offset = 0)
      : node_size(node_size),
        index(index),
        position(position),
        offset_(offset) {}

  /// Addition
  template <typename Distance>
  __device__ __forceinline__ self_type operator+(Distance n) const {
    self_type retval(node_size, index, position, offset_ + n);
    return retval;
  }

  /// Array subscript
  template <typename Distance>
  __host__ __device__ __forceinline__ self_type operator[](Distance n) const {
    self_type retval(node_size, index, position, offset_ + n);
    return retval;
  }

  __device__ __forceinline__ void operator=(uint2 const &v) {
    unsigned segment = v.y >> (position + 2);
    segment = segment << 1;
    unsigned idx;
    if ((v.y >> position) % 2 == 0)
      idx = node_size[segment] + v.x - 1;
    else
      idx = node_size[segment + 2] - (offset_ - node_size[segment] - v.x + 1);

    index[offset_] = idx;
  }
};

template <typename T>
class PartitioningIterator {
 public:
  using self_type = PartitioningIterator;
  using difference_type = unsigned;
  using value_type = void;
  using pointer = value_type *;
  using reference = T;
  using iterator_category = typename thrust::detail::iterator_facade_category<
    thrust::any_system_tag, thrust::random_access_traversal_tag, value_type,
    reference>::type;

 private:
  difference_type offset_;
  const unsigned *node_size;
  const T *in;
  T *out;
  const int position;

 public:
  __host__ __device__ __forceinline__
  PartitioningIterator(const unsigned *node_size, const T *in, T *out,
                       const int position, difference_type offset = 0)
      : node_size(node_size),
        in(in),
        out(out),
        position(position),
        offset_(offset) {}

  /// Addition
  template <typename Distance>
  __device__ __forceinline__ self_type operator+(Distance n) const {
    self_type retval(node_size, in, out, position, offset_ + n);
    return retval;
  }

  /// Array subscript
  template <typename Distance>
  __host__ __device__ __forceinline__ self_type operator[](Distance n) const {
    self_type retval(node_size, in, out, position, offset_ + n);
    return retval;
  }

  __device__ __forceinline__ void operator=(uint2 const &v) {
    unsigned segment = v.y >> (position + 2);
    segment = segment << 1;
    unsigned idx;
    if ((v.y >> position) % 2 == 0)
      idx = node_size[segment] + v.x - 1;
    else
      idx = node_size[segment + 2] - (offset_ - node_size[segment] - v.x + 1);

    out[idx] = in[offset_];
  }
};

template <typename InputIteratorT>
class SegmentedInputIterator {
 public:
  typedef typename std::iterator_traits<InputIteratorT>::value_type input_type;
  // Required iterator traits
  typedef SegmentedInputIterator self_type;  ///< My own type
  typedef unsigned difference_type;          ///< Type to express the result of
                                     ///< subtracting one iterator from another
  typedef uint2
    value_type;  ///< The type of the element the iterator can point to
  typedef value_type *pointer;   ///< The type of a pointer to an element
                                 ///< the iterator can point to
  typedef value_type reference;  ///< The type of a reference to an
                                 ///< element the iterator can point to

  typedef typename thrust::detail::iterator_facade_category<
    thrust::any_system_tag, thrust::random_access_traversal_tag, value_type,
    reference>::type iterator_category;  ///< The iterator category

 private:
  InputIteratorT row2node;
  int position;
  difference_type offset;

 public:
  /// Constructor
  __host__ __device__ __forceinline__ SegmentedInputIterator(
    InputIteratorT row2node,  ///< Input iterator to wrap
    int position,
    difference_type offset = 0)  ///< OffsetT (in items) from \p itr denoting
                                 ///< the position of the iterator
      : row2node(row2node), position(position), offset(offset) {}

  /// Indirection
  __host__ __device__ __forceinline__ reference operator*() const {
    input_type value = row2node[offset];
    // unsigned segment = value >> (position + 1);
    // segment = (segment >> 1) << 1;

    uint2 retval = make_uint2((value >> position) % 2 == 0, value);
    return retval;
  }

  /// Addition
  template <typename Distance>
  __host__ __device__ __forceinline__ self_type operator+(Distance n) const {
    self_type retval(row2node, position, offset + n);
    return retval;
  }

  /// Array subscript
  template <typename Distance>
  __host__ __device__ __forceinline__ reference operator[](Distance n) const {
    self_type retval(row2node, position, offset + n);
    return *retval;
  }

  /// ostream operator
  friend std::ostream &operator<<(std::ostream &os, const self_type & /*itr*/) {
    return os;
  }
};
}  // namespace core
}  // namespace arboretum

#endif