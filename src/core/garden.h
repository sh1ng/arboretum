#ifndef BOOSTER_H
#define BOOSTER_H

#include <stdio.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include "../io/io.h"
#include "objective.h"
#include "param.h"

namespace arboretum {
namespace core {
using namespace arboretum;

inline __host__ __device__ float ThresholdWeight(const float w,
                                                 const TreeParam &param) {
  if (param.max_leaf_weight != 0.0f) {
    if (w > param.max_leaf_weight) return param.max_leaf_weight;
    if (w < -param.max_leaf_weight) return -param.max_leaf_weight;
  }
  return w;
}

inline __host__ __device__ float Weight(const float s, const unsigned int c,
                                        const TreeParam &param) {
  float w = 0.0;
  if (c >= param.min_leaf_size && c >= param.min_child_weight)
    w = s / (c + param.lambda);
  return ThresholdWeight(w, param);
}

inline __host__ __device__ float Weight(const double s, const unsigned int c,
                                        const TreeParam &param) {
  float w = 0.0;
  if (c >= param.min_leaf_size && c >= param.min_child_weight)
    w = s / (c + param.lambda);
  return ThresholdWeight(w, param);
}

inline __host__ __device__ float Weight(const float2 s, const unsigned int c,
                                        const TreeParam &param) {
  float w = 0.0;
  if (c >= param.min_leaf_size && s.y >= param.min_child_weight)
    w = s.x / (s.y + param.lambda);
  return ThresholdWeight(w, param);
}
inline __host__ __device__ float Weight(const double2 s, const unsigned int c,
                                        const TreeParam &param) {
  float w = 0.0;
  if (c >= param.min_leaf_size && s.y >= param.min_child_weight)
    w = s.x / (s.y + param.lambda);
  return ThresholdWeight(w, param);
}

template <class grad_type>
struct Split {
  float split_value;
  unsigned category;
  int fid;
  double gain;
  grad_type sum_grad;
  unsigned count;
  unsigned quantized;
  Split() { Clean(); }
  void Clean() {
    fid = category = (unsigned int)-1;
    gain = 0.0;
    init(sum_grad);
    count = 0;
    split_value = std::numeric_limits<float>::infinity();
    quantized = (unsigned)-1;
  }
  inline float LeafWeight(const TreeParam &param) const {
    return Weight(sum_grad, count, param);
  }

  inline float LeafWeight(const unsigned parent_size,
                          const grad_type parent_sum,
                          const TreeParam &param) const {
    return Weight(parent_sum - sum_grad, parent_size - count, param);
  }
};

template <class grad_type>
struct NodeStat {
  unsigned int count;
  grad_type sum_grad;
  double gain;

  NodeStat() { Clean(); }

  void Clean() {
    count = 0;
    gain = 0.0;
    init(sum_grad);
  }
};

struct Node {
  static inline unsigned int Left(unsigned int parent) {
    return 2 * parent + 1;
  }

  static inline unsigned int Right(unsigned int parent) {
    return 2 * parent + 2;
  }

  static inline unsigned int HeapOffset(unsigned int level) {
    return (1 << level) - 1;
  }

  Node(int id) : id(id), fid(0), category((unsigned int)-1) {}

  unsigned id;
  float threshold;
  unsigned int fid;
  unsigned int category;
  unsigned quantized;
};

struct RegTree {
  static std::vector<int> InitLeft(unsigned int depth) {
    std::vector<int> tmp(1 << depth);
    for (int i = 0; i < (1 << depth); ++i) {
      tmp[i] = Node::Left(i);
    }
    return tmp;
  }

  static std::vector<int> InitRight(unsigned int depth) {
    std::vector<int> tmp(1 << depth);
    for (int i = 0; i < (1 << depth); ++i) {
      tmp[i] = Node::Right(i);
    }
    return tmp;
  }

  std::vector<Node> nodes;
  const unsigned int depth;
  const unsigned int offset;
  const unsigned short label;
  std::vector<float> leaf_level;
  std::vector<int> _node_lookup[2];

  RegTree(unsigned int depth, unsigned short label)
      : depth(depth), offset((1 << (depth - 1)) - 1), label(label) {
    unsigned int nodes_num = (1 << depth) - 1;
    nodes.reserve(nodes_num);
    _node_lookup[0] = RegTree::InitRight(depth);
    _node_lookup[1] = RegTree::InitLeft(depth);

    for (size_t i = 0; i < nodes_num; ++i) {
      nodes.push_back(Node(i));
    }
    leaf_level.resize(1 << (depth - 1));
  }

  inline int ChildNode(const unsigned int parent, const bool isLeft) const {
    return _node_lookup[isLeft][parent];
  }

  void Predict(const arboretum::io::DataMatrix *data,
               thrust::host_vector<float> &out) const {
#pragma omp parallel
    {
#pragma omp for simd
      for (size_t i = 0; i < data->rows; ++i) {
        unsigned int node_id = 0;
        // todo: check
        Node current_node = nodes[node_id];
        for (size_t j = 1, len = depth; j < len; ++j) {
          current_node = nodes[node_id];
          bool isLeft =
            (current_node.fid < data->columns_dense &&
             data->data[current_node.fid][i] < current_node.threshold) ||
            (current_node.fid >= data->columns_dense &&
             data->data_categories[current_node.fid - data->columns_dense][i] ==
               current_node.category);

          node_id = ChildNode(node_id, isLeft);
        }
        out[i + label * data->rows] += leaf_level[node_id - offset];
      }
    }
  }

  void PredictByQuantized(const arboretum::io::DataMatrix *data,
                          thrust::host_vector<float> &out) const {
#pragma omp parallel
    {
#pragma omp for simd
      for (size_t i = 0; i < data->rows; ++i) {
        unsigned int node_id = 0;
        // todo: check
        Node current_node = nodes[node_id];
        for (size_t j = 1, len = depth; j < len; ++j) {
          current_node = nodes[node_id];
          bool isLeft =
            current_node.fid < data->columns_dense &&
            data->data_reduced[current_node.fid][i] < current_node.quantized;
          // FIXME: support category
          //    ||
          //   (current_node.fid >= data->columns_dense &&
          //    data->data_categories[current_node.fid - data->columns_dense][i]
          //    ==
          //      current_node.category);

          node_id = ChildNode(node_id, isLeft);
        }
        out[i + label * data->rows] += leaf_level[node_id - offset];
      }
    }
  }

  void Predict(const arboretum::io::DataMatrix *data,
               const thrust::host_vector<size_t> &row2Node,
               thrust::host_vector<float> &out) const {
#pragma omp parallel
    {
#pragma omp for simd
      for (size_t i = 0; i < data->rows; ++i) {
        out[i + label * data->rows] += leaf_level[row2Node[i]];
      }
    }
  }
};

class GardenBuilderBase {
 public:
  virtual ~GardenBuilderBase() {}
  virtual void InitGrowingTree(const size_t columns) = 0;
  virtual void InitTreeLevel(const int level, const size_t columns) = 0;
  virtual void GrowTree(RegTree *tree, io::DataMatrix *data,
                        const unsigned short label) = 0;
  virtual void PredictByGrownTree(RegTree *tree, io::DataMatrix *data,
                                  thrust::host_vector<float> &out) const = 0;
};

class Garden {
 public:
  Garden(const TreeParam &param, const Verbose &verbose,
         const InternalConfiguration &cfg);
  ~Garden();

  const TreeParam param;
  const Verbose verbose;
  const InternalConfiguration cfg;
  void GrowTree(io::DataMatrix *data, float *grad);
  void Predict(const arboretum::io::DataMatrix *data,
               std::vector<float> &out) const;
  void UpdateByLastTree(arboretum::io::DataMatrix *data);
  void GetY(arboretum::io::DataMatrix *data, std::vector<float> &out) const;
  const char *GetModel() const;

 private:
  bool _init;
  GardenBuilderBase *_builder;
  ApproximatedObjectiveBase *_objective;
  std::vector<RegTree *> _trees;
};
}  // namespace core
}  // namespace arboretum

#endif  // BOOSTER_H
