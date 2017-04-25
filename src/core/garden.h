#ifndef BOOSTER_H
#define BOOSTER_H

#include "../io/io.h"
#include "cuda_helpers.h"
#include "param.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdio.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/system/cuda/experimental/pinned_allocator.h>

namespace arboretum {
namespace core {
using namespace arboretum;

template <class grad_type> struct Split {
  float split_value;
  unsigned int category;
  int fid;
  double gain;
  grad_type sum_grad;
  unsigned int count;
  Split() { Clean(); }
  void Clean() {
    fid = category = (unsigned int)-1;
    gain = 0.0;
    init(sum_grad);
    count = 0;
    split_value = std::numeric_limits<float>::infinity();
  }
  inline float LeafWeight(const TreeParam &param) const {
    const float w = Split<grad_type>::LeafWeight(sum_grad, count, param);
    if (param.max_leaf_weight != 0.0f) {
      if (w > param.max_leaf_weight)
        return param.max_leaf_weight;
      if (w < -param.max_leaf_weight)
        return -param.max_leaf_weight;
    }
    return w;
  }

  template <class node_stat>
  inline float LeafWeight(node_stat &parent, const TreeParam &param) const {
    const float w = Split<grad_type>::LeafWeight(parent.sum_grad - sum_grad,
                                                 parent.count - count, param);
    if (param.max_leaf_weight != 0.0f) {
      if (w > param.max_leaf_weight)
        return param.max_leaf_weight;
      if (w < -param.max_leaf_weight)
        return -param.max_leaf_weight;
    }
    return w;
  }

private:
  static inline float LeafWeight(const float s, const unsigned int c,
                                 const TreeParam &param) {
    return s / (c + param.lambda);
  }
  static inline float LeafWeight(const float2 s, const unsigned int c,
                                 const TreeParam &param) {
    return s.x / (s.y + param.lambda);
  }
  static inline float LeafWeight(const double2 s, const unsigned int c,
                                 const TreeParam &param) {
    return s.x / (s.y + param.lambda);
  }
};

template <class grad_type> struct NodeStat {
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
               std::vector<float> &out) const {
#pragma omp parallel for simd
    for (size_t i = 0; i < data->rows; ++i) {
      unsigned int node_id = 0;
      // todo: check
      Node current_node = nodes[node_id];
      for (size_t j = 1, len = depth; j < len; ++j) {
        current_node = nodes[node_id];
        bool isLeft =
            (current_node.fid < data->columns_dense &&
             data->data[current_node.fid][i] <= current_node.threshold) ||
            (current_node.fid >= data->columns_dense &&
             data->data_categories[current_node.fid - data->columns_dense][i] ==
                 current_node.category);

        node_id = ChildNode(node_id, isLeft);
      }
      out[i + label * data->rows] += leaf_level[node_id - offset];
    }
  }

  void Predict(const arboretum::io::DataMatrix *data,
               const thrust::host_vector<size_t> &row2Node,
               std::vector<float> &out) const {

#pragma omp parallel for simd
    for (size_t i = 0; i < data->rows; ++i) {
      out[i + label * data->rows] += leaf_level[row2Node[i]];
    }
  }
};

class ApproximatedObjectiveBase {
public:
  ApproximatedObjectiveBase(io::DataMatrix *data) : data(data) {}
  virtual ~ApproximatedObjectiveBase() {}
  const io::DataMatrix *data;
  virtual void UpdateGrad() = 0;
  virtual float IntoInternal(float v) { return v; }
  virtual inline void FromInternal(std::vector<float> &in,
                                   std::vector<float> &out) {
    out = in;
  }
};

template <class grad_type>
class ApproximatedObjective : public ApproximatedObjectiveBase {
public:
  ApproximatedObjective(io::DataMatrix *data)
      : ApproximatedObjectiveBase(data) {}
  thrust::host_vector<grad_type,
                      thrust::cuda::experimental::pinned_allocator<grad_type>>
      grad;
};

class RegressionObjective : public ApproximatedObjective<float> {
public:
  RegressionObjective(io::DataMatrix *data, float initial_y)
      : ApproximatedObjective<float>(data) {
    grad.resize(data->rows);
    data->y_internal.resize(data->rows, IntoInternal(initial_y));
  }
  virtual void UpdateGrad() override {
#pragma omp parallel for simd
    for (size_t i = 0; i < data->rows; ++i) {
      grad[i] = data->y_hat[i] - data->y_internal[i];
    }
  }
};

class LogisticRegressionObjective : public ApproximatedObjective<float2> {
public:
  LogisticRegressionObjective(io::DataMatrix *data, float initial_y)
      : ApproximatedObjective<float2>(data) {
    grad.resize(data->rows);
    data->y_internal.resize(data->rows, IntoInternal(initial_y));
  }
  virtual void UpdateGrad() override {
#pragma omp parallel for simd
    for (size_t i = 0; i < data->rows; ++i) {
      const float sigmoid = Sigmoid(data->y_internal[i]);
      grad[i].x = data->y_hat[i] - sigmoid;
      grad[i].y = sigmoid * (1.0f - sigmoid);
    }
  }
  virtual inline float IntoInternal(float v) override {
    return std::log(v / (1 - v));
  }
  virtual inline void FromInternal(std::vector<float> &in,
                                   std::vector<float> &out) override {
#pragma omp parallel for simd
    for (size_t i = 0; i < out.size(); ++i) {
      out[i] = Sigmoid(in[i]);
    }
  }

private:
  inline float Sigmoid(float x) { return 1.0 / (1.0 + std::exp(-x)); }
};

class SoftMaxObjective : public ApproximatedObjective<float2> {
public:
  SoftMaxObjective(io::DataMatrix *data, unsigned char labels_count,
                   float initial_y)
      : ApproximatedObjective<float2>(data), labels_count(labels_count) {
    grad.resize(data->rows * labels_count);
    data->y_internal.resize(data->rows * labels_count, IntoInternal(initial_y));
  }
  virtual void UpdateGrad() override {
    const float label_lookup[] = {0.0, 1.0};
#pragma omp parallel for simd
    for (size_t i = 0; i < data->rows; ++i) {
      std::vector<double> labels_prob(labels_count);

      for (unsigned char j = 0; j < labels_count; ++j) {
        labels_prob[j] = data->y_internal[i + j * data->rows];
      }

      SoftMax(labels_prob);

      const unsigned char label = data->labels[i];

      for (unsigned char j = 0; j < labels_count; ++j) {
        const double pred = labels_prob[j];
        grad[j * data->rows + i].x = label_lookup[j == label] - pred;
        grad[j * data->rows + i].y = 2.0 * pred * (1.0 - pred);
      }
    }
  }
  virtual inline float IntoInternal(float v) override { return v; }
  virtual inline void FromInternal(std::vector<float> &in,
                                   std::vector<float> &out) override {
    const size_t n = in.size() / labels_count;
#pragma omp parallel for simd
    for (size_t i = 0; i < n; ++i) {
      std::vector<double> labels_prob(labels_count);

      for (unsigned char j = 0; j < labels_count; ++j) {
        labels_prob[j] = in[i + n * j];
      }

      SoftMax(labels_prob);

      for (unsigned char j = 0; j < labels_count; ++j) {
        out[i * labels_count + j] = labels_prob[j];
      }
    }
  }

private:
  const unsigned char labels_count;
  inline float Sigmoid(float x) { return 1.0 / (1.0 + std::exp(-x)); }

  //  #pragma omp declare simd
  inline void SoftMax(std::vector<double> &values) const {
    double sum = 0.0;
    for (unsigned short i = 0; i < labels_count; ++i) {
      values[i] = std::exp(values[i]);
      sum += values[i];
    }
    for (unsigned short i = 0; i < labels_count; ++i) {
      values[i] /= sum;
    }
  }
};

class GardenBuilderBase {
public:
  virtual ~GardenBuilderBase() {}
  virtual size_t MemoryRequirementsPerRecord() = 0;
  virtual void InitGrowingTree(const size_t columns) = 0;
  virtual void InitTreeLevel(const int level, const size_t columns) = 0;
  virtual void GrowTree(RegTree *tree, const io::DataMatrix *data,
                        const unsigned short label) = 0;
  virtual void PredictByGrownTree(RegTree *tree, io::DataMatrix *data,
                                  std::vector<float> &out) const = 0;
};

class Garden {
public:
  Garden(const TreeParam &param, const Verbose &verbose,
         const InternalConfiguration &cfg);
  ~Garden() {
    if (_builder)
      delete _builder;
    if (_objective)
      delete _objective;
    for (size_t i = 0; i < _trees.size(); ++i) {
      delete _trees[i];
    }
  }

  const TreeParam param;
  const Verbose verbose;
  const InternalConfiguration cfg;
  void GrowTree(io::DataMatrix *data, float *grad);
  void Predict(const arboretum::io::DataMatrix *data,
               std::vector<float> &out) const;
  void UpdateByLastTree(arboretum::io::DataMatrix *data);
  void GetY(arboretum::io::DataMatrix *data, std::vector<float> &out) const;

private:
  bool _init;
  GardenBuilderBase *_builder;
  ApproximatedObjectiveBase *_objective;
  std::vector<RegTree *> _trees;
};
} // namespace core
} // namespace arboretum

#endif // BOOSTER_H
