#ifndef SRC_CORE_GARDEN_H
#define SRC_CORE_GARDEN_H

#include <stdio.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include "../io/io.h"
#include "objective.h"
#include "param.h"
#include "reg_tree.h"

namespace arboretum {
namespace core {
using namespace arboretum;

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

struct RegTree : public DecisionTree {
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

  const unsigned int offset;
  const unsigned short label;
  std::vector<int> _node_lookup[2];

  RegTree(unsigned int depth, unsigned short label)
      : DecisionTree(depth), offset((1 << (depth - 1)) - 1), label(label) {
    unsigned int nodes_num = (1 << depth) - 1;
    nodes.reserve(nodes_num);
    _node_lookup[0] = RegTree::InitRight(depth);
    _node_lookup[1] = RegTree::InitLeft(depth);

    for (size_t i = 0; i < nodes_num; ++i) {
      nodes.push_back(Node(i, depth));
    }
    weights.resize(1 << (depth - 1));
  }

  inline int ChildNode(const unsigned int parent, const bool isLeft) const {
    return _node_lookup[isLeft][parent];
  }

  void Predict(const arboretum::io::DataMatrix *data,
               thrust::host_vector<float> &out) const {
#pragma omp parallel for
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
      out[i + label * data->rows] += weights[node_id - offset];
    }
  }

  void PredictByQuantized(const arboretum::io::DataMatrix *data,
                          thrust::host_vector<float> &out) const {
    // #pragma omp parallel for
    //     for (size_t i = 0; i < data->rows; ++i) {
    //       unsigned int node_id = 0;
    //       // todo: check
    //       Node current_node = nodes[node_id];
    //       for (size_t j = 1, len = depth; j < len; ++j) {
    //         current_node = nodes[node_id];
    //         bool isLeft =
    //           current_node.fid < data->columns_dense &&
    //           data->data_reduced[current_node.fid][i] <
    //           current_node.quantized;
    //         // FIXME: support category
    //         //    ||
    //         //   (current_node.fid >= data->columns_dense &&
    //         //    data->data_categories[current_node.fid -
    //         data->columns_dense][i]
    //         //    ==
    //         //      current_node.category);

    //         node_id = ChildNode(node_id, isLeft);
    //       }
    //       out[i + label * data->rows] += weights[node_id - offset];
    //     }
  }

  void Predict(const arboretum::io::DataMatrix *data,
               const thrust::host_vector<size_t> &row2Node,
               thrust::host_vector<float> &out) const {
#pragma omp parallel for
    for (size_t i = 0; i < data->rows; ++i) {
      out[i + label * data->rows] += weights[row2Node[i]];
    }
  }
};  // namespace core

class GardenBuilderBase {
 public:
  virtual ~GardenBuilderBase() {}
  virtual void InitGrowingTree(const size_t columns) = 0;
  virtual void InitTreeLevel(const int level, const size_t columns) = 0;
  virtual void GrowTree(RegTree *tree, io::DataMatrix *data,
                        const unsigned short label) = 0;
  virtual void PredictByGrownTree(RegTree *tree, io::DataMatrix *data,
                                  thrust::host_vector<float> &out) const = 0;
  virtual void UpdateGrad() = 0;
};

class Garden {
 public:
  Garden(const Configuration &cfg);
  ~Garden();

  const Configuration cfg;
  void GrowTree(io::DataMatrix *data, float *grad);
  void Predict(const arboretum::io::DataMatrix *data, std::vector<float> &out,
               const int n_rounds = INT_MAX) const;
  void UpdateByLastTree(arboretum::io::DataMatrix *data);
  void GetY(arboretum::io::DataMatrix *data, std::vector<float> &out) const;
  const char *GetModel() const;
  void Restore(const char *json_model);

 private:
  bool _init;
  GardenBuilderBase *_builder;
  ApproximatedObjectiveBase *_objective;
  std::vector<RegTree *> _trees;
};
}  // namespace core
}  // namespace arboretum

#endif  // BOOSTER_H
