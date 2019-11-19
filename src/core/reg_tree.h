#ifndef SRC_CORE_REG_TREE_H
#define SRC_CORE_REG_TREE_H

#include <stdio.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>
#include "param.h"

namespace arboretum {
namespace core {
using namespace arboretum;

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

  Node(int id = -1, unsigned depth = 0)
      : id(id), depth(depth), fid(0), category((unsigned int)-1) {}

  unsigned id;
  unsigned depth;
  float threshold;
  unsigned int fid;
  unsigned int category;
  unsigned quantized;
};

struct DecisionTree {
  std::vector<Node> nodes;
  std::vector<float> weights;
};

}  // namespace core
}  // namespace arboretum

#endif