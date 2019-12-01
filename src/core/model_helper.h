#ifndef SRC_CORE_MODEL_HELPER_H
#define SRC_CORE_MODEL_HELPER_H

#include <stdio.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>
#include "param.h"
#include "reg_tree.h"

namespace arboretum {
namespace core {
using namespace arboretum;

const char *DumpModel(const Configuration &cfg,
                      const std::vector<DecisionTree> &trees);

std::vector<DecisionTree> LoadModel(const char *model);

}  // namespace core
}  // namespace arboretum

#endif