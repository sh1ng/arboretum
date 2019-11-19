#ifndef SRC_CORE_MODEL_HELPER_H
#define SRC_CORE_MODEL_HELPER_H

#include <stdio.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include "param.h"
#include "reg_tree.h"

namespace arboretum {
namespace core {
using namespace arboretum;

const char *DumpModel(const Configuration &cfg,
                      const std::vector<DecisionTree> &trees);

}  // namespace core
}  // namespace arboretum

#endif