#include "model_helper.h"
#include <stdio.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include "json.hpp"
#include "json_conversion.h"
#include "param.h"
#include "reg_tree.h"

namespace arboretum {
namespace core {
using namespace arboretum;
using nlohmann::json;

const char *DumpModel(const Configuration &cfg,
                      const std::vector<DecisionTree> &trees) {
  json j = json{{"configuration", cfg}, {"model", trees}};
  auto str = j.dump();
  char *cstr = new char[str.length() + 1];
  std::strcpy(cstr, str.c_str());
  return cstr;
}

std::vector<DecisionTree> LoadModel(const char *model) {
  json j = json::parse(model);
  std::vector<DecisionTree> trees;
  j.at("model").get_to(trees);
  return trees;
}

}  // namespace core
}  // namespace arboretum