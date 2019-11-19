#include "param.h"
#include "iostream"
#include "json.hpp"
#include "json_conversion.h"

namespace arboretum {
namespace core {
using nlohmann::json;

Verbose::Verbose(bool gpu, bool booster, bool data)
    : gpu(gpu), booster(booster), data(data) {}

TreeParam::TreeParam(Method method, Objective objective, int depth,
                     float min_child_weight, unsigned int min_leaf_size,
                     float colsample_bytree, float colsample_bylevel,
                     float gamma_absolute, float gamma_relative, float lambda,
                     float alpha, float initial_y, float eta,
                     float max_leaf_weight, float scale_pos_weight,
                     unsigned short labels_count)
    : method(method),
      objective(objective),
      depth(depth),
      min_child_weight(min_child_weight),
      min_leaf_size(min_leaf_size),
      colsample_bytree(colsample_bytree),
      colsample_bylevel(colsample_bylevel),
      gamma_absolute(gamma_absolute),
      gamma_relative(gamma_relative),
      lambda(lambda),
      alpha(alpha),
      initial_y(initial_y),
      eta(eta),
      max_leaf_weight(max_leaf_weight),
      scale_pos_weight(scale_pos_weight),
      labels_count(labels_count) {}

InternalConfiguration::InternalConfiguration(
  bool double_precision, unsigned short overlap, unsigned int seed,
  bool use_hist_subtraction_trick, bool upload_features, int hist_size)
    : double_precision(double_precision),
      overlap(overlap),
      seed(seed),
      use_hist_subtraction_trick(use_hist_subtraction_trick),
      upload_features(upload_features),
      hist_size(hist_size) {}

Configuration Configuration::Parse(const char *configuration) {
  const nlohmann::json js = json::parse(configuration);
  const Configuration cfg = js.get<Configuration>();
  return cfg;
}

}  // namespace core
}  // namespace arboretum
