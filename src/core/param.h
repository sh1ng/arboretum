#ifndef PARAM_H
#define PARAM_H

#include "json.hpp"

namespace arboretum {
namespace core {
using nlohmann::json;

enum Objective {
  LinearRegression,
  LogisticRegression,
  LogisticRegressionNoSigmoind,
  SoftMaxOneVsAll,
  SoftMaxOptimal
};

enum EvalMetric { RMSE, ROC_AUC };

template <typename T> struct ThreadSpecific {
  T data;
  int thread;
};

struct TreeParam {
public:
  static TreeParam Parse(const json &cfg) {
    int objective = cfg.value("/objective"_json_pointer, -1);
    float eta = cfg.value("/tree/eta"_json_pointer, 0.3);
    unsigned int max_depth = cfg.value("/tree/max_depth"_json_pointer, 8);
    float gamma = cfg.value("/tree/gamma"_json_pointer, 0.0);
    float min_child_weight =
        cfg.value("/tree/min_child_weight"_json_pointer, 1.0);
    unsigned int min_leaf_size =
        cfg.value("/tree/min_leaf_size"_json_pointer, 1);
    float lambda = cfg.value("/tree/lambda"_json_pointer, 1.0);
    float alpha = cfg.value("/tree/alpha"_json_pointer, 0.0);
    float colsample_bytree =
        cfg.value("/tree/colsample_bytree"_json_pointer, 1.0);
    float colsample_bylevel =
        cfg.value("/tree/colsample_bylevel"_json_pointer, 1.0);
    float initial = cfg.value("/tree/initial_y"_json_pointer, 0.5);
    float max_leaf_weight =
        cfg.value("/tree/max_leaf_weight"_json_pointer, 0.0);
    float scale_pos_weight =
        cfg.value("/tree/scale_pos_weight"_json_pointer, 0.5);
    unsigned char labels_count =
        cfg.value("/tree/labels_count"_json_pointer, 1);

    return TreeParam((Objective)objective, max_depth, min_child_weight,
                     min_leaf_size, colsample_bytree, colsample_bylevel, gamma,
                     lambda, alpha, initial, eta, max_leaf_weight,
                     scale_pos_weight, labels_count);
  }

  TreeParam(Objective objective, int depth, float min_child_weight,
            unsigned int min_leaf_size, float colsample_bytre,
            float colsample_bylevel, float gamma, float lambda, float alpha,
            float initial_y, float eta, float max_leaf_weight,
            float scale_pos_weight, unsigned short labels_count)
      : objective(objective), depth(depth), min_child_weight(min_child_weight),
        min_leaf_size(min_leaf_size), colsample_bytree(colsample_bytre),
        colsample_bylevel(colsample_bylevel), gamma(gamma), lambda(lambda),
        alpha(alpha), initial_y(initial_y), eta(eta),
        max_leaf_weight(max_leaf_weight), scale_pos_weight(scale_pos_weight),
        labels_count(labels_count) {}
  const Objective objective;
  const unsigned int depth;
  const float min_child_weight;
  const unsigned int min_leaf_size;
  const float colsample_bytree;
  const float colsample_bylevel;
  const float gamma;
  const float lambda;
  const float alpha;
  const float initial_y;
  const float eta;
  const float max_leaf_weight;
  const float scale_pos_weight;
  const unsigned char labels_count;
};

struct Verbose {
public:
  static Verbose Parse(const json &cfg) {
    return Verbose(cfg.value("/verbose/gpu"_json_pointer, false),
                   cfg.value("/verbose/booster"_json_pointer, false),
                   cfg.value("/verbose/data"_json_pointer, false));
  }
  Verbose(bool gpu, bool booster, bool data)
      : gpu(gpu), booster(booster), data(data) {}
  const bool gpu;
  const bool booster;
  const bool data;
};

struct InternalConfiguration {
public:
  static InternalConfiguration Parse(const json &cfg) {
    return InternalConfiguration(
        cfg.value("/internals/double_precision"_json_pointer, false),
        cfg.value("/internals/compute_overlap"_json_pointer, 2),
        cfg.value("/internals/seed"_json_pointer, 0));
  }
  InternalConfiguration(bool double_precision, unsigned short overlap,
                        unsigned int seed)
      : double_precision(double_precision), overlap(overlap), seed(seed) {}
  const bool double_precision;
  const unsigned short overlap;
  const unsigned int seed;
};
}
}

#endif // PARAM_H
