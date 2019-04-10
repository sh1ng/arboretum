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
  static TreeParam Parse(const json &cfg);

  TreeParam(Objective objective, int depth, float min_child_weight,
            unsigned int min_leaf_size, float colsample_bytre,
            float colsample_bylevel, float gamma_absolute, float gamma_relative,
            float lambda, float alpha, float initial_y, float eta,
            float max_leaf_weight, float scale_pos_weight,
            unsigned short labels_count);

  const Objective objective;
  const unsigned int depth;
  const float min_child_weight;
  const unsigned int min_leaf_size;
  const float colsample_bytree;
  const float colsample_bylevel;
  const float gamma_absolute;
  const float gamma_relative;
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
  static InternalConfiguration Parse(const json &cfg);
  InternalConfiguration(bool double_precision, unsigned short overlap,
                        unsigned int seed);
  const bool double_precision;
  const unsigned short overlap;
  const unsigned int seed;
};
} // namespace core
} // namespace arboretum

#endif // PARAM_H
