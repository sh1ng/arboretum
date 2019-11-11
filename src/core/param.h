#ifndef PARAM_H
#define PARAM_H

namespace arboretum {
namespace core {

enum Objective {
  LinearRegression,
  LogisticRegression,
  LogisticRegressionNoSigmoind,
  SoftMaxOneVsAll,
  SoftMaxOptimal
};

enum Method { Exact, Hist };

enum EvalMetric { RMSE, ROC_AUC };

template <typename T>
struct ThreadSpecific {
  T data;
  int thread;
};

struct TreeParam {
 public:
  static TreeParam Parse(const char *configuration);

  TreeParam(Method method, Objective objective, int depth,
            float min_child_weight, unsigned int min_leaf_size,
            float colsample_bytre, float colsample_bylevel,
            float gamma_absolute, float gamma_relative, float lambda,
            float alpha, float initial_y, float eta, float max_leaf_weight,
            float scale_pos_weight, unsigned short labels_count,
            unsigned hist_size);
  const Method method;
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
  const unsigned hist_size;
};

struct Verbose {
 public:
  static Verbose Parse(const char *configuration);
  Verbose(bool gpu, bool booster, bool data);
  const bool gpu;
  const bool booster;
  const bool data;
};

struct InternalConfiguration {
 public:
  static InternalConfiguration Parse(const char *configuration);
  InternalConfiguration(bool double_precision = false,
                        unsigned short overlap = 2, unsigned int seed = 0,
                        bool use_hist_subtraction_trick = true,
                        bool upload_features = true);
  const bool double_precision;
  const unsigned short overlap;
  const unsigned int seed;
  const bool use_hist_subtraction_trick;
  const bool upload_features;
};
}  // namespace core
}  // namespace arboretum

#endif  // PARAM_H
