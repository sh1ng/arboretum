#ifndef SRC_CORE_PARAM_H
#define SRC_CORE_PARAM_H

namespace arboretum {
namespace core {

enum Objective {
  Undefined = -1,
  LinearRegression = 0,
  LogisticRegression = 1,
  LogisticRegressionNoSigmoind = 2,
  SoftMaxOneVsAll = 3,
  SoftMaxOptimal = 4
};

enum Method { Exact, Hist };

enum EvalMetric { RMSE, ROC_AUC };

struct TreeParam {
  TreeParam(Method method = Hist, Objective objective = Undefined,
            int depth = 8, float min_child_weight = 1.0,
            unsigned int min_leaf_size = 1, float colsample_bytree = 1.0,
            float colsample_bylevel = 1.0, float gamma_absolute = 0.0,
            float gamma_relative = 0.0, float lambda = 1.0, float alpha = 0.0,
            float initial_y = 0.5, float eta = 0.3, float max_leaf_weight = 0.0,
            float scale_pos_weight = 1.0, unsigned short labels_count = 1);
  Method method;
  Objective objective;
  unsigned int depth;
  float min_child_weight;
  unsigned int min_leaf_size;
  float colsample_bytree;
  float colsample_bylevel;
  float gamma_absolute;
  float gamma_relative;
  float lambda;
  float alpha;
  float initial_y;
  float eta;
  float max_leaf_weight;
  float scale_pos_weight;
  unsigned char labels_count;
};

struct Verbose {
  Verbose(bool gpu = false, bool booster = false, bool data = false);
  bool gpu;
  bool booster;
  bool data;
};

struct InternalConfiguration {
  InternalConfiguration(bool double_precision = false,
                        unsigned short overlap = 2, unsigned int seed = 0,
                        bool use_hist_subtraction_trick = true,
                        bool upload_features = true, int hist_size = 255);
  bool double_precision;
  unsigned short overlap;
  unsigned int seed;
  bool use_hist_subtraction_trick;
  bool upload_features;
  int hist_size;
};

struct Configuration {
  Method method;
  Objective objective;
  TreeParam tree_param;
  Verbose verbose;
  InternalConfiguration internal;
  Configuration() {}
  static Configuration Parse(const char *configuration);
};

}  // namespace core
}  // namespace arboretum

#endif
