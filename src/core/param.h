#ifndef PARAM_H
#define PARAM_H

#include "json.hpp"

namespace arboretum {
  namespace core {
    using nlohmann::json;

    enum Objective {
      LinearRegression,
      LogisticRegression
    };

    template<typename T>
    struct ThreadSpecific{
      T data;
      int thread;
    };

    struct TreeParam{
    public:
      static TreeParam Parse(const json& cfg){
        int objective = cfg.value("/objective"_json_pointer, -1);
        float eta = cfg.value("/tree/eta"_json_pointer, 0.3);
        unsigned int max_depth = cfg.value("/tree/max_depth"_json_pointer, 8);
        float gamma = cfg.value("/tree/gamma"_json_pointer, 0.0);
        float min_child_weight = cfg.value("/tree/min_child_weight"_json_pointer, 1.0);
        unsigned int min_leaf_size = cfg.value("/tree/min_leaf_size"_json_pointer, 1);
        float lambda = cfg.value("/tree/lambda"_json_pointer, 1.0);
        float alpha = cfg.value("/tree/alpha"_json_pointer, 0.0);
        float colsample_bytree = cfg.value("/tree/colsample_bytree"_json_pointer, 1.0);
        float colsample_bylevel = cfg.value("/tree/colsample_bylevel"_json_pointer, 1.0);
        float initial = cfg.value("/tree/initial_y"_json_pointer, 0.5);

        return TreeParam((Objective)objective,
                         max_depth,
                         min_child_weight,
                         min_leaf_size,
                         colsample_bytree,
                         colsample_bylevel,
                         gamma,
                         lambda,
                         alpha,
                         initial,
                         eta);
      }

      TreeParam(Objective objective,
                int depth,
                float min_child_weight,
                unsigned int min_leaf_size,
                float colsample_bytre,
                float colsample_bylevel,
                float gamma,
                float lambda,
                float alpha,
                float initial_y,
                float eta) :
        objective(objective),
        depth(depth),
        min_child_weight(min_child_weight),
        min_leaf_size(min_leaf_size),
        colsample_bytree(colsample_bytre),
        colsample_bylevel(colsample_bylevel),
        gamma(gamma),
        lambda(lambda),
        alpha(alpha),
        initial_y(initial_y),
        eta(eta) {
      }
      const Objective objective;
      const int depth;
      const float min_child_weight;
      const unsigned int min_leaf_size;
      const float colsample_bytree;
      const float colsample_bylevel;
      const float gamma;
      const float lambda;
      const float alpha;
      const float initial_y;
      const float eta;
    };

    struct Verbose {
    public:
      static Verbose Parse(const json& cfg){
        return Verbose(cfg.value("/verbose/gpu"_json_pointer, false));
      }
      Verbose(bool gpu) :
        gpu(gpu)
      {}
      const bool gpu;
    };

    struct InternalConfiguration {
    public:
      static InternalConfiguration Parse(const json& cfg){
        return InternalConfiguration(cfg.value("/internals/double_precision"_json_pointer, false),
                                     cfg.value("/internals/compute_overlap"_json_pointer, 2));
      }
      InternalConfiguration(bool double_precision, unsigned short overlap) :
        double_precision(double_precision),
        overlap(overlap) {}
      const bool double_precision;
      const unsigned short overlap;
    };
  }
}

#endif // PARAM_H
