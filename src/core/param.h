#ifndef PARAM_H
#define PARAM_H

namespace arboretum {
  namespace core {
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
      TreeParam(Objective objective, int depth, int min_child_weight, float colsample_bytre, float eta) : objective(objective),
        depth(depth), min_child_weight(min_child_weight), colsample_bytree(colsample_bytre), initial_y(0.5), eta(eta) {

      }
      Objective objective;
      int depth;
      int min_child_weight;
      float colsample_bytree;
      float initial_y;
      float eta;
    };
  }
}

#endif // PARAM_H
