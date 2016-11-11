#ifndef OBJECTIVE_H
#define OBJECTIVE_H

#include <vector>

class GradBuilder {
public:
  static inline float Sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }
  static inline double Sigmoid(double x) {
    return 1.0f / (1.0f + std::exp(-x));
  }
  static inline float2 Regression(float y, float y_hat) {
    return make_float2(2.0 * (y - y_hat), 2.0);
  }
  static inline float2 LogReg(float y, float y_hat) {
    return make_float2(y - y_hat, std::max(y * (1.0f - y_hat), 1e-16f));
  }
};

#endif // OBJECTIVE_H
