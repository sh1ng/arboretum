#ifndef OBJECTIVE_H
#define OBJECTIVE_H

#include <vector>

class GradBuilder{
public:
  static inline float Sigmoid(float x){
    return 1.0f / (1.0f + std::exp(-x));
  }
  static inline float Regression(float y, float y_hat){
    return 2.0 * (y - y_hat);
  }
  static inline float LogReg(float y, float y_hat){
    return Sigmoid(y) - y_hat;
  }
};

#endif // OBJECTIVE_H
