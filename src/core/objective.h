#ifndef SRC_CORE_OBJECTIVE_H
#define SRC_CORE_OBJECTIVE_H

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <vector>
#include "../io/io.h"
#include "cuda_runtime.h"

#define MAX_THREADS 1024

namespace arboretum {
namespace core {

class ApproximatedObjectiveBase {
 public:
  ApproximatedObjectiveBase() {}
  virtual ~ApproximatedObjectiveBase() {}
  virtual float IntoInternal(float v) { return v; }
  virtual inline void FromInternal(thrust::host_vector<float> &in,
                                   std::vector<float> &out) {
#pragma omp parallel for
    for (size_t i = 0; i < out.size(); ++i) {
      out[i] = in[i];
    }
  }
};

template <class grad_type>
class ApproximatedObjective : public ApproximatedObjectiveBase {
 public:
  ApproximatedObjective() : ApproximatedObjectiveBase() {}
  virtual void UpdateGrad(thrust::device_vector<grad_type> &grad,
                          const thrust::device_vector<float> &y_hat_d,
                          const thrust::device_vector<float> &y_d) = 0;
};

class RegressionObjective : public ApproximatedObjective<float> {
 public:
  RegressionObjective(float initial_y);
  virtual void UpdateGrad(thrust::device_vector<float> &grad,
                          const thrust::device_vector<float> &y_hat_d,
                          const thrust::device_vector<float> &y_d) override;
  //   virtual void UpdateGrad() override;
};

class LogisticRegressionObjective : public ApproximatedObjective<float2> {
 public:
  LogisticRegressionObjective(float initial_y);
  virtual void UpdateGrad(thrust::device_vector<float2> &grad,
                          const thrust::device_vector<float> &y_hat_d,
                          const thrust::device_vector<float> &y_d) override;
  //   virtual void UpdateGrad() override;

  virtual inline float IntoInternal(float v) override {
    return std::log(v / (1 - v));
  }
  virtual inline void FromInternal(thrust::host_vector<float> &in,
                                   std::vector<float> &out) override {
#pragma omp parallel for
    for (size_t i = 0; i < out.size(); ++i) {
      out[i] = Sigmoid(in[i]);
    }
  }

 private:
  inline __host__ __device__ float Sigmoid(float x) {
    return 1.0 / (1.0 + std::exp(-x));
  }
};

class SoftMaxObjective : public ApproximatedObjective<float2> {
 public:
  SoftMaxObjective(unsigned char labels_count, float initial_y);
  virtual void UpdateGrad(thrust::device_vector<float2> &grad,
                          const thrust::device_vector<float> &y_hat_d,
                          const thrust::device_vector<float> &y_d) override;
  //   virtual void UpdateGrad() override;
  virtual inline float IntoInternal(float v) override { return v; }
  virtual inline void FromInternal(thrust::host_vector<float> &in,
                                   std::vector<float> &out) override {
    const size_t n = in.size() / labels_count;
#pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
      std::vector<double> labels_prob(labels_count);

      for (unsigned char j = 0; j < labels_count; ++j) {
        labels_prob[j] = in[i + n * j];
      }

      SoftMax(labels_prob);

      for (unsigned char j = 0; j < labels_count; ++j) {
        out[i * labels_count + j] = labels_prob[j];
      }
    }
  }

 private:
  const unsigned char labels_count;
  inline float Sigmoid(float x) { return 1.0 / (1.0 + std::exp(-x)); }

  //  #pragma omp declare simd
  inline void SoftMax(std::vector<double> &values) const {
    double sum = 0.0;
    for (unsigned short i = 0; i < labels_count; ++i) {
      values[i] = std::exp(values[i]);
      sum += values[i];
    }
    for (unsigned short i = 0; i < labels_count; ++i) {
      values[i] /= sum;
    }
  }
};

}  // namespace core
}  // namespace arboretum

#endif  // SRC_CORE_OBJECTIVE_H
