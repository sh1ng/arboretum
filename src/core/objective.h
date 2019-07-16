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

class ApproximatedObjectiveBase {
 public:
  ApproximatedObjectiveBase(io::DataMatrix *data) : data(data) {}
  virtual ~ApproximatedObjectiveBase() {}
  const io::DataMatrix *data;
  virtual void UpdateGrad() = 0;
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
  ApproximatedObjective(io::DataMatrix *data)
      : ApproximatedObjectiveBase(data) {
    blockSize = MAX_THREADS;
    gridSize = (data->rows + blockSize - 1) / blockSize;
  }
  thrust::device_vector<grad_type> grad;
  int gridSize;
  int blockSize;
};

class RegressionObjective : public ApproximatedObjective<float> {
 public:
  RegressionObjective(io::DataMatrix *data, float initial_y);
  virtual void UpdateGrad() override;
};

class LogisticRegressionObjective : public ApproximatedObjective<float2> {
 public:
  LogisticRegressionObjective(io::DataMatrix *data, float initial_y);
  virtual void UpdateGrad() override;

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
  SoftMaxObjective(io::DataMatrix *data, unsigned char labels_count,
                   float initial_y);
  virtual void UpdateGrad() override;
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
