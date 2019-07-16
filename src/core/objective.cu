#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <vector>
#include "../io/io.h"
#include "cuda_helpers.h"
#include "objective.h"

namespace arboretum {
namespace core {
using namespace thrust;
using namespace thrust::cuda;
using thrust::device_vector;
using thrust::host_vector;

template <typename GRAD_TYPE, typename F>
__global__ void update_grad(GRAD_TYPE *grad, const float *y_hat, const float *y,
                            F func, const size_t n) {
  for (size_t i = blockDim.x * blockIdx.x + threadIdx.x; i < n;
       i += gridDim.x * blockDim.x) {
    GRAD_TYPE g = func(y_hat[i], y[i]);
    grad[i] = g;
  }
}

RegressionObjective::RegressionObjective(io::DataMatrix *data, float initial_y)
    : ApproximatedObjective<float>(data) {
  grad.resize(data->rows);
  data->y_internal.resize(data->rows, IntoInternal(initial_y));
  data->y_internal_d = data->y_internal;
  data->y_hat_d = data->y_hat;
}
void RegressionObjective::UpdateGrad() {
  auto func = [] __device__(float y_hat, float y) { return y_hat - y; };

  update_grad<<<gridSize, blockSize>>>(
    thrust::raw_pointer_cast(grad.data()),
    thrust::raw_pointer_cast(data->y_hat_d.data()),
    thrust::raw_pointer_cast(data->y_internal_d.data()), func, data->rows);
}

LogisticRegressionObjective::LogisticRegressionObjective(io::DataMatrix *data,
                                                         float initial_y)
    : ApproximatedObjective<float2>(data) {
  grad.resize(data->rows);
  data->y_internal.resize(data->rows, IntoInternal(initial_y));
  data->y_internal_d = data->y_internal;
  data->y_hat_d = data->y_hat;
}
void LogisticRegressionObjective::UpdateGrad() {
  auto func = [=] __device__(float y_hat, float y) {
    const float sigmoid = Sigmoid(y);
    return make_float2(y_hat - sigmoid, sigmoid * (1.0f - sigmoid));
  };

  update_grad<<<gridSize, blockSize>>>(
    thrust::raw_pointer_cast(grad.data()),
    thrust::raw_pointer_cast(data->y_hat_d.data()),
    thrust::raw_pointer_cast(data->y_internal_d.data()), func, data->rows);
}

SoftMaxObjective::SoftMaxObjective(io::DataMatrix *data,
                                   unsigned char labels_count, float initial_y)
    : ApproximatedObjective<float2>(data), labels_count(labels_count) {
  grad.resize(data->rows * labels_count);
  data->y_internal.resize(data->rows * labels_count, IntoInternal(initial_y));
}
void SoftMaxObjective::UpdateGrad() {
  //     const float label_lookup[] = {0.0, 1.0};
  // #pragma omp parallel for simd
  //     for (size_t i = 0; i < data->rows; ++i) {
  //       std::vector<double> labels_prob(labels_count);

  //       for (unsigned char j = 0; j < labels_count; ++j) {
  //         labels_prob[j] = data->y_internal[i + j * data->rows];
  //       }

  //       SoftMax(labels_prob);

  //       const unsigned char label = data->labels[i];

  //       for (unsigned char j = 0; j < labels_count; ++j) {
  //         const double pred = labels_prob[j];
  //         grad[j * data->rows + i].x = label_lookup[j == label] - pred;
  //         grad[j * data->rows + i].y = 2.0 * pred * (1.0 - pred);
  //       }
  //     }

  //     if (data->weights.size() > 0) {
  // #pragma omp parallel for simd
  //       for (size_t i = 0; i < data->rows; ++i) {
  //         for (unsigned char j = 0; j < labels_count; ++j) {
  //           grad[j * data->rows + i].x *= data->weights[i];
  //         }
  //       }
  //     }
}

}  // namespace core
}  // namespace arboretum