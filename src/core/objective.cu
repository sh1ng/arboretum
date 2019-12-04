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

RegressionObjective::RegressionObjective(float initial_y)
    : ApproximatedObjective<float>() {}

void RegressionObjective::UpdateGrad(
  thrust::device_vector<float> &grad,
  const thrust::device_vector<float> &y_hat_d,
  const thrust::device_vector<float> &y_d) {
  auto func = [] __device__(float y_hat, float y) { return y - y_hat; };
  int gridSize = (grad.size() + MAX_THREADS - 1) / MAX_THREADS;
  update_grad<<<gridSize, MAX_THREADS>>>(
    thrust::raw_pointer_cast(grad.data()),
    thrust::raw_pointer_cast(y_hat_d.data()),
    thrust::raw_pointer_cast(y_d.data()), func, grad.size());
}

LogisticRegressionObjective::LogisticRegressionObjective(float initial_y)
    : ApproximatedObjective<float2>() {}

void LogisticRegressionObjective::UpdateGrad(
  thrust::device_vector<float2> &grad,
  const thrust::device_vector<float> &y_hat_d,
  const thrust::device_vector<float> &y_d) {
  int gridSize = (grad.size() + MAX_THREADS - 1) / MAX_THREADS;
  auto func = [=] __device__(float y_hat, float y) {
    const float sigmoid = Sigmoid(y_hat);
    return make_float2(y - sigmoid, sigmoid * (1.0f - sigmoid));
  };
  update_grad<<<gridSize, MAX_THREADS>>>(
    thrust::raw_pointer_cast(grad.data()),
    thrust::raw_pointer_cast(y_hat_d.data()),
    thrust::raw_pointer_cast(y_d.data()), func, grad.size());
}

SoftMaxObjective::SoftMaxObjective(unsigned char labels_count, float initial_y)
    : ApproximatedObjective<float2>(), labels_count(labels_count) {}
// void SoftMaxObjective::UpdateGrad() {
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
// }

void SoftMaxObjective::UpdateGrad(thrust::device_vector<float2> &grad,
                                  const thrust::device_vector<float> &y_hat_d,
                                  const thrust::device_vector<float> &y_d) {
  //   auto func = [] __device__(float y_hat, float y) { return y_hat - y; };

  //   update_grad<<<gridSize, blockSize>>>(
  //     thrust::raw_pointer_cast(grad.data()),
  //     thrust::raw_pointer_cast(data->y_hat_d.data()),
  //     thrust::raw_pointer_cast(data->y_internal_d.data()), func, data->rows);
}

}  // namespace core
}  // namespace arboretum