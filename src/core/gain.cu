#include <cuda_runtime.h>
#include "gain.cuh"
#include "param.h"

namespace arboretum {
namespace core {

template <typename T>
inline __host__ __device__ int sign(T val) {
  return (T(0) < val) - (val < T(0));
}

template <typename T1, typename T2>
inline __host__ __device__ float gain(T1 grad, T2 hess, float w, float lambda,
                                      float alpha) {
  return -2 * (grad * (-w) + 0.5 * (hess + lambda) * w * w + alpha * abs(w));
}

template <typename T>
inline __host__ __device__ T grad_regularized(T grad, const float alpha) {
  const T grad_abs = abs(grad);
  return sign(grad) * max(0.0, grad_abs - alpha);
}

inline __host__ __device__ float ThresholdWeight(const float w,
                                                 const float max_leaf_weight) {
  if (max_leaf_weight != 0.0f) {
    if (w > max_leaf_weight) return max_leaf_weight;
    if (w < -max_leaf_weight) return -max_leaf_weight;
  }
  return w;
}

inline __host__ __device__ float Weight(const float s, const unsigned int c,
                                        const float min_child_weight,
                                        const unsigned min_leaf_size,
                                        const float lambda, const float alpha,
                                        const float max_leaf_weight) {
  float w = 0.0;
  if (c >= min_leaf_size && c >= min_child_weight)
    w = grad_regularized(s, alpha) / (c + lambda);
  return ThresholdWeight(w, max_leaf_weight);
}

inline __host__ __device__ float Weight(const double s, const unsigned int c,
                                        const float min_child_weight,
                                        const unsigned min_leaf_size,
                                        const float lambda, const float alpha,
                                        const float max_leaf_weight) {
  float w = 0.0;
  if (c >= min_leaf_size && c >= min_child_weight)
    w = grad_regularized(s, alpha) / (c + lambda);
  return ThresholdWeight(w, max_leaf_weight);
}

inline __host__ __device__ float Weight(const float2 s, const unsigned int c,
                                        const float min_child_weight,
                                        const unsigned min_leaf_size,
                                        const float lambda, const float alpha,
                                        const float max_leaf_weight) {
  float w = 0.0;
  if (c >= min_leaf_size && s.y >= min_child_weight)
    w = grad_regularized(s.x, alpha) / (s.y + lambda);
  return ThresholdWeight(w, max_leaf_weight);
}

inline __host__ __device__ float Weight(const double2 s, const unsigned int c,
                                        const float min_child_weight,
                                        const unsigned min_leaf_size,
                                        const float lambda, const float alpha,
                                        const float max_leaf_weight) {
  float w = 0.0;
  if (c >= min_leaf_size && s.y >= min_child_weight)
    w = grad_regularized(s.x, alpha) / (s.y + lambda);
  return ThresholdWeight(w, max_leaf_weight);
}

__host__ __device__ float Weight(const float s, const unsigned int c,
                                 const TreeParam &param) {
  return Weight(s, c, param.min_child_weight, param.min_leaf_size, param.lambda,
                param.alpha, param.max_leaf_weight);
}

__host__ __device__ float Weight(const double s, const unsigned int c,
                                 const TreeParam &param) {
  return Weight(s, c, param.min_child_weight, param.min_leaf_size, param.lambda,
                param.alpha, param.max_leaf_weight);
}

__host__ __device__ float Weight(const float2 s, const unsigned int c,
                                 const TreeParam &param) {
  return Weight(s, c, param.min_child_weight, param.min_leaf_size, param.lambda,
                param.alpha, param.max_leaf_weight);
}
__host__ __device__ float Weight(const double2 s, const unsigned int c,
                                 const TreeParam &param) {
  return Weight(s, c, param.min_child_weight, param.min_leaf_size, param.lambda,
                param.alpha, param.max_leaf_weight);
}

__host__ __device__ float Weight(const float s, const unsigned int c,
                                 const GainFunctionParameters &param) {
  return Weight(s, c, param.hess, param.min_leaf_size, param.lambda,
                param.alpha, param.max_leaf_weight);
}

__host__ __device__ float Weight(const double s, const unsigned int c,
                                 const GainFunctionParameters &param) {
  return Weight(s, c, param.hess, param.min_leaf_size, param.lambda,
                param.alpha, param.max_leaf_weight);
}

__host__ __device__ float Weight(const float2 s, const unsigned int c,
                                 const GainFunctionParameters &param) {
  return Weight(s, c, param.hess, param.min_leaf_size, param.lambda,
                param.alpha, param.max_leaf_weight);
}
__host__ __device__ float Weight(const double2 s, const unsigned int c,
                                 const GainFunctionParameters &param) {
  return Weight(s, c, param.hess, param.min_leaf_size, param.lambda,
                param.alpha, param.max_leaf_weight);
}

__device__ __host__ float gain_func(const double2 left_sum,
                                    const double2 total_sum,
                                    const size_t left_count,
                                    const size_t total_count,
                                    const GainFunctionParameters &params) {
  const double2 right_sum = total_sum - left_sum;
  const size_t right_count = total_count - left_count;
  if (left_count >= params.min_leaf_size &&
      right_count >= params.min_leaf_size &&
      std::abs(left_sum.y) >= params.hess &&
      std::abs(right_sum.y) >= params.hess) {
    const float weight_left = Weight(left_sum, left_count, params);
    const float weight_right = Weight(right_sum, right_count, params);
    const float weight_no_split = Weight(total_sum, total_count, params);

    const float l =
      gain(left_sum.x, left_sum.y, weight_left, params.lambda, params.alpha);
    const float r =
      gain(right_sum.x, right_sum.y, weight_right, params.lambda, params.alpha);
    const float p = gain(total_sum.x, total_sum.y, weight_no_split,
                         params.lambda, params.alpha);
    const float diff = l + r - p;
    return (diff > params.gamma_absolute && diff > params.gamma_relative * p) *
           diff;
  } else {
    return 0.0;
  }
}

__device__ __host__ float gain_func(const float2 left_sum,
                                    const float2 total_sum,
                                    const size_t left_count,
                                    const size_t total_count,
                                    const GainFunctionParameters &params) {
  const float2 right_sum = total_sum - left_sum;
  const size_t right_count = total_count - left_count;
  if (left_count >= params.min_leaf_size &&
      right_count >= params.min_leaf_size && abs(left_sum.y) >= params.hess &&
      abs(right_sum.y) >= params.hess) {
    const float weight_left = Weight(left_sum, left_count, params);
    const float weight_right = Weight(right_sum, right_count, params);
    const float weight_no_split = Weight(total_sum, total_count, params);

    const float l =
      gain(left_sum.x, left_sum.y, weight_left, params.lambda, params.alpha);
    const float r =
      gain(right_sum.x, right_sum.y, weight_right, params.lambda, params.alpha);
    const float p = gain(total_sum.x, total_sum.y, weight_no_split,
                         params.lambda, params.alpha);

    const float diff = l + r - p;
    return (diff > params.gamma_absolute && diff > params.gamma_relative * p) *
           diff;
  } else {
    return 0.0;
  }
}

__device__ __host__ float gain_func(const float left_sum, const float total_sum,
                                    const size_t left_count,
                                    const size_t total_count,
                                    const GainFunctionParameters &params) {
  const size_t right_count = total_count - left_count;
  const float right_sum = total_sum - left_sum;

  if (left_count >= params.min_leaf_size &&
      right_count >= params.min_leaf_size) {
    const float weight_left = Weight(left_sum, left_count, params);
    const float weight_right = Weight(right_sum, right_count, params);
    const float weight_no_split = Weight(total_sum, total_count, params);

    const float l =
      gain(left_sum, left_count, weight_left, params.lambda, params.alpha);
    const float r =
      gain(right_sum, right_count, weight_right, params.lambda, params.alpha);
    const float p = gain(total_sum, total_count, weight_no_split, params.lambda,
                         params.alpha);

    const float diff = l + r - p;

    return (diff > params.gamma_absolute && diff > params.gamma_relative * p) *
           diff;
  } else {
    return 0.0;
  }
}

__device__ __host__ float gain_func(const double left_sum,
                                    const double total_sum,
                                    const size_t left_count,
                                    const size_t total_count,
                                    const GainFunctionParameters &params) {
  const size_t right_count = total_count - left_count;
  const double right_sum = total_sum - left_sum;
  if (left_count >= params.min_leaf_size &&
      right_count >= params.min_leaf_size) {
    const float weight_left = Weight(left_sum, left_count, params);
    const float weight_right = Weight(right_sum, right_count, params);
    const float weight_no_split = Weight(total_sum, total_count, params);

    const float l =
      gain(left_sum, left_count, weight_left, params.lambda, params.alpha);
    const float r =
      gain(right_sum, right_count, weight_right, params.lambda, params.alpha);
    const float p = gain(total_sum, total_count, weight_no_split, params.lambda,
                         params.alpha);

    const float diff = l + r - p;

    return (diff > params.gamma_absolute && diff > params.gamma_relative * p) *
           diff;
  } else {
    return 0.0;
  }
}

}  // namespace core
}  // namespace arboretum