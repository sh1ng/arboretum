#ifndef SRC_CORE_GAIN_CUH
#define SRC_CORE_GAIN_CUH

#include <cuda_runtime.h>
#include "cuda_helpers.h"
#include "param.h"

namespace arboretum {
namespace core {

struct GainFunctionParameters {
  const unsigned int min_leaf_size;
  const float hess;
  const float gamma_absolute;
  const float gamma_relative;
  const float lambda;
  const float alpha;
  const float max_leaf_weight;
  __host__ __device__ GainFunctionParameters(
    const unsigned int min_leaf_size, const float hess,
    const float gamma_absolute, const float gamma_relative, const float lambda,
    const float alpha, const float max_leaf_weight = 0)
      : min_leaf_size(min_leaf_size),
        hess(hess),
        gamma_absolute(gamma_absolute),
        gamma_relative(gamma_relative),
        lambda(lambda),
        alpha(alpha),
        max_leaf_weight(max_leaf_weight) {}
};

__host__ __device__ float Weight(const float s, const unsigned int c,
                                 const TreeParam &param);

__host__ __device__ float Weight(const double s, const unsigned int c,
                                 const TreeParam &param);

__host__ __device__ float Weight(const float2 s, const unsigned int c,
                                 const TreeParam &param);

__host__ __device__ float Weight(const double2 s, const unsigned int c,
                                 const TreeParam &param);

__device__ __host__ float gain_func(const double2 left_sum,
                                    const double2 total_sum,
                                    const size_t left_count,
                                    const size_t total_count,
                                    const GainFunctionParameters &params);

__device__ __host__ float gain_func(const float2 left_sum,
                                    const float2 total_sum,
                                    const size_t left_count,
                                    const size_t total_count,
                                    const GainFunctionParameters &params);

__device__ __host__ float gain_func(const float left_sum, const float total_sum,
                                    const size_t left_count,
                                    const size_t total_count,
                                    const GainFunctionParameters &params);

__device__ __host__ float gain_func(const double left_sum,
                                    const double total_sum,
                                    const size_t left_count,
                                    const size_t total_count,
                                    const GainFunctionParameters &params);

}  // namespace core
}  // namespace arboretum

#endif