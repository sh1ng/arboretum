#include "core/builder.h"
#include "gtest/gtest.h"
#include "io/io.h"
#include "test_utils.h"
#include "thrust/device_vector.h"
#include "thrust/host_vector.h"

namespace arboretum_test {
using arboretum::core::ContinuousTreeGrower;
using arboretum::core::GainFunctionParameters;
using arboretum::core::my_atomics;

TEST(ContinuousTreeGrower, RootSearchCategoryFeature) {
  const size_t size = 32;
  auto grower = ContinuousTreeGrower<unsigned int, float2, float2>(size, 1);
  thrust::device_vector<unsigned int> row2Node(size, 0);
  thrust::device_vector<float2> grad(32);
  thrust::host_vector<unsigned int> fvalue_h(32);
  thrust::device_vector<unsigned int> fvalue_d(32);
  float2 sum = make_float2(0.0, 0.0);
  for (int i = 0; i < size; ++i) {
    grad[i] = make_float2(float(i), 1.0);
    fvalue_h[i] = fvalue_d[i] = i / 2;
    sum += grad[i];
  }

  thrust::device_vector<float2> parent_node_sum(2);
  parent_node_sum[0] = make_float2(0, 0);
  parent_node_sum[1] = sum;

  thrust::device_vector<unsigned int> parent_node_count(2, 0);
  parent_node_count[0] = 0;
  parent_node_count[1] = size;

  auto p = GainFunctionParameters(0, 0, 0, 0, 0, 0);

  grower.ProcessCategoryFeature<unsigned int>(
      row2Node, thrust::raw_pointer_cast(grad.data()), fvalue_d, fvalue_h,
      parent_node_sum, parent_node_count, 3, 0, p);
  TEST_OK(cudaStreamSynchronize(grower.stream));

  // copy pasted result
  ASSERT_EQ(grower.result_h[0].ints[1], 0);
  ASSERT_FLOAT_EQ(grower.result_h[0].floats[0], 480.0);
}

TEST(ContinuousTreeGrower, RootSearchContinuousFeature) {
  const size_t size = 32;
  auto grower = ContinuousTreeGrower<unsigned int, float2, float2>(size, 1);
  thrust::device_vector<unsigned int> row2Node(size, 0);
  thrust::device_vector<float2> grad(32);
  thrust::host_vector<unsigned int> fvalue_h(32);
  thrust::device_vector<unsigned int> fvalue_d(32);
  float2 sum = make_float2(0.0, 0.0);
  for (int i = 0; i < size; ++i) {
    grad[i] = make_float2(float(i), 1.0);
    fvalue_h[i] = fvalue_d[i] = i / 4;
    sum += grad[i];
  }

  thrust::device_vector<float2> parent_node_sum(2);
  parent_node_sum[0] = make_float2(0, 0);
  parent_node_sum[1] = sum;

  thrust::device_vector<unsigned int> parent_node_count(2, 0);
  parent_node_count[0] = 0;
  parent_node_count[1] = size;

  auto p = GainFunctionParameters(0, 0, 0, 0, 0, 0);

  grower.ProcessDenseFeature<unsigned int>(
      row2Node, thrust::raw_pointer_cast(grad.data()), fvalue_d, fvalue_h,
      parent_node_sum, parent_node_count, 3, 0, p);
  TEST_OK(cudaStreamSynchronize(grower.stream));

  // copy pasted result
  ASSERT_EQ(grower.result_h[0].ints[1], 16);
  ASSERT_FLOAT_EQ(grower.result_h[0].floats[0], 2048.0);
}
}  // namespace arboretum_test