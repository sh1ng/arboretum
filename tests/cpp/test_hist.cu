#include "core/best_splits.h"
#include "core/builder.h"
#include "core/hist_tree_grower.h"
#include "core/histogram.h"
#include "core/param.h"
#include "gtest/gtest.h"
#include "io/io.h"
#include "test_utils.h"
#include "thrust/device_vector.h"
#include "thrust/host_vector.h"

namespace arboretum_test {
using arboretum::core::BestSplit;
using arboretum::core::GainFunctionParameters;
using arboretum::core::Histogram;
using arboretum::core::HistTreeGrower;
using arboretum::core::InternalConfiguration;
using arboretum::core::my_atomics;

TEST(HistTreeGrower, RootSearchContinuousFeature) {
  const InternalConfiguration config(false, 1, 0, true, true);
  const size_t size = 32;
  const unsigned hist_size = 4;
  const unsigned depth = 1;
  BestSplit<float2> best(1 << depth, hist_size);
  Histogram<float2> histogram(1 << depth, hist_size, 1);
  auto grower = HistTreeGrower<unsigned int, float2, float2>(
    size, depth, hist_size, &best, &histogram, &config);
  thrust::device_vector<unsigned int> row2Node(size, 0);
  thrust::device_vector<float2> grad(32);
  thrust::host_vector<unsigned int> fvalue_h(32);
  thrust::device_vector<unsigned int> fvalue_d(32);
  float2 sum = make_float2(0.0, 0.0);
  for (int i = 0; i < size; ++i) {
    grad[i] = make_float2(float(i), 1.0);
    fvalue_h[i] = fvalue_d[i] = i / 8;
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
    row2Node, grad, thrust::raw_pointer_cast(fvalue_d.data()),
    thrust::raw_pointer_cast(fvalue_h.data()), parent_node_sum,
    parent_node_count, 3, 0, 1, p, false, 0);
  TEST_OK(cudaStreamSynchronize(grower.stream));

  thrust::host_vector<my_atomics> result_h = grower.result_d;
  // copy pasted result
  ASSERT_EQ(result_h[0].ints[1], 1);
  ASSERT_FLOAT_EQ(result_h[0].floats[0], 2048.0);
}

TEST(HistTreeGrower, Level1SearchContinuousFeatureNoTrickDynamic) {
  const InternalConfiguration config(false, 1, 0, false, true);
  const size_t size = 32;
  const unsigned depth = 2;
  const unsigned hist_size = 8;
  BestSplit<float2> best(1 << depth, hist_size);
  Histogram<float2> histogram(1 << depth, hist_size, 1);
  auto grower = HistTreeGrower<unsigned int, float2, float2>(
    size, depth, hist_size, &best, &histogram, &config);
  thrust::device_vector<unsigned int> row2Node(size, 0);
  thrust::device_vector<float2> grad(32);
  thrust::host_vector<unsigned int> fvalue_h(32);
  thrust::device_vector<unsigned int> fvalue_d(32);
  float2 sum = make_float2(0.0, 0.0);

  thrust::device_vector<unsigned> hist_count((1 << depth) * hist_size, 0);
  thrust::device_vector<float2> hist_sum((1 << depth) * hist_size);

  thrust::device_vector<float2> parent_node_sum(3, make_float2(0, 0));

  thrust::device_vector<unsigned int> parent_node_count(3, 0);

  for (int i = 0; i < size; ++i) {
    unsigned row = i % 2;
    row2Node[i] = row;
    grad[i] = make_float2(i * i, 1.0);
    fvalue_h[i] = fvalue_d[i] = i / 4;
    parent_node_sum[(i / 16) + 1] += grad[i];

    hist_count[fvalue_h[i]] += 1;
  }
  // sort feature values by row according to cub's partition logic
  hist_sum[0] = make_float2(0.0 + 1.0 + 900.0 + 961.0, 4.0);
  hist_sum[1] = make_float2(4.0 + 9.0 + 784.0 + 841.0, 4.0);
  hist_sum[2] = make_float2(16.0 + 25.0 + 676.0 + 729.0, 4.0);
  hist_sum[3] = make_float2(36.0 + 49.0 + 576.0 + 625.0, 4.0);
  hist_sum[4] = make_float2(64.0 + 81.0 + 484.0 + 529.0, 4.0);
  hist_sum[5] = make_float2(100.0 + 121.0 + 400.0 + 441.0, 4.0);
  hist_sum[6] = make_float2(144.0 + 169.0 + 324.0 + 361.0, 4.0);
  hist_sum[7] = make_float2(196.0 + 225.0 + 256.0 + 289.0, 4.0);

  parent_node_count[1] += 16;
  parent_node_count[2] += 32;

  parent_node_sum[1] += parent_node_sum[0];
  parent_node_sum[2] += parent_node_sum[1];

  auto p = GainFunctionParameters(0, 0, 0, 0, 0, 0);

  grower.ProcessDenseFeature<unsigned int>(
    row2Node, grad, thrust::raw_pointer_cast(fvalue_d.data()),
    thrust::raw_pointer_cast(fvalue_h.data()), parent_node_sum,
    parent_node_count, 3, 1, 2, p, false, 0);
  TEST_OK(cudaStreamSynchronize(grower.stream));

  thrust::host_vector<my_atomics> result_h = grower.result_d;

  for (unsigned i = 0; i < hist_size * 2; ++i) {
    ASSERT_EQ(2, grower.hist_bin_count[i]);
  }

  float2 tmp = grower.sum[0];
  ASSERT_FLOAT_EQ(1.0, tmp.x);
  tmp = grower.sum[1];
  ASSERT_FLOAT_EQ(13.0, tmp.x);
  tmp = grower.sum[2];
  ASSERT_FLOAT_EQ(41.0, tmp.x);
  tmp = grower.sum[3];
  ASSERT_FLOAT_EQ(85.0, tmp.x);
  tmp = grower.sum[4];
  ASSERT_FLOAT_EQ(145.0, tmp.x);
  tmp = grower.sum[5];
  ASSERT_FLOAT_EQ(221.0, tmp.x);
  tmp = grower.sum[6];
  ASSERT_FLOAT_EQ(313.0, tmp.x);
  tmp = grower.sum[7];
  ASSERT_FLOAT_EQ(421.0, tmp.x);
  tmp = grower.sum[8];
  ASSERT_FLOAT_EQ(1861.0, tmp.x);
  tmp = grower.sum[9];
  ASSERT_FLOAT_EQ(1625.0, tmp.x);
  tmp = grower.sum[10];
  ASSERT_FLOAT_EQ(1405.0, tmp.x);
  tmp = grower.sum[11];
  ASSERT_FLOAT_EQ(1201.0, tmp.x);
  tmp = grower.sum[12];
  ASSERT_FLOAT_EQ(1013.0, tmp.x);
  tmp = grower.sum[13];
  ASSERT_FLOAT_EQ(841.0, tmp.x);
  tmp = grower.sum[14];
  ASSERT_FLOAT_EQ(685.0, tmp.x);
  tmp = grower.sum[15];
  ASSERT_FLOAT_EQ(545.0, tmp.x);

  ASSERT_EQ(result_h[0].ints[1], 4);
  ASSERT_FLOAT_EQ(result_h[0].floats[0], 64026.672);

  ASSERT_EQ(result_h[1].ints[1], 11);
  ASSERT_FLOAT_EQ(result_h[1].floats[0], 565504);
}

TEST(HistTreeGrower, Level1SearchContinuousFeatureNoTrickStatic) {
  const InternalConfiguration config(false, 1, 0, false, false);
  const size_t size = 32;
  const unsigned depth = 2;
  const unsigned hist_size = 8;
  BestSplit<float2> best(1 << depth, hist_size);
  Histogram<float2> histogram(1 << depth, hist_size, 1);
  auto grower = HistTreeGrower<unsigned int, float2, float2>(
    size, depth, hist_size, &best, &histogram, &config);
  thrust::device_vector<unsigned int> row2Node(size, 0);
  thrust::device_vector<float2> grad(32);
  thrust::host_vector<unsigned int> fvalue_h(32);
  thrust::device_vector<unsigned int> fvalue_d(32);
  float2 sum = make_float2(0.0, 0.0);

  thrust::device_vector<unsigned> hist_count((1 << depth) * hist_size, 0);
  thrust::device_vector<float2> hist_sum((1 << depth) * hist_size);

  thrust::device_vector<float2> parent_node_sum(3, make_float2(0, 0));

  thrust::device_vector<unsigned int> parent_node_count(3, 0);

  for (int i = 0; i < size; ++i) {
    unsigned row = i % 2;
    row2Node[i] = row;
    grad[i] = make_float2(i * i, 1.0);
    fvalue_h[i] = fvalue_d[i] = i / 4;
    parent_node_sum[(i / 16) + 1] += grad[i];

    hist_count[fvalue_h[i]] += 1;
  }
  // sort feature values by row according to cub's partition logic
  hist_sum[0] = make_float2(0.0 + 1.0 + 900.0 + 961.0, 4.0);
  hist_sum[1] = make_float2(4.0 + 9.0 + 784.0 + 841.0, 4.0);
  hist_sum[2] = make_float2(16.0 + 25.0 + 676.0 + 729.0, 4.0);
  hist_sum[3] = make_float2(36.0 + 49.0 + 576.0 + 625.0, 4.0);
  hist_sum[4] = make_float2(64.0 + 81.0 + 484.0 + 529.0, 4.0);
  hist_sum[5] = make_float2(100.0 + 121.0 + 400.0 + 441.0, 4.0);
  hist_sum[6] = make_float2(144.0 + 169.0 + 324.0 + 361.0, 4.0);
  hist_sum[7] = make_float2(196.0 + 225.0 + 256.0 + 289.0, 4.0);

  parent_node_count[1] += 16;
  parent_node_count[2] += 32;

  parent_node_sum[1] += parent_node_sum[0];
  parent_node_sum[2] += parent_node_sum[1];

  best.parent_node_count_h = parent_node_count;

  auto p = GainFunctionParameters(0, 0, 0, 0, 0, 0);

  grower.ProcessDenseFeature<unsigned int>(
    row2Node, grad, thrust::raw_pointer_cast(fvalue_d.data()),
    thrust::raw_pointer_cast(fvalue_h.data()), parent_node_sum,
    parent_node_count, 3, 1, 2, p, false, 0);
  TEST_OK(cudaStreamSynchronize(grower.stream));

  thrust::host_vector<my_atomics> result_h = grower.result_d;

  for (unsigned i = 0; i < hist_size * 2; ++i) {
    ASSERT_EQ(2, grower.hist_bin_count[i]);
  }

  float2 tmp = grower.sum[0];
  ASSERT_FLOAT_EQ(1.0, tmp.x);
  tmp = grower.sum[1];
  ASSERT_FLOAT_EQ(13.0, tmp.x);
  tmp = grower.sum[2];
  ASSERT_FLOAT_EQ(41.0, tmp.x);
  tmp = grower.sum[3];
  ASSERT_FLOAT_EQ(85.0, tmp.x);
  tmp = grower.sum[4];
  ASSERT_FLOAT_EQ(145.0, tmp.x);
  tmp = grower.sum[5];
  ASSERT_FLOAT_EQ(221.0, tmp.x);
  tmp = grower.sum[6];
  ASSERT_FLOAT_EQ(313.0, tmp.x);
  tmp = grower.sum[7];
  ASSERT_FLOAT_EQ(421.0, tmp.x);
  tmp = grower.sum[8];
  ASSERT_FLOAT_EQ(1861.0, tmp.x);
  tmp = grower.sum[9];
  ASSERT_FLOAT_EQ(1625.0, tmp.x);
  tmp = grower.sum[10];
  ASSERT_FLOAT_EQ(1405.0, tmp.x);
  tmp = grower.sum[11];
  ASSERT_FLOAT_EQ(1201.0, tmp.x);
  tmp = grower.sum[12];
  ASSERT_FLOAT_EQ(1013.0, tmp.x);
  tmp = grower.sum[13];
  ASSERT_FLOAT_EQ(841.0, tmp.x);
  tmp = grower.sum[14];
  ASSERT_FLOAT_EQ(685.0, tmp.x);
  tmp = grower.sum[15];
  ASSERT_FLOAT_EQ(545.0, tmp.x);

  ASSERT_EQ(result_h[0].ints[1], 4);
  ASSERT_FLOAT_EQ(result_h[0].floats[0], 64026.672);

  ASSERT_EQ(result_h[1].ints[1], 11);
  ASSERT_FLOAT_EQ(result_h[1].floats[0], 565504);
}

TEST(HistTreeGrower, Level1SearchContinuousFeatureWithTrickDynamic) {
  const InternalConfiguration config(false, 1, 0, true, true);

  const size_t size = 32;
  const unsigned depth = 2;
  const unsigned hist_size = 8;
  BestSplit<float2> best(1 << depth, hist_size);
  Histogram<float2> histogram(1 << depth, hist_size, 1);
  auto grower = HistTreeGrower<unsigned int, float2, float2>(
    size, depth, hist_size, &best, &histogram, &config);
  thrust::device_vector<unsigned int> row2Node(size, 0);
  thrust::device_vector<float2> grad(32);
  thrust::host_vector<unsigned int> fvalue_h(32);
  thrust::device_vector<unsigned int> fvalue_d(32);
  float2 sum = make_float2(0.0, 0.0);

  thrust::device_vector<unsigned> hist_count((1 << depth) * hist_size, 0);
  thrust::device_vector<float2> hist_sum((1 << depth) * hist_size);

  thrust::device_vector<float2> parent_node_sum(3, make_float2(0, 0));

  thrust::device_vector<unsigned int> parent_node_count(3, 0);

  for (int i = 0; i < size; ++i) {
    unsigned row = i % 2;
    row2Node[i] = row;
    grad[i] = make_float2(i * i, 1.0);
    fvalue_h[i] = fvalue_d[i] = i / 4;
    parent_node_sum[(i / 16) + 1] += grad[i];

    hist_count[fvalue_h[i]] += 1;
  }
  // sort feature values by row according to cub's partition logic
  hist_sum[0] = make_float2(0.0 + 1.0 + 900.0 + 961.0, 4.0);
  hist_sum[1] = make_float2(4.0 + 9.0 + 784.0 + 841.0, 4.0);
  hist_sum[2] = make_float2(16.0 + 25.0 + 676.0 + 729.0, 4.0);
  hist_sum[3] = make_float2(36.0 + 49.0 + 576.0 + 625.0, 4.0);
  hist_sum[4] = make_float2(64.0 + 81.0 + 484.0 + 529.0, 4.0);
  hist_sum[5] = make_float2(100.0 + 121.0 + 400.0 + 441.0, 4.0);
  hist_sum[6] = make_float2(144.0 + 169.0 + 324.0 + 361.0, 4.0);
  hist_sum[7] = make_float2(196.0 + 225.0 + 256.0 + 289.0, 4.0);

  histogram.Update(hist_sum, hist_count, 0, 1 << (depth - 1), grower.stream);

  parent_node_count[1] += 16;
  parent_node_count[2] += 32;

  parent_node_sum[1] += parent_node_sum[0];
  parent_node_sum[2] += parent_node_sum[1];

  auto p = GainFunctionParameters(0, 0, 0, 0, 0, 0);

  grower.ProcessDenseFeature<unsigned int>(
    row2Node, grad, thrust::raw_pointer_cast(fvalue_d.data()),
    thrust::raw_pointer_cast(fvalue_h.data()), parent_node_sum,
    parent_node_count, 3, 1, 2, p, false, 0);
  TEST_OK(cudaStreamSynchronize(grower.stream));

  thrust::host_vector<my_atomics> result_h = grower.result_d;

  for (unsigned i = 0; i < hist_size * 2; ++i) {
    unsigned count = grower.hist_bin_count[i];
    ASSERT_EQ(2, grower.hist_bin_count[i]);
  }

  float2 tmp = grower.sum[0];
  ASSERT_FLOAT_EQ(1.0, tmp.x);
  tmp = grower.sum[1];
  ASSERT_FLOAT_EQ(13.0, tmp.x);
  tmp = grower.sum[2];
  ASSERT_FLOAT_EQ(41.0, tmp.x);
  tmp = grower.sum[3];
  ASSERT_FLOAT_EQ(85.0, tmp.x);
  tmp = grower.sum[4];
  ASSERT_FLOAT_EQ(145.0, tmp.x);
  tmp = grower.sum[5];
  ASSERT_FLOAT_EQ(221.0, tmp.x);
  tmp = grower.sum[6];
  ASSERT_FLOAT_EQ(313.0, tmp.x);
  tmp = grower.sum[7];
  ASSERT_FLOAT_EQ(421.0, tmp.x);
  tmp = grower.sum[8];
  ASSERT_FLOAT_EQ(1861.0, tmp.x);
  tmp = grower.sum[9];
  ASSERT_FLOAT_EQ(1625.0, tmp.x);
  tmp = grower.sum[10];
  ASSERT_FLOAT_EQ(1405.0, tmp.x);
  tmp = grower.sum[11];
  ASSERT_FLOAT_EQ(1201.0, tmp.x);
  tmp = grower.sum[12];
  ASSERT_FLOAT_EQ(1013.0, tmp.x);
  tmp = grower.sum[13];
  ASSERT_FLOAT_EQ(841.0, tmp.x);
  tmp = grower.sum[14];
  ASSERT_FLOAT_EQ(685.0, tmp.x);
  tmp = grower.sum[15];
  ASSERT_FLOAT_EQ(545.0, tmp.x);

  ASSERT_EQ(result_h[0].ints[1], 4);
  ASSERT_FLOAT_EQ(result_h[0].floats[0], 64026.672);

  ASSERT_EQ(result_h[1].ints[1], 11);
  ASSERT_FLOAT_EQ(result_h[1].floats[0], 565504);
}

TEST(HistTreeGrower, Level1SearchContinuousFeatureWithTrickStatic) {
  const InternalConfiguration config(false, 1, 0, true, false);

  const size_t size = 32;
  const unsigned depth = 2;
  const unsigned hist_size = 8;
  BestSplit<float2> best(1 << depth, hist_size);
  Histogram<float2> histogram(1 << depth, hist_size, 1);
  auto grower = HistTreeGrower<unsigned int, float2, float2>(
    size, depth, hist_size, &best, &histogram, &config);
  thrust::device_vector<unsigned int> row2Node(size, 0);
  thrust::device_vector<float2> grad(32);
  thrust::host_vector<unsigned int> fvalue_h(32);
  thrust::device_vector<unsigned int> fvalue_d(32);
  float2 sum = make_float2(0.0, 0.0);

  thrust::device_vector<unsigned> hist_count((1 << depth) * hist_size, 0);
  thrust::device_vector<float2> hist_sum((1 << depth) * hist_size);

  thrust::device_vector<float2> parent_node_sum(3, make_float2(0, 0));

  thrust::device_vector<unsigned int> parent_node_count(3, 0);

  for (int i = 0; i < size; ++i) {
    unsigned row = i % 2;
    row2Node[i] = row;
    grad[i] = make_float2(i * i, 1.0);
    fvalue_h[i] = fvalue_d[i] = i / 4;
    parent_node_sum[(i / 16) + 1] += grad[i];

    hist_count[fvalue_h[i]] += 1;
  }
  // sort feature values by row according to cub's partition logic
  hist_sum[0] = make_float2(0.0 + 1.0 + 900.0 + 961.0, 4.0);
  hist_sum[1] = make_float2(4.0 + 9.0 + 784.0 + 841.0, 4.0);
  hist_sum[2] = make_float2(16.0 + 25.0 + 676.0 + 729.0, 4.0);
  hist_sum[3] = make_float2(36.0 + 49.0 + 576.0 + 625.0, 4.0);
  hist_sum[4] = make_float2(64.0 + 81.0 + 484.0 + 529.0, 4.0);
  hist_sum[5] = make_float2(100.0 + 121.0 + 400.0 + 441.0, 4.0);
  hist_sum[6] = make_float2(144.0 + 169.0 + 324.0 + 361.0, 4.0);
  hist_sum[7] = make_float2(196.0 + 225.0 + 256.0 + 289.0, 4.0);

  histogram.Update(hist_sum, hist_count, 0, 1 << (depth - 1), grower.stream);

  parent_node_count[1] += 16;
  parent_node_count[2] += 32;

  parent_node_sum[1] += parent_node_sum[0];
  parent_node_sum[2] += parent_node_sum[1];

  best.parent_node_count_h = parent_node_count;

  auto p = GainFunctionParameters(0, 0, 0, 0, 0, 0);

  grower.ProcessDenseFeature<unsigned int>(
    row2Node, grad, thrust::raw_pointer_cast(fvalue_d.data()),
    thrust::raw_pointer_cast(fvalue_h.data()), parent_node_sum,
    parent_node_count, 3, 1, 2, p, false, 0);
  TEST_OK(cudaStreamSynchronize(grower.stream));

  thrust::host_vector<my_atomics> result_h = grower.result_d;

  for (unsigned i = 0; i < hist_size * 2; ++i) {
    unsigned count = grower.hist_bin_count[i];
    ASSERT_EQ(2, grower.hist_bin_count[i]);
  }

  float2 tmp = grower.sum[0];
  ASSERT_FLOAT_EQ(1.0, tmp.x);
  tmp = grower.sum[1];
  ASSERT_FLOAT_EQ(13.0, tmp.x);
  tmp = grower.sum[2];
  ASSERT_FLOAT_EQ(41.0, tmp.x);
  tmp = grower.sum[3];
  ASSERT_FLOAT_EQ(85.0, tmp.x);
  tmp = grower.sum[4];
  ASSERT_FLOAT_EQ(145.0, tmp.x);
  tmp = grower.sum[5];
  ASSERT_FLOAT_EQ(221.0, tmp.x);
  tmp = grower.sum[6];
  ASSERT_FLOAT_EQ(313.0, tmp.x);
  tmp = grower.sum[7];
  ASSERT_FLOAT_EQ(421.0, tmp.x);
  tmp = grower.sum[8];
  ASSERT_FLOAT_EQ(1861.0, tmp.x);
  tmp = grower.sum[9];
  ASSERT_FLOAT_EQ(1625.0, tmp.x);
  tmp = grower.sum[10];
  ASSERT_FLOAT_EQ(1405.0, tmp.x);
  tmp = grower.sum[11];
  ASSERT_FLOAT_EQ(1201.0, tmp.x);
  tmp = grower.sum[12];
  ASSERT_FLOAT_EQ(1013.0, tmp.x);
  tmp = grower.sum[13];
  ASSERT_FLOAT_EQ(841.0, tmp.x);
  tmp = grower.sum[14];
  ASSERT_FLOAT_EQ(685.0, tmp.x);
  tmp = grower.sum[15];
  ASSERT_FLOAT_EQ(545.0, tmp.x);

  ASSERT_EQ(result_h[0].ints[1], 4);
  ASSERT_FLOAT_EQ(result_h[0].floats[0], 64026.672);

  ASSERT_EQ(result_h[1].ints[1], 11);
  ASSERT_FLOAT_EQ(result_h[1].floats[0], 565504);
}

}  // namespace arboretum_test