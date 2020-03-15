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

TEST(HistTreeGrower, CreatePartitioningIndexes) {
  const unsigned level = 1;
  const unsigned depth = 2;
  const InternalConfiguration config(false, 1, 0, false, true);
  const size_t size = 32;
  const unsigned hist_size = 8;
  BestSplit<float2> best(1 << depth, hist_size);
  Histogram<float2> histogram(1 << depth, hist_size, 1);
  auto grower = HistTreeGrower<unsigned int, unsigned short, float2, float2>(
    size, depth, hist_size, &best, &histogram, &config);
  thrust::device_vector<unsigned int> row2Node(size, 0);
  thrust::device_vector<unsigned int> partitioning_indexes(size, 0);

  thrust::device_vector<unsigned int> parent_node_count(3, 0);

  for (int i = 0; i < size; ++i) {
    unsigned row = i % 2;
    row2Node[i] = row;
  }

  parent_node_count[1] += 16;
  parent_node_count[2] += 32;

  grower.CreatePartitioningIndexes(partitioning_indexes, row2Node,
                                   parent_node_count, level, depth);

  ASSERT_EQ(partitioning_indexes[0], 0);
  ASSERT_EQ(partitioning_indexes[1], 31);
  ASSERT_EQ(partitioning_indexes[2], 1);
  ASSERT_EQ(partitioning_indexes[3], 30);
  ASSERT_EQ(partitioning_indexes[4], 2);
  ASSERT_EQ(partitioning_indexes[5], 29);
  ASSERT_EQ(partitioning_indexes[6], 3);
  ASSERT_EQ(partitioning_indexes[7], 28);
  ASSERT_EQ(partitioning_indexes[8], 4);
  ASSERT_EQ(partitioning_indexes[9], 27);
  ASSERT_EQ(partitioning_indexes[10], 5);
  ASSERT_EQ(partitioning_indexes[11], 26);
  ASSERT_EQ(partitioning_indexes[12], 6);
  ASSERT_EQ(partitioning_indexes[13], 25);
  ASSERT_EQ(partitioning_indexes[14], 7);
  ASSERT_EQ(partitioning_indexes[15], 24);
  ASSERT_EQ(partitioning_indexes[16], 8);
  ASSERT_EQ(partitioning_indexes[17], 23);
  ASSERT_EQ(partitioning_indexes[18], 9);
  ASSERT_EQ(partitioning_indexes[19], 22);
  ASSERT_EQ(partitioning_indexes[20], 10);
  ASSERT_EQ(partitioning_indexes[21], 21);
  ASSERT_EQ(partitioning_indexes[22], 11);
  ASSERT_EQ(partitioning_indexes[23], 20);
  ASSERT_EQ(partitioning_indexes[24], 12);
  ASSERT_EQ(partitioning_indexes[25], 19);
  ASSERT_EQ(partitioning_indexes[26], 13);
  ASSERT_EQ(partitioning_indexes[27], 18);
  ASSERT_EQ(partitioning_indexes[28], 14);
  ASSERT_EQ(partitioning_indexes[29], 17);
  ASSERT_EQ(partitioning_indexes[30], 15);
  ASSERT_EQ(partitioning_indexes[31], 16);
}

TEST(HistTreeGrower, RootSearchContinuousFeature) {
  const InternalConfiguration config(false, 1, 0, true, true);
  const size_t size = 32;
  const unsigned hist_size = 4;
  const unsigned depth = 1;
  BestSplit<float2> best(1 << depth, hist_size);
  Histogram<float2> histogram(1 << depth, hist_size, 1);
  auto grower = HistTreeGrower<unsigned int, unsigned short, float2, float2>(
    size, depth, hist_size, &best, &histogram, &config);
  thrust::device_vector<unsigned int> row2Node(size, 0);
  thrust::device_vector<unsigned int> partitioning_indexes(size, 0);
  thrust::device_vector<float2> grad(32);
  thrust::host_vector<unsigned short> fvalue_h(32);
  thrust::device_vector<unsigned short> fvalue_d(32);
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

  grower.ProcessDenseFeature(partitioning_indexes, row2Node, grad, fvalue_d,
                             thrust::raw_pointer_cast(fvalue_h.data()),
                             parent_node_sum, parent_node_count, 3, 0, 1, p,
                             false, 0);
  TEST_OK(cudaStreamSynchronize(grower.stream));

  thrust::host_vector<my_atomics> result_h = grower.result_d;
  // copy pasted result
  ASSERT_EQ(result_h[0].ints[1], 1);
  ASSERT_FLOAT_EQ(result_h[0].floats[0], 2048.0);
}

TEST(HistTreeGrower, Level1SearchContinuousFeatureNoTrickDynamic) {
  const unsigned level = 1;
  const unsigned depth = 2;
  const InternalConfiguration config(false, 1, 0, false, true);
  const size_t size = 32;
  const unsigned hist_size = 8;
  BestSplit<float2> best(1 << depth, hist_size);
  Histogram<float2> histogram(1 << depth, hist_size, 1);
  auto grower = HistTreeGrower<unsigned int, unsigned short, float2, float2>(
    size, depth, hist_size, &best, &histogram, &config);
  thrust::device_vector<unsigned int> row2Node(size, 0);
  thrust::device_vector<unsigned int> partitioning_indexes(size, 0);
  thrust::device_vector<float2> grad(32);
  thrust::host_vector<unsigned short> fvalue_h(32);
  thrust::device_vector<unsigned short> fvalue_d(32);
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

  grower.CreatePartitioningIndexes(partitioning_indexes, row2Node,
                                   parent_node_count, level, depth);

  grower.ProcessDenseFeature(partitioning_indexes, row2Node, grad, fvalue_d,
                             thrust::raw_pointer_cast(fvalue_h.data()),
                             parent_node_sum, parent_node_count, 3, level,
                             depth, p, false, 0);
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

  ASSERT_EQ(fvalue_d[0], 0);
  ASSERT_EQ(fvalue_d[1], 0);
  ASSERT_EQ(fvalue_d[2], 1);
  ASSERT_EQ(fvalue_d[3], 1);
  ASSERT_EQ(fvalue_d[4], 2);
  ASSERT_EQ(fvalue_d[5], 2);
  ASSERT_EQ(fvalue_d[6], 3);
  ASSERT_EQ(fvalue_d[7], 3);
  ASSERT_EQ(fvalue_d[8], 4);
  ASSERT_EQ(fvalue_d[9], 4);
  ASSERT_EQ(fvalue_d[10], 5);
  ASSERT_EQ(fvalue_d[11], 5);
  ASSERT_EQ(fvalue_d[12], 6);
  ASSERT_EQ(fvalue_d[13], 6);
  ASSERT_EQ(fvalue_d[14], 7);
  ASSERT_EQ(fvalue_d[15], 7);
  ASSERT_EQ(fvalue_d[16], 7);
  ASSERT_EQ(fvalue_d[17], 7);
  ASSERT_EQ(fvalue_d[18], 6);
  ASSERT_EQ(fvalue_d[19], 6);
  ASSERT_EQ(fvalue_d[20], 5);
  ASSERT_EQ(fvalue_d[21], 5);
  ASSERT_EQ(fvalue_d[22], 4);
  ASSERT_EQ(fvalue_d[23], 4);
  ASSERT_EQ(fvalue_d[24], 3);
  ASSERT_EQ(fvalue_d[25], 3);
  ASSERT_EQ(fvalue_d[26], 2);
  ASSERT_EQ(fvalue_d[27], 2);
  ASSERT_EQ(fvalue_d[28], 1);
  ASSERT_EQ(fvalue_d[29], 1);
  ASSERT_EQ(fvalue_d[30], 0);
  ASSERT_EQ(fvalue_d[31], 0);
}

TEST(HistTreeGrower, Level1SearchContinuousFeatureNoTrickDynamic_Upload) {
  const unsigned level = 1;
  const unsigned depth = 2;
  const InternalConfiguration config(false, 1, 0, false, true);
  const size_t size = 32;
  const unsigned hist_size = 8;
  BestSplit<float2> best(1 << depth, hist_size);
  Histogram<float2> histogram(1 << depth, hist_size, 1);
  auto grower = HistTreeGrower<unsigned int, unsigned short, float2, float2>(
    size, depth, hist_size, &best, &histogram, &config);
  thrust::device_vector<unsigned int> row2Node(size, 0);
  thrust::device_vector<unsigned int> partitioning_indexes(size, 0);
  thrust::device_vector<float2> grad(32);
  thrust::host_vector<unsigned short> fvalue_h(32);
  thrust::device_vector<unsigned short> fvalue_d(0);
  float2 sum = make_float2(0.0, 0.0);

  thrust::device_vector<unsigned> hist_count((1 << depth) * hist_size, 0);
  thrust::device_vector<float2> hist_sum((1 << depth) * hist_size);

  thrust::device_vector<float2> parent_node_sum(3, make_float2(0, 0));

  thrust::device_vector<unsigned int> parent_node_count(3, 0);

  for (int i = 0; i < size; ++i) {
    unsigned row = i % 2;
    row2Node[i] = row;
    grad[i] = make_float2(i * i, 1.0);
    fvalue_h[i] = i / 4;
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

  grower.CreatePartitioningIndexes(partitioning_indexes, row2Node,
                                   parent_node_count, level, depth);

  grower.ProcessDenseFeature(partitioning_indexes, row2Node, grad, fvalue_d,
                             thrust::raw_pointer_cast(fvalue_h.data()),
                             parent_node_sum, parent_node_count, 3, level,
                             depth, p, false, 0);
  TEST_OK(cudaStreamSynchronize(grower.stream));
  TEST_OK(cudaStreamSynchronize(grower.copy_d2h_stream));

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

  ASSERT_EQ(fvalue_h[0], 0);
  ASSERT_EQ(fvalue_h[1], 0);
  ASSERT_EQ(fvalue_h[2], 1);
  ASSERT_EQ(fvalue_h[3], 1);
  ASSERT_EQ(fvalue_h[4], 2);
  ASSERT_EQ(fvalue_h[5], 2);
  ASSERT_EQ(fvalue_h[6], 3);
  ASSERT_EQ(fvalue_h[7], 3);
  ASSERT_EQ(fvalue_h[8], 4);
  ASSERT_EQ(fvalue_h[9], 4);
  ASSERT_EQ(fvalue_h[10], 5);
  ASSERT_EQ(fvalue_h[11], 5);
  ASSERT_EQ(fvalue_h[12], 6);
  ASSERT_EQ(fvalue_h[13], 6);
  ASSERT_EQ(fvalue_h[14], 7);
  ASSERT_EQ(fvalue_h[15], 7);
  ASSERT_EQ(fvalue_h[16], 7);
  ASSERT_EQ(fvalue_h[17], 7);
  ASSERT_EQ(fvalue_h[18], 6);
  ASSERT_EQ(fvalue_h[19], 6);
  ASSERT_EQ(fvalue_h[20], 5);
  ASSERT_EQ(fvalue_h[21], 5);
  ASSERT_EQ(fvalue_h[22], 4);
  ASSERT_EQ(fvalue_h[23], 4);
  ASSERT_EQ(fvalue_h[24], 3);
  ASSERT_EQ(fvalue_h[25], 3);
  ASSERT_EQ(fvalue_h[26], 2);
  ASSERT_EQ(fvalue_h[27], 2);
  ASSERT_EQ(fvalue_h[28], 1);
  ASSERT_EQ(fvalue_h[29], 1);
  ASSERT_EQ(fvalue_h[30], 0);
  ASSERT_EQ(fvalue_h[31], 0);
}

TEST(HistTreeGrower,
     Level1SearchContinuousFeatureNoTrickDynamic_PartitionOnly) {
  const unsigned level = 1;
  const unsigned depth = 2;
  const InternalConfiguration config(false, 1, 0, false, true);
  const size_t size = 32;
  const unsigned hist_size = 8;
  BestSplit<float2> best(1 << depth, hist_size);
  Histogram<float2> histogram(1 << depth, hist_size, 1);
  auto grower = HistTreeGrower<unsigned int, unsigned short, float2, float2>(
    size, depth, hist_size, &best, &histogram, &config);
  thrust::device_vector<unsigned int> row2Node(size, 0);
  thrust::device_vector<unsigned int> partitioning_indexes(size, 0);
  thrust::device_vector<float2> grad(32);
  thrust::host_vector<unsigned short> fvalue_h(32);
  thrust::device_vector<unsigned short> fvalue_d(32);
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

  grower.CreatePartitioningIndexes(partitioning_indexes, row2Node,
                                   parent_node_count, level, depth);

  grower.ProcessDenseFeature(partitioning_indexes, row2Node, grad, fvalue_d,
                             thrust::raw_pointer_cast(fvalue_h.data()),
                             parent_node_sum, parent_node_count, 3, level,
                             depth, p, true, 0);
  TEST_OK(cudaStreamSynchronize(grower.stream));
  TEST_OK(cudaStreamSynchronize(grower.copy_d2h_stream));

  ASSERT_EQ(fvalue_d[0], 0);
  ASSERT_EQ(fvalue_d[1], 0);
  ASSERT_EQ(fvalue_d[2], 1);
  ASSERT_EQ(fvalue_d[3], 1);
  ASSERT_EQ(fvalue_d[4], 2);
  ASSERT_EQ(fvalue_d[5], 2);
  ASSERT_EQ(fvalue_d[6], 3);
  ASSERT_EQ(fvalue_d[7], 3);
  ASSERT_EQ(fvalue_d[8], 4);
  ASSERT_EQ(fvalue_d[9], 4);
  ASSERT_EQ(fvalue_d[10], 5);
  ASSERT_EQ(fvalue_d[11], 5);
  ASSERT_EQ(fvalue_d[12], 6);
  ASSERT_EQ(fvalue_d[13], 6);
  ASSERT_EQ(fvalue_d[14], 7);
  ASSERT_EQ(fvalue_d[15], 7);
  ASSERT_EQ(fvalue_d[16], 7);
  ASSERT_EQ(fvalue_d[17], 7);
  ASSERT_EQ(fvalue_d[18], 6);
  ASSERT_EQ(fvalue_d[19], 6);
  ASSERT_EQ(fvalue_d[20], 5);
  ASSERT_EQ(fvalue_d[21], 5);
  ASSERT_EQ(fvalue_d[22], 4);
  ASSERT_EQ(fvalue_d[23], 4);
  ASSERT_EQ(fvalue_d[24], 3);
  ASSERT_EQ(fvalue_d[25], 3);
  ASSERT_EQ(fvalue_d[26], 2);
  ASSERT_EQ(fvalue_d[27], 2);
  ASSERT_EQ(fvalue_d[28], 1);
  ASSERT_EQ(fvalue_d[29], 1);
  ASSERT_EQ(fvalue_d[30], 0);
  ASSERT_EQ(fvalue_d[31], 0);
}

TEST(HistTreeGrower,
     Level1SearchContinuousFeatureNoTrickDynamic_PartitionOnly_Upload) {
  const unsigned level = 1;
  const unsigned depth = 2;
  const InternalConfiguration config(false, 1, 0, false, true);
  const size_t size = 32;
  const unsigned hist_size = 8;
  BestSplit<float2> best(1 << depth, hist_size);
  Histogram<float2> histogram(1 << depth, hist_size, 1);
  auto grower = HistTreeGrower<unsigned int, unsigned short, float2, float2>(
    size, depth, hist_size, &best, &histogram, &config);
  thrust::device_vector<unsigned int> row2Node(size, 0);
  thrust::device_vector<unsigned int> partitioning_indexes(size, 0);
  thrust::device_vector<float2> grad(32);
  thrust::host_vector<unsigned short> fvalue_h(32);
  thrust::device_vector<unsigned short> fvalue_d(0);
  float2 sum = make_float2(0.0, 0.0);

  thrust::device_vector<unsigned> hist_count((1 << depth) * hist_size, 0);
  thrust::device_vector<float2> hist_sum((1 << depth) * hist_size);

  thrust::device_vector<float2> parent_node_sum(3, make_float2(0, 0));

  thrust::device_vector<unsigned int> parent_node_count(3, 0);

  for (int i = 0; i < size; ++i) {
    unsigned row = i % 2;
    row2Node[i] = row;
    grad[i] = make_float2(i * i, 1.0);
    fvalue_h[i] = i / 4;
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

  grower.CreatePartitioningIndexes(partitioning_indexes, row2Node,
                                   parent_node_count, level, depth);

  grower.ProcessDenseFeature(partitioning_indexes, row2Node, grad, fvalue_d,
                             thrust::raw_pointer_cast(fvalue_h.data()),
                             parent_node_sum, parent_node_count, 3, level,
                             depth, p, true, 0);
  TEST_OK(cudaStreamSynchronize(grower.stream));
  TEST_OK(cudaStreamSynchronize(grower.copy_d2h_stream));

  ASSERT_EQ(fvalue_h[0], 0);
  ASSERT_EQ(fvalue_h[1], 0);
  ASSERT_EQ(fvalue_h[2], 1);
  ASSERT_EQ(fvalue_h[3], 1);
  ASSERT_EQ(fvalue_h[4], 2);
  ASSERT_EQ(fvalue_h[5], 2);
  ASSERT_EQ(fvalue_h[6], 3);
  ASSERT_EQ(fvalue_h[7], 3);
  ASSERT_EQ(fvalue_h[8], 4);
  ASSERT_EQ(fvalue_h[9], 4);
  ASSERT_EQ(fvalue_h[10], 5);
  ASSERT_EQ(fvalue_h[11], 5);
  ASSERT_EQ(fvalue_h[12], 6);
  ASSERT_EQ(fvalue_h[13], 6);
  ASSERT_EQ(fvalue_h[14], 7);
  ASSERT_EQ(fvalue_h[15], 7);
  ASSERT_EQ(fvalue_h[16], 7);
  ASSERT_EQ(fvalue_h[17], 7);
  ASSERT_EQ(fvalue_h[18], 6);
  ASSERT_EQ(fvalue_h[19], 6);
  ASSERT_EQ(fvalue_h[20], 5);
  ASSERT_EQ(fvalue_h[21], 5);
  ASSERT_EQ(fvalue_h[22], 4);
  ASSERT_EQ(fvalue_h[23], 4);
  ASSERT_EQ(fvalue_h[24], 3);
  ASSERT_EQ(fvalue_h[25], 3);
  ASSERT_EQ(fvalue_h[26], 2);
  ASSERT_EQ(fvalue_h[27], 2);
  ASSERT_EQ(fvalue_h[28], 1);
  ASSERT_EQ(fvalue_h[29], 1);
  ASSERT_EQ(fvalue_h[30], 0);
  ASSERT_EQ(fvalue_h[31], 0);
}

TEST(HistTreeGrower, Level1SearchContinuousFeatureNoTrickStatic) {
  const InternalConfiguration config(false, 1, 0, false, false);
  const size_t size = 32;
  const unsigned depth = 2;
  const unsigned level = 1;
  const unsigned hist_size = 8;
  BestSplit<float2> best(1 << depth, hist_size);
  Histogram<float2> histogram(1 << depth, hist_size, 1);
  auto grower = HistTreeGrower<unsigned int, unsigned short, float2, float2>(
    size, depth, hist_size, &best, &histogram, &config);
  thrust::device_vector<unsigned int> row2Node(size, 0);
  thrust::device_vector<unsigned int> partitioning_indexes(size, 0);
  thrust::device_vector<float2> grad(32);
  thrust::host_vector<unsigned short> fvalue_h(32);
  thrust::device_vector<unsigned short> fvalue_d(32);
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

  grower.CreatePartitioningIndexes(partitioning_indexes, row2Node,
                                   parent_node_count, level, depth);

  grower.ProcessDenseFeature(partitioning_indexes, row2Node, grad, fvalue_d,
                             thrust::raw_pointer_cast(fvalue_h.data()),
                             parent_node_sum, parent_node_count, 3, level,
                             depth, p, false, 0);
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
  const unsigned level = 1;
  const unsigned depth = 2;
  const size_t size = 32;
  const unsigned hist_size = 8;
  BestSplit<float2> best(1 << depth, hist_size);
  Histogram<float2> histogram(1 << depth, hist_size, 1);
  auto grower = HistTreeGrower<unsigned int, unsigned short, float2, float2>(
    size, depth, hist_size, &best, &histogram, &config);
  thrust::device_vector<unsigned int> row2Node(size, 0);
  thrust::device_vector<unsigned> partitioning_indexes(size, 0);
  thrust::device_vector<float2> grad(32);
  thrust::host_vector<unsigned short> fvalue_h(32);
  thrust::device_vector<unsigned short> fvalue_d(32);
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

  grower.CreatePartitioningIndexes(partitioning_indexes, row2Node,
                                   parent_node_count, level, depth);

  grower.ProcessDenseFeature(partitioning_indexes, row2Node, grad, fvalue_d,
                             thrust::raw_pointer_cast(fvalue_h.data()),
                             parent_node_sum, parent_node_count, 3, level,
                             depth, p, false, 0);
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
  const unsigned level = 1;
  const unsigned hist_size = 8;
  BestSplit<float2> best(1 << depth, hist_size);
  Histogram<float2> histogram(1 << depth, hist_size, 1);
  auto grower = HistTreeGrower<unsigned int, unsigned short, float2, float2>(
    size, depth, hist_size, &best, &histogram, &config);
  thrust::device_vector<unsigned int> row2Node(size, 0);
  thrust::device_vector<unsigned int> partitioning_indexes(size, 0);
  thrust::device_vector<float2> grad(32);
  thrust::host_vector<unsigned short> fvalue_h(32);
  thrust::device_vector<unsigned short> fvalue_d(32);
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

  grower.CreatePartitioningIndexes(partitioning_indexes, row2Node,
                                   parent_node_count, level, depth);

  grower.ProcessDenseFeature(partitioning_indexes, row2Node, grad, fvalue_d,
                             thrust::raw_pointer_cast(fvalue_h.data()),
                             parent_node_sum, parent_node_count, 3, level,
                             depth, p, false, 0);
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