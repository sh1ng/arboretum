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
  auto grower = ContinuousTreeGrower<unsigned int, float2, float2>(size, 1, 0);
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

  grower.ProcessCategoryFeature<unsigned int>(row2Node, grad, fvalue_d,
                                              fvalue_h, parent_node_sum,
                                              parent_node_count, 3, 0, p);
  TEST_OK(cudaStreamSynchronize(grower.stream));

  thrust::host_vector<my_atomics> result_h = grower.result_d;
  // copy pasted result
  ASSERT_EQ(result_h[0].ints[1], 0);
  ASSERT_FLOAT_EQ(result_h[0].floats[0], 480.0);
}

TEST(ContinuousTreeGrower, RootSearchContinuousFeature) {
  const size_t size = 32;
  auto grower = ContinuousTreeGrower<unsigned int, float2, float2>(size, 1, 0);
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
    row2Node, grad, thrust::raw_pointer_cast(fvalue_d.data()),
    thrust::raw_pointer_cast(fvalue_h.data()), parent_node_sum,
    parent_node_count, 3, 0, 1, p, false);
  TEST_OK(cudaStreamSynchronize(grower.stream));

  thrust::host_vector<my_atomics> result_h = grower.result_d;
  // copy pasted result
  ASSERT_EQ(result_h[0].ints[1], 16);
  ASSERT_FLOAT_EQ(result_h[0].floats[0], 2048.0);
}

TEST(ContinuousTreeGrower, FloatUnstableExample) {
  const size_t size = 10;
  auto grower =
    ContinuousTreeGrower<unsigned int, float, float>(size, 4, 0, NULL);
  thrust::host_vector<unsigned int> node(size, 0);
  thrust::host_vector<float> grad(size);
  thrust::host_vector<unsigned int> feature(size);

  thrust::host_vector<float> parent_node_sum(3, 0.0);

  thrust::host_vector<unsigned int> parent_node_count(3, 0);

  grad[0] = 2.350000;
  feature[0] = 4;
  node[0] = 0;
  grad[1] = 0.750004;
  feature[1] = 3;
  node[1] = 4;
  grad[2] = 2.100000;
  feature[2] = 6;
  node[2] = 2;
  grad[3] = -2.100000;
  feature[3] = 2;
  node[3] = 4;
  grad[4] = -1.200003;
  feature[4] = 1;
  node[4] = 4;
  grad[5] = 1.199997;
  feature[5] = 0;
  node[5] = 4;
  grad[6] = -1.199997;
  feature[6] = 6;
  node[6] = 6;
  grad[7] = 1.200003;
  feature[7] = 6;
  node[7] = 6;
  grad[8] = -0.749996;
  feature[8] = 5;
  node[8] = 4;
  grad[9] = -2.350000;
  feature[9] = 6;
  node[9] = 4;

  // sum was computed with a slight error
  parent_node_sum[0] = 0.000000;
  parent_node_count[0] = 0;
  // prcise value 3.1
  parent_node_sum[1] = 3.100004;
  parent_node_count[1] = 2;
  // prcise value 0.0
  parent_node_sum[2] = 0.000008;
  parent_node_count[2] = 10;

  auto p = GainFunctionParameters(2, 2, 0, 0, 0, 0);

  thrust::device_vector<unsigned> feature_d = feature;

  grower.ProcessDenseFeature<unsigned int>(
    node, grad, thrust::raw_pointer_cast(feature_d.data()),
    thrust::raw_pointer_cast(feature.data()), parent_node_sum,
    parent_node_count, 3, 1, 4, p, false);
  TEST_OK(cudaStreamSynchronize(grower.stream));

  thrust::host_vector<my_atomics> result_h = grower.result_d;
  // copy pasted result
  ASSERT_EQ(result_h[0].ints[1], 0);
  ASSERT_FLOAT_EQ(result_h[0].floats[0], 0);

  ASSERT_EQ(result_h[1].ints[1], 7);
  ASSERT_FLOAT_EQ(result_h[1].floats[0], 5.676724);
}

TEST(ContinuousTreeGrower, DoubleUnstableExample) {
  const size_t size = 10;
  auto grower = ContinuousTreeGrower<unsigned int, float, double>(size, 4, 0);
  thrust::host_vector<unsigned int> node(size, 0);
  thrust::host_vector<float> grad(size);
  thrust::host_vector<unsigned int> feature(size);

  thrust::host_vector<double> parent_node_sum(3, 0.0);

  thrust::host_vector<unsigned int> parent_node_count(3, 0);

  grad[0] = 2.350000;
  feature[0] = 4;
  node[0] = 0;
  grad[1] = 0.750000;
  feature[1] = 3;
  node[1] = 4;
  grad[2] = 2.100000;
  feature[2] = 6;
  node[2] = 2;
  grad[3] = -2.100000;
  feature[3] = 2;
  node[3] = 4;
  grad[4] = -1.199999;
  feature[4] = 1;
  node[4] = 4;
  grad[5] = 1.200001;
  feature[5] = 0;
  node[5] = 4;
  grad[6] = -1.200001;
  feature[6] = 6;
  node[6] = 6;
  grad[7] = 1.199999;
  feature[7] = 6;
  node[7] = 6;
  grad[8] = -0.75;
  feature[8] = 5;
  node[8] = 4;
  grad[9] = -2.350000;
  feature[9] = 6;
  node[9] = 4;

  // precise sums
  parent_node_sum[0] = 0.0;
  parent_node_count[0] = 0;
  parent_node_sum[1] = 3.1;
  parent_node_count[1] = 2;
  parent_node_sum[2] = 0.0;
  parent_node_count[2] = 10;

  auto p = GainFunctionParameters(2, 2, 0, 0, 0, 0);

  thrust::device_vector<unsigned> feature_d = feature;

  grower.ProcessDenseFeature<unsigned int>(
    node, grad, thrust::raw_pointer_cast(feature_d.data()),
    thrust::raw_pointer_cast(feature.data()), parent_node_sum,
    parent_node_count, 3, 1, 4, p, false);
  TEST_OK(cudaStreamSynchronize(grower.stream));

  thrust::host_vector<my_atomics> result_h = grower.result_d;
  // copy pasted result
  ASSERT_EQ(result_h[0].ints[1], 0);
  ASSERT_FLOAT_EQ(result_h[0].floats[0], 0);

  ASSERT_EQ(result_h[1].ints[1], 7);
  ASSERT_FLOAT_EQ(result_h[1].floats[0], 5.6767569);
}

TEST(ContinuousTreeGrower, Level1SearchContinuousFeature) {
  const size_t size = 32;
  auto grower = ContinuousTreeGrower<unsigned int, float2, float2>(size, 2, 0);
  thrust::device_vector<unsigned int> row2Node(size, 0);
  thrust::device_vector<float2> grad(32);
  thrust::host_vector<unsigned int> fvalue_h(32);
  thrust::device_vector<unsigned int> fvalue_d(32);
  float2 sum = make_float2(0.0, 0.0);

  thrust::device_vector<float2> parent_node_sum(3, make_float2(0, 0));

  thrust::device_vector<unsigned int> parent_node_count(3, 0);

  for (int i = 0; i < size; ++i) {
    row2Node[i] = i % 2;
    grad[i] = make_float2(i * i, 1.0);
    fvalue_h[i] = fvalue_d[i] = i / 4;
    parent_node_sum[(i / 16) + 1] += grad[i];
    // parent_node_count[(i / 16) + 1] += 1;
  }
  parent_node_count[1] += 16;
  parent_node_count[2] += 32;

  parent_node_sum[1] += parent_node_sum[0];
  parent_node_sum[2] += parent_node_sum[1];

  auto p = GainFunctionParameters(0, 0, 0, 0, 0, 0);

  grower.ProcessDenseFeature<unsigned int>(
    row2Node, grad, thrust::raw_pointer_cast(fvalue_d.data()),
    thrust::raw_pointer_cast(fvalue_h.data()), parent_node_sum,
    parent_node_count, 3, 1, 2, p, false);
  TEST_OK(cudaStreamSynchronize(grower.stream));

  thrust::host_vector<my_atomics> result_h = grower.result_d;
  // copy pasted result
  ASSERT_EQ(result_h[0].ints[1], 10);
  ASSERT_FLOAT_EQ(result_h[0].floats[0], 64026.672);

  ASSERT_EQ(result_h[1].ints[1], 24);
  ASSERT_FLOAT_EQ(result_h[1].floats[0], 565504);

  ASSERT_EQ(grower.node_fvalue[0], 0);
  ASSERT_EQ(grower.node_fvalue[1], 0);
  ASSERT_EQ(grower.node_fvalue[2], 1);
  ASSERT_EQ(grower.node_fvalue[3], 1);
  ASSERT_EQ(grower.node_fvalue[4], 2);
  ASSERT_EQ(grower.node_fvalue[5], 2);
  ASSERT_EQ(grower.node_fvalue[6], 3);
  ASSERT_EQ(grower.node_fvalue[7], 3);
  ASSERT_EQ(grower.node_fvalue[8], 4);
  ASSERT_EQ(grower.node_fvalue[9], 4);
  ASSERT_EQ(grower.node_fvalue[10], 5);
  ASSERT_EQ(grower.node_fvalue[11], 5);
  ASSERT_EQ(grower.node_fvalue[12], 6);
  ASSERT_EQ(grower.node_fvalue[13], 6);
  ASSERT_EQ(grower.node_fvalue[14], 7);
  ASSERT_EQ(grower.node_fvalue[15], 7);
  ASSERT_EQ(grower.node_fvalue[16], 7);
  ASSERT_EQ(grower.node_fvalue[17], 7);
  ASSERT_EQ(grower.node_fvalue[18], 6);
  ASSERT_EQ(grower.node_fvalue[19], 6);
  ASSERT_EQ(grower.node_fvalue[20], 5);
  ASSERT_EQ(grower.node_fvalue[21], 5);
  ASSERT_EQ(grower.node_fvalue[22], 4);
  ASSERT_EQ(grower.node_fvalue[23], 4);
  ASSERT_EQ(grower.node_fvalue[24], 3);
  ASSERT_EQ(grower.node_fvalue[25], 3);
  ASSERT_EQ(grower.node_fvalue[26], 2);
  ASSERT_EQ(grower.node_fvalue[27], 2);
  ASSERT_EQ(grower.node_fvalue[28], 1);
  ASSERT_EQ(grower.node_fvalue[29], 1);
  ASSERT_EQ(grower.node_fvalue[30], 0);
  ASSERT_EQ(grower.node_fvalue[31], 0);
}

TEST(ContinuousTreeGrower, Level1SearchContinuousFeatureDouble) {
  const size_t size = 32;
  auto grower =
    ContinuousTreeGrower<unsigned int, float2, mydouble2>(size, 2, 0);
  thrust::device_vector<unsigned int> row2Node(size, 0);
  thrust::device_vector<float2> grad(32);
  thrust::host_vector<unsigned int> fvalue_h(32);
  thrust::device_vector<unsigned int> fvalue_d(32);
  float2 sum = make_float2(0.0, 0.0);

  thrust::device_vector<mydouble2> parent_node_sum(3, make_double2(0, 0));

  thrust::device_vector<unsigned int> parent_node_count(3, 0);

  for (int i = 0; i < size; ++i) {
    row2Node[i] = i % 2;
    grad[i] = make_float2(i * i, 1.0);
    fvalue_h[i] = fvalue_d[i] = i / 4;
    mydouble2 tmp = mydouble2(grad[i]);
    parent_node_sum[(i / 16) + 1] += tmp;
    // parent_node_count[(i / 16) + 1] += 1;
  }
  parent_node_count[1] += 16;
  parent_node_count[2] += 32;

  parent_node_sum[1] += parent_node_sum[0];
  parent_node_sum[2] += parent_node_sum[1];

  auto p = GainFunctionParameters(0, 0, 0, 0, 0, 0);

  grower.ProcessDenseFeature<unsigned int>(
    row2Node, grad, thrust::raw_pointer_cast(fvalue_d.data()),
    thrust::raw_pointer_cast(fvalue_h.data()), parent_node_sum,
    parent_node_count, 3, 1, 2, p, false);
  TEST_OK(cudaStreamSynchronize(grower.stream));

  thrust::host_vector<my_atomics> result_h = grower.result_d;
  // copy pasted result
  ASSERT_EQ(result_h[0].ints[1], 10);
  ASSERT_FLOAT_EQ(result_h[0].floats[0], 64026.672);

  ASSERT_EQ(result_h[1].ints[1], 24);
  ASSERT_FLOAT_EQ(result_h[1].floats[0], 565504);

  ASSERT_EQ(grower.node_fvalue[0], 0);
  ASSERT_EQ(grower.node_fvalue[1], 0);
  ASSERT_EQ(grower.node_fvalue[2], 1);
  ASSERT_EQ(grower.node_fvalue[3], 1);
  ASSERT_EQ(grower.node_fvalue[4], 2);
  ASSERT_EQ(grower.node_fvalue[5], 2);
  ASSERT_EQ(grower.node_fvalue[6], 3);
  ASSERT_EQ(grower.node_fvalue[7], 3);
  ASSERT_EQ(grower.node_fvalue[8], 4);
  ASSERT_EQ(grower.node_fvalue[9], 4);
  ASSERT_EQ(grower.node_fvalue[10], 5);
  ASSERT_EQ(grower.node_fvalue[11], 5);
  ASSERT_EQ(grower.node_fvalue[12], 6);
  ASSERT_EQ(grower.node_fvalue[13], 6);
  ASSERT_EQ(grower.node_fvalue[14], 7);
  ASSERT_EQ(grower.node_fvalue[15], 7);
  ASSERT_EQ(grower.node_fvalue[16], 7);
  ASSERT_EQ(grower.node_fvalue[17], 7);
  ASSERT_EQ(grower.node_fvalue[18], 6);
  ASSERT_EQ(grower.node_fvalue[19], 6);
  ASSERT_EQ(grower.node_fvalue[20], 5);
  ASSERT_EQ(grower.node_fvalue[21], 5);
  ASSERT_EQ(grower.node_fvalue[22], 4);
  ASSERT_EQ(grower.node_fvalue[23], 4);
  ASSERT_EQ(grower.node_fvalue[24], 3);
  ASSERT_EQ(grower.node_fvalue[25], 3);
  ASSERT_EQ(grower.node_fvalue[26], 2);
  ASSERT_EQ(grower.node_fvalue[27], 2);
  ASSERT_EQ(grower.node_fvalue[28], 1);
  ASSERT_EQ(grower.node_fvalue[29], 1);
  ASSERT_EQ(grower.node_fvalue[30], 0);
  ASSERT_EQ(grower.node_fvalue[31], 0);
}

TEST(ContinuousTreeGrower, PartitionLevel1) {
  const size_t size = 32;
  auto grower = ContinuousTreeGrower<unsigned int, float2, float2>(size, 2, 0);
  thrust::device_vector<float2> grad(size, make_float2(0, 0));
  thrust::device_vector<unsigned> row2Node(size, 0);
  thrust::device_vector<unsigned> parent_node_count(3, 0);
  parent_node_count[1] = 16;
  parent_node_count[2] = size;

  for (size_t i = 0; i < grad.size(); i++) {
    row2Node[i] = i % 2;
    grad[i] = make_float2(float(i), 1.0);
  }

  grower.Partition(thrust::raw_pointer_cast(grad.data()),
                   thrust::raw_pointer_cast(row2Node.data()), parent_node_count,
                   1, 2);
  TEST_OK(cudaStreamSynchronize(grower.stream));

  thrust::host_vector<float2> grad_h = grad;

  for (size_t i = 0; i < size / 2; i++) {
    ASSERT_FLOAT_EQ(grad_h[i].x, float(i * 2));
    ASSERT_FLOAT_EQ(grad_h[i].y, float(1));
  }
  for (size_t i = size / 2; i < size; i++) {
    float g = float((size - i) * 2 - 1);
    ASSERT_FLOAT_EQ(grad_h[i].x, g);
    ASSERT_FLOAT_EQ(grad_h[i].y, float(1));
  }
}

TEST(ContinuousTreeGrower, ApplySplitLevel0) {
  const size_t size = 32;
  auto grower = ContinuousTreeGrower<unsigned int, float2, float2>(size, 2, 0);
  thrust::device_vector<unsigned int> row2Node(size, 0);

  row2Node[3] = 1;
  row2Node[4] = 1;
  row2Node[6] = 1;

  grower.node_fvalue[0] = 0;
  grower.node_fvalue[1] = 5;
  grower.node_fvalue[2] = 2;
  grower.node_fvalue[3] = 5;
  grower.node_fvalue[4] = 3;
  grower.node_fvalue[5] = 4;
  grower.node_fvalue[6] = 4;

  grower.ApplySplit(thrust::raw_pointer_cast(row2Node.data()), 0, 4, 0, size);
  TEST_OK(cudaStreamSynchronize(grower.stream));
  thrust::host_vector<unsigned> nodes = row2Node;

  ASSERT_EQ(nodes[0], 0);
  ASSERT_EQ(nodes[1], 1);
  ASSERT_EQ(nodes[2], 0);
  ASSERT_EQ(nodes[3], 1);
  ASSERT_EQ(nodes[4], 0);
  ASSERT_EQ(nodes[5], 1);
  ASSERT_EQ(nodes[6], 1);

  for (int i = 7; i < size; ++i) {
    ASSERT_EQ(nodes[i], 0);
  }
}

TEST(ContinuousTreeGrower, ApplySplitLevel1) {
  const size_t size = 32;
  auto grower = ContinuousTreeGrower<unsigned int, float2, float2>(size, 2, 0);
  thrust::device_vector<unsigned int> row2Node(size, 3);

  row2Node[3] = 1;
  row2Node[4] = 1;
  row2Node[6] = 1;

  grower.node_fvalue[0] = 0;
  grower.node_fvalue[1] = 5;
  grower.node_fvalue[2] = 2;
  grower.node_fvalue[3] = 5;
  grower.node_fvalue[4] = 3;
  grower.node_fvalue[5] = 4;
  grower.node_fvalue[6] = 4;

  grower.ApplySplit(thrust::raw_pointer_cast(row2Node.data()), 1, 4, 1, 7);
  TEST_OK(cudaStreamSynchronize(grower.stream));
  thrust::host_vector<unsigned> nodes = row2Node;

  ASSERT_EQ(nodes[0], 3);
  ASSERT_EQ(nodes[1], 3);
  ASSERT_EQ(nodes[2], 1);
  ASSERT_EQ(nodes[3], 3);
  ASSERT_EQ(nodes[4], 1);
  ASSERT_EQ(nodes[5], 3);
  ASSERT_EQ(nodes[6], 3);

  for (int i = 7; i < size; ++i) {
    ASSERT_EQ(nodes[i], 3);
  }
}

TEST(ContinuousTreeGrower, PartitioningLevel1) {
  const size_t size = 32;
  auto grower = ContinuousTreeGrower<unsigned int, float, float>(size, 2, 0);
  thrust::device_vector<unsigned int> row2Node(size, 0);
  thrust::device_vector<float> src(size);
  thrust::device_vector<unsigned> counts(3);
  counts[0] = 0;
  counts[1] = size / 2;
  counts[2] = size;

  for (size_t i = 0; i < size; i++) {
    row2Node[i] = i % 2;
    src[i] = float(i);
  }

  grower.Partition(thrust::raw_pointer_cast(src.data()),
                   thrust::raw_pointer_cast(row2Node.data()), counts, 1, 2);
  TEST_OK(cudaStreamSynchronize(grower.stream));
  thrust::host_vector<float> src_h = src;

  for (size_t i = 0; i < size / 2; i++) {
    ASSERT_FLOAT_EQ(src_h[i], float(i * 2)) << "at" << i;
    ASSERT_FLOAT_EQ(src_h[i + size / 2], float(size - 1 - 2 * i))
      << "at" << i + size / 2;
  }
}
}  // namespace arboretum_test