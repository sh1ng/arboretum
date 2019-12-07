#include "core/builder.h"
#include "core/hist_grad_sum.cuh"
#include "core/hist_tree_grower.h"
#include "gtest/gtest.h"
#include "test_utils.h"
#include "thrust/device_vector.h"
#include "thrust/host_vector.h"

namespace arboretum_test {

TEST(SingleNodeHistSumFloat, Naive) {
  const size_t size = 1 << 5;
  thrust::device_vector<float> grad(size);
  thrust::device_vector<float> sum(size, 0.0);
  thrust::device_vector<unsigned> count(size, 0);
  thrust::device_vector<unsigned short> bin(size);
  thrust::device_vector<unsigned> node_size(2);
  node_size[0] = 0;
  node_size[1] = size;
  for (unsigned i = 0; i < size; ++i) {
    grad[i] = float(i);
    bin[i] = i;
  }

  arboretum::core::HistTreeGrower<unsigned, float, float>::HistSumSingleNode(
    thrust::raw_pointer_cast(sum.data()),
    thrust::raw_pointer_cast(count.data()),
    thrust::raw_pointer_cast(grad.data()),
    thrust::raw_pointer_cast(node_size.data()),
    thrust::raw_pointer_cast(bin.data()), 6, size);

  TEST_OK(cudaDeviceSynchronize());
  TEST_OK(cudaGetLastError());

  for (unsigned i = 0; i < size; ++i) {
    ASSERT_EQ(count[i], 1);
    ASSERT_FLOAT_EQ(grad[i], sum[i]);
  }
}

TEST(SingleNodeHistSumFloat, SingleSegment) {
  const size_t size = 1 << 5;
  thrust::device_vector<float> grad(size);
  thrust::device_vector<float> sum(size, 0.0);
  thrust::device_vector<unsigned> count(size, 0);
  thrust::device_vector<unsigned short> bin(size);
  thrust::device_vector<unsigned> node_size(2);
  node_size[0] = 0;
  node_size[1] = size;
  for (unsigned i = 0; i < size; ++i) {
    grad[i] = float(i);
    bin[i] = 0;
  }

  arboretum::core::HistTreeGrower<unsigned, float, float>::HistSumSingleNode(
    thrust::raw_pointer_cast(sum.data()),
    thrust::raw_pointer_cast(count.data()),
    thrust::raw_pointer_cast(grad.data()),
    thrust::raw_pointer_cast(node_size.data()),
    thrust::raw_pointer_cast(bin.data()), 6, size);

  TEST_OK(cudaDeviceSynchronize());
  TEST_OK(cudaGetLastError());

  ASSERT_EQ(count[0], size);
  // sum of 0 + 1 + .. + size-1
  float true_sum = size * (size - 1) / 2;
  ASSERT_FLOAT_EQ(sum[0], float(true_sum));
  //   }
}

TEST(SingleNodeHistSumFloat, SingleSegmentFullSize) {
  const size_t size = HIST_SUM_BLOCK_DIM * HIST_SUM_ITEMS_PER_THREAD;

  thrust::device_vector<float> grad(size);
  thrust::device_vector<float> sum(size, 0.0);
  thrust::device_vector<unsigned> count(size, 0);
  thrust::device_vector<unsigned short> bin(size);
  thrust::device_vector<unsigned> node_size(2);
  node_size[0] = 0;
  node_size[1] = size;
  for (unsigned i = 0; i < size; ++i) {
    grad[i] = float(i);
    bin[i] = 0 % 1024;
  }

  arboretum::core::HistTreeGrower<unsigned, float, float>::HistSumSingleNode(
    thrust::raw_pointer_cast(sum.data()),
    thrust::raw_pointer_cast(count.data()),
    thrust::raw_pointer_cast(grad.data()),
    thrust::raw_pointer_cast(node_size.data()),
    thrust::raw_pointer_cast(bin.data()), 11, size);

  TEST_OK(cudaDeviceSynchronize());
  TEST_OK(cudaGetLastError());

  ASSERT_EQ(count[0], size);
  // sum of 0 + 1 + .. + size-1
  float true_sum = size * (size - 1) / 2;
  ASSERT_FLOAT_EQ(sum[0], float(true_sum));
  //   }
}

TEST(SingleNodeHistSumDouble, Naive) {
  const size_t size = 1 << 5;
  thrust::device_vector<float> grad(size);
  thrust::device_vector<double> sum(size, 0.0);
  thrust::device_vector<unsigned> count(size, 0);
  thrust::device_vector<unsigned short> bin(size);
  thrust::device_vector<unsigned> node_size(2);
  node_size[0] = 0;
  node_size[1] = size;
  for (unsigned i = 0; i < size; ++i) {
    grad[i] = float(i);
    bin[i] = i;
  }

  arboretum::core::HistTreeGrower<unsigned, float, double>::HistSumSingleNode(
    thrust::raw_pointer_cast(sum.data()),
    thrust::raw_pointer_cast(count.data()),
    thrust::raw_pointer_cast(grad.data()),
    thrust::raw_pointer_cast(node_size.data()),
    thrust::raw_pointer_cast(bin.data()), 6, size);

  TEST_OK(cudaDeviceSynchronize());
  TEST_OK(cudaGetLastError());

  for (unsigned i = 0; i < size; ++i) {
    ASSERT_EQ(count[i], 1);
    ASSERT_DOUBLE_EQ(grad[i], sum[i]);
  }
}

TEST(SingleNodeHistSumDouble, SingleSegment) {
  const size_t size = 1 << 5;
  thrust::device_vector<float> grad(size);
  thrust::device_vector<double> sum(size, 0.0);
  thrust::device_vector<unsigned> count(size, 0);
  thrust::device_vector<unsigned short> bin(size);
  thrust::device_vector<unsigned> node_size(2);
  node_size[0] = 0;
  node_size[1] = size;
  for (unsigned i = 0; i < size; ++i) {
    grad[i] = float(i);
    bin[i] = 0;
  }

  arboretum::core::HistTreeGrower<unsigned, float, double>::HistSumSingleNode(
    thrust::raw_pointer_cast(sum.data()),
    thrust::raw_pointer_cast(count.data()),
    thrust::raw_pointer_cast(grad.data()),
    thrust::raw_pointer_cast(node_size.data()),
    thrust::raw_pointer_cast(bin.data()), 6, size);

  TEST_OK(cudaDeviceSynchronize());
  TEST_OK(cudaGetLastError());

  ASSERT_EQ(count[0], size);
  // sum of 0 + 1 + .. + size-1
  double true_sum = size * (size - 1) / 2;
  ASSERT_DOUBLE_EQ(sum[0], true_sum);
  //   }
}

TEST(SingleNodeHistSumDouble, SingleSegmentFullSize) {
  const size_t size = HIST_SUM_BLOCK_DIM * HIST_SUM_ITEMS_PER_THREAD;

  thrust::device_vector<float> grad(size);
  thrust::device_vector<double> sum(size, 0.0);
  thrust::device_vector<unsigned> count(size, 0);
  thrust::device_vector<unsigned short> bin(size);
  thrust::device_vector<unsigned> node_size(2);
  node_size[0] = 0;
  node_size[1] = size;
  for (unsigned i = 0; i < size; ++i) {
    grad[i] = float(i);
    bin[i] = 0 % 1024;
  }

  arboretum::core::HistTreeGrower<unsigned, float, double>::HistSumSingleNode(
    thrust::raw_pointer_cast(sum.data()),
    thrust::raw_pointer_cast(count.data()),
    thrust::raw_pointer_cast(grad.data()),
    thrust::raw_pointer_cast(node_size.data()),
    thrust::raw_pointer_cast(bin.data()), 11, size);

  TEST_OK(cudaDeviceSynchronize());
  TEST_OK(cudaGetLastError());

  ASSERT_EQ(count[0], size);
  // sum of 0 + 1 + .. + size-1
  double true_sum = size * (size - 1) / 2;
  ASSERT_DOUBLE_EQ(sum[0], true_sum);
  //   }
}

// TEST(MultiNodeHistSumDouble, _2_NodesNoTrick) {
//   const unsigned hist_size = 4;
//   const size_t size = HIST_SUM_BLOCK_DIM * HIST_SUM_ITEMS_PER_THREAD;

//   thrust::device_vector<float> grad(size);
//   thrust::device_vector<double> sum(hist_size * 2, 0.0);
//   thrust::device_vector<unsigned> count(hist_size * 2, 0);
//   thrust::device_vector<unsigned short> bin(size);
//   thrust::device_vector<unsigned> node_size(3);
//   node_size[0] = 0;
//   node_size[1] = size / 2;
//   node_size[2] = size;
//   for (unsigned i = 0; i < size; ++i) {
//     grad[i] = float(i);
//     bin[i] = i % hist_size;
//   }

//   arboretum::core::HistTreeGrower<unsigned, float, double>::HistSum(
//     thrust::raw_pointer_cast(sum.data()),
//     thrust::raw_pointer_cast(count.data()), NULL, NULL,
//     thrust::raw_pointer_cast(grad.data()),
//     thrust::raw_pointer_cast(node_size.data()),
//     thrust::raw_pointer_cast(bin.data()), 10, hist_size, 2, false);

//   TEST_OK(cudaDeviceSynchronize());
//   TEST_OK(cudaGetLastError());

//   for (int i = 0; i < hist_size * 2; ++i) {
//     ASSERT_EQ(count[i], size / 8) << i;
//   }
//   // sum of 0 + 1 + .. + size / 2 -1
//   double true_sum = (size / 2) * (size / 2 - 1) / 2;
//   ASSERT_DOUBLE_EQ(sum[0] + sum[1] + sum[2] + sum[3], true_sum);

//   // sum of size / 2 + ... + size -1

//   true_sum = (size / 2) * (size / 2 + size - 1) / 2;
//   ASSERT_DOUBLE_EQ(sum[hist_size] + sum[hist_size + 1] + sum[hist_size + 2] +
//                      sum[hist_size + 3],
//                    true_sum);
//   //   }
// }

// TEST(MultiNodeHistSumDouble, _2_NodesAsymmetricNoTrick) {
//   const unsigned hist_size = 4;
//   const size_t size = HIST_SUM_BLOCK_DIM * HIST_SUM_ITEMS_PER_THREAD;

//   thrust::device_vector<float> grad(size);
//   thrust::device_vector<double> sum(hist_size * 2, 0.0);
//   thrust::device_vector<unsigned> count(hist_size * 2, 0);
//   thrust::device_vector<unsigned short> bin(size);
//   thrust::device_vector<unsigned> node_size(3);
//   node_size[0] = 0;
//   node_size[1] = 10;
//   node_size[2] = size;
//   for (unsigned i = 0; i < size; ++i) {
//     grad[i] = float(i);
//     bin[i] = i % hist_size;
//   }

//   arboretum::core::HistTreeGrower<unsigned, float, double>::HistSum(
//     thrust::raw_pointer_cast(sum.data()),
//     thrust::raw_pointer_cast(count.data()), NULL, NULL,
//     thrust::raw_pointer_cast(grad.data()),
//     thrust::raw_pointer_cast(node_size.data()),
//     thrust::raw_pointer_cast(bin.data()), 10, hist_size, 2, false);

//   TEST_OK(cudaDeviceSynchronize());
//   TEST_OK(cudaGetLastError());

//   ASSERT_EQ(count[0], 3);
//   ASSERT_EQ(count[1], 3);
//   ASSERT_EQ(count[2], 2);
//   ASSERT_EQ(count[3], 2);

//   ASSERT_EQ(count[4], 317);
//   ASSERT_EQ(count[5], 317);
//   ASSERT_EQ(count[6], 318);
//   ASSERT_EQ(count[7], 318);

//   ASSERT_DOUBLE_EQ(sum[0], 12.0);
//   ASSERT_DOUBLE_EQ(sum[1], 15.0);
//   ASSERT_DOUBLE_EQ(sum[2], 8.0);
//   ASSERT_DOUBLE_EQ(sum[3], 10.0);

//   ASSERT_DOUBLE_EQ(sum[4], 204148.0);
//   ASSERT_DOUBLE_EQ(sum[5], 204465.0);
//   ASSERT_DOUBLE_EQ(sum[6], 204792.0);
//   ASSERT_DOUBLE_EQ(sum[7], 205110.0);
//   double true_sum = (size) * (size - 1) / 2;
//   ASSERT_DOUBLE_EQ(sum[0] + sum[1] + sum[2] + sum[3] + sum[hist_size] +
//                      sum[hist_size + 1] + sum[hist_size + 2] +
//                      sum[hist_size + 3],
//                    true_sum);
// }

// TEST(MultiNodeHistSumDouble, _2_NodesAsymmetricWithTrick) {
//   const unsigned hist_size = 4;
//   const size_t size = HIST_SUM_BLOCK_DIM * HIST_SUM_ITEMS_PER_THREAD;

//   thrust::device_vector<float> grad(size);
//   thrust::device_vector<double> sum(hist_size * 2, 0.0);
//   thrust::device_vector<unsigned> count(hist_size * 2, 0);
//   thrust::device_vector<unsigned short> bin(size);
//   thrust::device_vector<unsigned> node_size(3);
//   node_size[0] = 0;
//   node_size[1] = 10;
//   node_size[2] = size;
//   for (unsigned i = 0; i < size; ++i) {
//     grad[i] = float(i);
//     bin[i] = i % hist_size;
//   }

//   thrust::device_vector<unsigned> parent_count(hist_size, 0);
//   thrust::device_vector<double> parent_sum(hist_size, 0);

//   parent_count[0] = parent_count[1] = parent_count[2] = parent_count[3] =
//   320;
//   // sum of 0 + 1 + .. + size / 2 -1

//   parent_sum[0] = 204148.0 + 12.0;
//   parent_sum[1] = 204465.0 + 15.0;
//   parent_sum[2] = 204792.0 + 8.0;
//   parent_sum[3] = 205110.0 + 10.0;

//   arboretum::core::HistTreeGrower<unsigned, float, double>::HistSum(
//     thrust::raw_pointer_cast(sum.data()),
//     thrust::raw_pointer_cast(count.data()),
//     thrust::raw_pointer_cast(parent_sum.data()),
//     thrust::raw_pointer_cast(parent_count.data()),
//     thrust::raw_pointer_cast(grad.data()),
//     thrust::raw_pointer_cast(node_size.data()),
//     thrust::raw_pointer_cast(bin.data()), 10, hist_size, 2, true);

//   TEST_OK(cudaDeviceSynchronize());
//   TEST_OK(cudaGetLastError());

//   ASSERT_EQ(count[0], 3);
//   ASSERT_EQ(count[1], 3);
//   ASSERT_EQ(count[2], 2);
//   ASSERT_EQ(count[3], 2);

//   ASSERT_EQ(count[4], 317);
//   ASSERT_EQ(count[5], 317);
//   ASSERT_EQ(count[6], 318);
//   ASSERT_EQ(count[7], 318);

//   ASSERT_DOUBLE_EQ(sum[0], 12.0);
//   ASSERT_DOUBLE_EQ(sum[1], 15.0);
//   ASSERT_DOUBLE_EQ(sum[2], 8.0);
//   ASSERT_DOUBLE_EQ(sum[3], 10.0);

//   ASSERT_DOUBLE_EQ(sum[4], 204148.0);
//   ASSERT_DOUBLE_EQ(sum[5], 204465.0);
//   ASSERT_DOUBLE_EQ(sum[6], 204792.0);
//   ASSERT_DOUBLE_EQ(sum[7], 205110.0);
// }

// TEST(MultiNodeHistSumDouble, _2_NodesAsymmetricWithTrick2) {
//   const unsigned hist_size = 4;
//   const size_t size = HIST_SUM_BLOCK_DIM * HIST_SUM_ITEMS_PER_THREAD;

//   thrust::device_vector<float> grad(size);
//   thrust::device_vector<double> sum(hist_size * 2, 0.0);
//   thrust::device_vector<unsigned> count(hist_size * 2, 0);
//   thrust::device_vector<unsigned short> bin(size);
//   thrust::device_vector<unsigned> node_size(3);
//   node_size[0] = 0;
//   node_size[1] = size - 10;
//   node_size[2] = size;
//   for (unsigned i = 0; i < size; ++i) {
//     grad[i] = float(i);
//     bin[i] = i % hist_size;
//   }

//   thrust::device_vector<unsigned> parent_count(hist_size, 0);
//   thrust::device_vector<double> parent_sum(hist_size, 0);

//   parent_count[0] = parent_count[1] = parent_count[2] = parent_count[3] =
//   320;
//   // sum of 0 + 1 + .. + size / 2 -1

//   parent_sum[0] = 204148.0 + 12.0;
//   parent_sum[1] = 204465.0 + 15.0;
//   parent_sum[2] = 204792.0 + 8.0;
//   parent_sum[3] = 205110.0 + 10.0;

//   arboretum::core::HistTreeGrower<unsigned, float, double>::HistSum(
//     thrust::raw_pointer_cast(sum.data()),
//     thrust::raw_pointer_cast(count.data()),
//     thrust::raw_pointer_cast(parent_sum.data()),
//     thrust::raw_pointer_cast(parent_count.data()),
//     thrust::raw_pointer_cast(grad.data()),
//     thrust::raw_pointer_cast(node_size.data()),
//     thrust::raw_pointer_cast(bin.data()), 10, hist_size, 2, true);

//   TEST_OK(cudaDeviceSynchronize());
//   TEST_OK(cudaGetLastError());

//   ASSERT_EQ(count[0], 318);
//   ASSERT_EQ(count[1], 318);
//   ASSERT_EQ(count[2], 317);
//   ASSERT_EQ(count[3], 317);

//   ASSERT_EQ(count[4], 2);
//   ASSERT_EQ(count[5], 2);
//   ASSERT_EQ(count[6], 3);
//   ASSERT_EQ(count[7], 3);

//   ASSERT_DOUBLE_EQ(sum[0], 201612.0);
//   ASSERT_DOUBLE_EQ(sum[1], 201930.0);
//   ASSERT_DOUBLE_EQ(sum[2], 200978.0);
//   ASSERT_DOUBLE_EQ(sum[3], 201295.0);

//   ASSERT_DOUBLE_EQ(sum[4], 2548.0);
//   ASSERT_DOUBLE_EQ(sum[5], 2550.0);
//   ASSERT_DOUBLE_EQ(sum[6], 3822.0);
//   ASSERT_DOUBLE_EQ(sum[7], 3825.0);
// }

// TEST(MultiNodeHistSumDouble, SingleSegmentWithTrick) {
//   const unsigned hist_size = 4;
//   const size_t size = HIST_SUM_BLOCK_DIM * HIST_SUM_ITEMS_PER_THREAD;

//   thrust::device_vector<float> grad(size);
//   thrust::device_vector<double> sum(hist_size * 2, 0.0);
//   thrust::device_vector<unsigned> count(hist_size * 2, 0);
//   thrust::device_vector<unsigned short> bin(size);
//   thrust::device_vector<unsigned> node_size(3);
//   node_size[0] = 0;
//   node_size[1] = size / 2;
//   node_size[2] = size;
//   for (unsigned i = 0; i < size; ++i) {
//     grad[i] = float(i);
//     bin[i] = 0;
//   }

//   thrust::device_vector<unsigned> parent_count(hist_size, 0);
//   thrust::device_vector<double> parent_sum(hist_size, 0);
//   parent_count[0] = size;
//   // sum of 0 + 1 + .. + size / 2 -1
//   double true_sum = (size) * (size - 1) / 2;
//   parent_sum[0] = true_sum;

//   arboretum::core::HistTreeGrower<unsigned, float, double>::HistSum(
//     thrust::raw_pointer_cast(sum.data()),
//     thrust::raw_pointer_cast(count.data()),
//     thrust::raw_pointer_cast(parent_sum.data()),
//     thrust::raw_pointer_cast(parent_count.data()),
//     thrust::raw_pointer_cast(grad.data()),
//     thrust::raw_pointer_cast(node_size.data()),
//     thrust::raw_pointer_cast(bin.data()), 10, hist_size, 2, true);

//   TEST_OK(cudaDeviceSynchronize());
//   TEST_OK(cudaGetLastError());

//   ASSERT_EQ(count[0], size / 2);
//   ASSERT_EQ(count[hist_size], size / 2);
//   // sum of 0 + 1 + .. + size / 2 -1
//   true_sum = (size / 2) * (size / 2 - 1) / 2;

//   ASSERT_DOUBLE_EQ(sum[0], true_sum);

//   // sum of size / 2 + ... + size -1

//   true_sum = (size / 2) * (size / 2 + size - 1) / 2;
//   ASSERT_DOUBLE_EQ(sum[hist_size], true_sum);
//   //   }
// }

}  // namespace arboretum_test