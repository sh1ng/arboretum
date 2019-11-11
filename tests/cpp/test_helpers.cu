#include "core/cuda_helpers.h"
#include "gtest/gtest.h"
#include "test_utils.h"
#include "thrust/host_vector.h"

namespace arboretum_test {

TEST(Helpers, linear_search) {
  int segments[7] = {0, 5, 5, 10, 10, 20, 20};
  int segment = linear_search(segments, 4, 7);
  ASSERT_EQ(segment, 0);

  segment = linear_search(segments, 5, 7);
  ASSERT_EQ(segment, 2);

  segment = linear_search(segments, 9, 7);
  ASSERT_EQ(segment, 2);

  segment = linear_search(segments, 10, 7);
  ASSERT_EQ(segment, 4);

  segment = linear_search(segments, 19, 7);
  ASSERT_EQ(segment, 4);
}

TEST(Helpers, binary_search) {
  int segments[7] = {0, 5, 5, 10, 10, 20, 20};
  int segment = binary_search(segments, 4, 7);
  ASSERT_EQ(segment, 0);

  segment = binary_search(segments, 5, 7);
  ASSERT_EQ(segment, 2);

  segment = binary_search(segments, 9, 7);
  ASSERT_EQ(segment, 2);

  segment = binary_search(segments, 10, 7);
  ASSERT_EQ(segment, 4);

  segment = binary_search(segments, 19, 7);
  ASSERT_EQ(segment, 4);
}
}  // namespace arboretum_test