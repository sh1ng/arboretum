#include "io/io.h"
#include "gtest/gtest.h"

namespace arboretum_test {
using arboretum::io::DataMatrix;

TEST(DataMatrix, DataReduce) {
  const int rows = 10;
  const int cols = 2;
  const int category_cols = 1;
  auto data = DataMatrix(rows, cols, category_cols);
  for (int i = 0; i < rows * cols; ++i) {
    data.data[i / rows][i % rows] = float(i % 2);
    data.data_categories[0][i % rows] = i % rows;
  }

  data.Init(false);

  ASSERT_EQ(data.max_feature_size, 4);
  ASSERT_EQ(data.reduced_size[0], 2);
  ASSERT_EQ(data.reduced_size[1], 2);
  ASSERT_EQ(data.category_size[0], 4);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
} // namespace arboretum_test