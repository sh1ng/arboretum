#include "gtest/gtest.h"
#include "io/io.h"

namespace arboretum_test {
using arboretum::io::DataMatrix;

TEST(DataMatrix, ExactDataReduce) {
  const int rows = 10;
  const int cols = 2;
  const int category_cols = 1;
  auto data = DataMatrix(rows, cols, category_cols);
  for (int i = 0; i < rows * cols; ++i) {
    data.data[i / rows][i % rows] = float(i % 2);
    data.data_categories[0][i % rows] = i % rows;
  }

  data.InitExact(false);

  ASSERT_EQ(data.max_feature_size, 4);
  ASSERT_EQ(data.reduced_size[0], 2);
  ASSERT_EQ(data.reduced_size[1], 2);
  ASSERT_EQ(data.category_size[0], 4);
}

TEST(DataMatrix, HistWideRange) {
  const int rows = 128;
  const int cols = 1;
  const int category_cols = 0;
  const int hist_size = 1 << 3;

  auto data = DataMatrix(rows, cols, category_cols);
  for (int i = 0; i < rows * cols; ++i) {
    data.data[i / rows][i % rows] = float(128 - i);
  }

  data.InitHist(hist_size, false);

  ASSERT_EQ(data.max_feature_size, 4);
  ASSERT_EQ(data.reduced_size[0], 4);
  for (auto i = 0; i < rows; ++i) {
    ASSERT_EQ(data.GetHostData<unsigned char>(0)[i], hist_size - (i / 16) - 1);
  }
}

TEST(DataMatrix, Hist255) {
  const int rows = 1024;
  const int cols = 1;
  const int category_cols = 0;
  const int hist_size = 255;

  auto data = DataMatrix(rows, cols, category_cols);
  for (int i = 0; i < rows * cols; ++i) {
    data.data[i / rows][i % rows] = float(i);
  }

  data.InitHist(hist_size, false);

  ASSERT_EQ(data.max_feature_size, 8);
  ASSERT_EQ(data.reduced_size[0], 8);
  int add = 0;
  for (auto i = 0; i < rows; ++i) {
    if (i == 256) add++;
    if (i == 513) add++;
    if (i == 770) add++;
    if (i == 1023) add++;
    ASSERT_EQ(data.GetHostData<unsigned char>(0)[i],
              ((i - add) / (rows / hist_size)));
  }
}

TEST(DataMatrix, HistSkewedRange) {
  const int rows = 12;
  const int cols = 1;
  const int category_cols = 0;
  const int hist_size = 1 << 2;

  auto data = DataMatrix(rows, cols, category_cols);
  data.data[0].resize(rows);
  data.data[0][0] = 10.0;
  data.data[0][1] = 0.0;
  data.data[0][2] = 11.0;
  data.data[0][3] = 1.0;
  data.data[0][4] = 0.0;
  data.data[0][5] = 2.0;
  data.data[0][6] = 3.0;
  data.data[0][7] = 0.0;
  data.data[0][8] = 0.0;
  data.data[0][9] = 0.0;
  data.data[0][10] = 0.0;
  data.data[0][11] = 0.0;

  data.InitHist(hist_size, false);

  ASSERT_EQ(data.max_feature_size, 3);
  ASSERT_EQ(data.reduced_size[0], 3);

  ASSERT_EQ(data.GetHostData<unsigned char>(0)[0], 3);
  ASSERT_EQ(data.GetHostData<unsigned char>(0)[1], 0);
  ASSERT_EQ(data.GetHostData<unsigned char>(0)[2], 3);
  ASSERT_EQ(data.GetHostData<unsigned char>(0)[3], 1);
  ASSERT_EQ(data.GetHostData<unsigned char>(0)[4], 0);
  ASSERT_EQ(data.GetHostData<unsigned char>(0)[5], 1);
  ASSERT_EQ(data.GetHostData<unsigned char>(0)[6], 2);
  ASSERT_EQ(data.GetHostData<unsigned char>(0)[7], 0);
  ASSERT_EQ(data.GetHostData<unsigned char>(0)[8], 0);
  ASSERT_EQ(data.GetHostData<unsigned char>(0)[9], 0);
  ASSERT_EQ(data.GetHostData<unsigned char>(0)[10], 0);
  ASSERT_EQ(data.GetHostData<unsigned char>(0)[11], 0);
}

TEST(DataMatrix, HistTinyRange1) {
  const int rows = 12;
  const int cols = 1;
  const int category_cols = 0;
  const int hist_size = 1 << 2;

  auto data = DataMatrix(rows, cols, category_cols);
  data.data[0].resize(rows);
  data.data[0][0] = 0.0;
  data.data[0][1] = 0.0;
  data.data[0][2] = 0.0;
  data.data[0][3] = 1.0;
  data.data[0][4] = 0.0;
  data.data[0][5] = 2.0;
  data.data[0][6] = 0.0;
  data.data[0][7] = 0.0;
  data.data[0][8] = 0.0;
  data.data[0][9] = 0.0;
  data.data[0][10] = 0.0;
  data.data[0][11] = 0.0;

  data.InitHist(hist_size, false);

  ASSERT_EQ(data.max_feature_size, 2);
  ASSERT_EQ(data.reduced_size[0], 2);

  ASSERT_EQ(data.GetHostData<unsigned char>(0)[0], 0);
  ASSERT_EQ(data.GetHostData<unsigned char>(0)[1], 0);
  ASSERT_EQ(data.GetHostData<unsigned char>(0)[2], 0);
  ASSERT_EQ(data.GetHostData<unsigned char>(0)[3], 1);
  ASSERT_EQ(data.GetHostData<unsigned char>(0)[4], 0);
  ASSERT_EQ(data.GetHostData<unsigned char>(0)[5], 2);
  ASSERT_EQ(data.GetHostData<unsigned char>(0)[6], 0);
  ASSERT_EQ(data.GetHostData<unsigned char>(0)[7], 0);
  ASSERT_EQ(data.GetHostData<unsigned char>(0)[8], 0);
  ASSERT_EQ(data.GetHostData<unsigned char>(0)[9], 0);
  ASSERT_EQ(data.GetHostData<unsigned char>(0)[10], 0);
  ASSERT_EQ(data.GetHostData<unsigned char>(0)[11], 0);
}

TEST(DataMatrix, HistTinyRange2) {
  const int rows = 12;
  const int cols = 1;
  const int category_cols = 0;
  const int hist_size = 1 << 2;

  auto data = DataMatrix(rows, cols, category_cols);
  data.data[0].resize(rows);
  data.data[0][0] = 10.0;
  data.data[0][1] = 10.0;
  data.data[0][2] = 10.0;
  data.data[0][3] = 1.0;
  data.data[0][4] = 10.0;
  data.data[0][5] = 2.0;
  data.data[0][6] = 10.0;
  data.data[0][7] = 10.0;
  data.data[0][8] = 10.0;
  data.data[0][9] = 10.0;
  data.data[0][10] = 10.0;
  data.data[0][11] = 10.0;

  data.InitHist(hist_size, false);

  ASSERT_EQ(data.max_feature_size, 2);
  ASSERT_EQ(data.reduced_size[0], 2);

  ASSERT_EQ(data.GetHostData<unsigned char>(0)[0], 2);
  ASSERT_EQ(data.GetHostData<unsigned char>(0)[1], 2);
  ASSERT_EQ(data.GetHostData<unsigned char>(0)[2], 2);
  ASSERT_EQ(data.GetHostData<unsigned char>(0)[3], 0);
  ASSERT_EQ(data.GetHostData<unsigned char>(0)[4], 2);
  ASSERT_EQ(data.GetHostData<unsigned char>(0)[5], 1);
  ASSERT_EQ(data.GetHostData<unsigned char>(0)[6], 2);
  ASSERT_EQ(data.GetHostData<unsigned char>(0)[7], 2);
  ASSERT_EQ(data.GetHostData<unsigned char>(0)[8], 2);
  ASSERT_EQ(data.GetHostData<unsigned char>(0)[9], 2);
  ASSERT_EQ(data.GetHostData<unsigned char>(0)[10], 2);
  ASSERT_EQ(data.GetHostData<unsigned char>(0)[11], 2);

  ASSERT_FLOAT_EQ(data.data_reduced_mapping[0][0], 1.5);
  ASSERT_FLOAT_EQ(data.data_reduced_mapping[0][1], 6.0);
  ASSERT_FLOAT_EQ(data.data_reduced_mapping[0][2],
                  std::numeric_limits<float>::infinity());
}

}  // namespace arboretum_test