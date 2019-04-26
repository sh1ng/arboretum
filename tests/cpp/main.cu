#include "gtest/gtest.h"

namespace arboretum_test {

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
} // namespace arboretum_test