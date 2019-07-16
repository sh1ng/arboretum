#ifndef TESTS_CPP_TEST_UTILS_H
#define TESTS_CPP_TEST_UTILS_H

#include "gtest/gtest.h"

#define TEST_OK(cmd)                                           \
  do {                                                         \
    cudaError_t e = cmd;                                       \
    ASSERT_EQ(e, cudaSuccess)                                  \
      << "Cuda failure " << __FILE__ << ":" << __LINE__ << "'" \
      << cudaGetErrorString(e) << "'\n";                       \
  } while (0)

#endif
