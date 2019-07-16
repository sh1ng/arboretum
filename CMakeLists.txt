CMAKE_MINIMUM_REQUIRED(VERSION 3.9 FATAL_ERROR)

PROJECT(arboretum LANGUAGES CXX CUDA)

FIND_PACKAGE(OpenMP)

# Compiler flags
SET(CMAKE_CXX_STANDARD 11)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)
SET(CMAKE_POSITION_INDEPENDENT_CODE ON)
SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -w")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

if (OPENMP_FOUND)
    SET (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    SET (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    SET (CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=${OpenMP_CXX_FLAGS}")
endif()

# SET(CUDA_64_BIT_DEVICE_CODE ON CACHE STRING "Compile device code in 64 bit mode" FORCE)
SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_61 -std=c++11 -Xcompiler=-fPIC --expt-extended-lambda --default-stream per-thread")

FILE(GLOB_RECURSE GPU_SOURCES
    src/*.cpp
    src/*.h
    src/*.cu
    src/*.cuh
    )

FILE(GLOB_RECURSE WRAPPER_SOURCES
    python-wrapper/*.cu
    python-wrapper/*.cpp
    python-wrapper/*.h
    python-wrapper/*.cuh
    )

INCLUDE_DIRECTORIES(src)
INCLUDE_DIRECTORIES(_cub/)
INCLUDE_DIRECTORIES(_json/single_include/nlohmann)
ADD_LIBRARY(arboretum_static STATIC ${GPU_SOURCES})
SET_TARGET_PROPERTIES(arboretum_static PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
ADD_LIBRARY(arboretum SHARED ${WRAPPER_SOURCES})
target_link_libraries(arboretum arboretum_static)

#============= GTest
ADD_SUBDIRECTORY (googletest)
enable_testing()
INCLUDE(GoogleTest)
include_directories(googletest/googletest/include googletest/googletest)

#============= Unit Tests
FILE(GLOB_RECURSE TEST_SOURCES
tests/cpp/*.cu
tests/cpp/*.cuh
tests/cpp/*.h
tests/cpp/*.cpp
)

ADD_EXECUTABLE(arboretum_test ${TEST_SOURCES})
SET_TARGET_PROPERTIES(arboretum_test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(arboretum_test arboretum_static gtest gtest_main)


