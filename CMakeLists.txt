CMAKE_MINIMUM_REQUIRED(VERSION 3.9 FATAL_ERROR)

PROJECT(arboretum LANGUAGES CXX CUDA)

FIND_PACKAGE(OpenMP)
FIND_PACKAGE(CUDA 8.0 REQUIRED)

function(format_gencode_flags flags out)
# Generate SASS
 foreach(ver ${flags})
   set(${out} "${${out}}-gencode=arch=compute_${ver},code=sm_${ver} ")
 endforeach()
# Generate PTX for last architecture
list(GET flags -1 ver)
set(${out} "${${out}}-gencode=arch=compute_${ver},code=compute_${ver} ")

set(${out} "${${out}}" PARENT_SCOPE)
endfunction(format_gencode_flags flags)

# Compiler flags
SET(CMAKE_CXX_STANDARD 11)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)
SET(CMAKE_POSITION_INDEPENDENT_CODE ON)
SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -w")

if(NOT CMAKE_BUILD_TYPE AND NOT DEV_BUILD)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CUDA_FLAGS_DEBUG " -lineinfo -Xcompiler=-g")
set(CMAKE_CUDA_FLAGS_RELEASE " -Xcompiler=-O3")

if (OPENMP_FOUND)
    SET (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    SET (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    SET (CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=${OpenMP_CXX_FLAGS}")
endif()

if(DEV_BUILD)
    MESSAGE(STATUS "Building DEVELOPER compute capability version.")
    SET(GPU_COMPUTE_VER 61)
    SET(CMAKE_BUILD_TYPE Debug)
else()
    MESSAGE(STATUS "Building RELEASE compute capability version.")
    SET(GPU_COMPUTE_VER 35;50;52;60;61)
endif()

if(((CUDA_VERSION_MAJOR EQUAL 9) OR (CUDA_VERSION_MAJOR GREATER 9)) AND NOT DEV_BUILD)
    MESSAGE(STATUS "CUDA GREATER OR EQUAL THAN 9.0 detected, adding Volta compute capability (7.0).")
    SET(GPU_COMPUTE_VER "${GPU_COMPUTE_VER};70")
endif()

if(((CUDA_VERSION_MAJOR EQUAL 10) OR (CUDA_VERSION_MAJOR GREATER 10)) AND NOT DEV_BUILD)
    MESSAGE(STATUS "CUDA GREATER OR EQUAL THAN 10.0 detected, adding Turing compute capability (7.5).")
    SET(GPU_COMPUTE_VER "${GPU_COMPUTE_VER};75")
endif()

SET(GENCODE_FLAGS "")
FORMAT_GENCODE_FLAGS("${GPU_COMPUTE_VER}" GENCODE_FLAGS)
MESSAGE(STATUS "cuda architecture flags: ${GENCODE_FLAGS}")

SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${GENCODE_FLAGS} -std=c++11 --expt-extended-lambda --default-stream per-thread")

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


# template instantiation
FILE(GLOB_RECURSE files src/*.cu)
add_custom_target(generate
    COMMAND python -m cogapp -r -c ${files}
    SOURCES ${files}
    COMMENT "Generates template instantiation"
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
INCLUDE_DIRECTORIES(googletest/googletest/include googletest/googletest)

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


