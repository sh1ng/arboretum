# CUDA compute capability
export sm = 610
# compiler by default
export CXX = g++
OS := $(shell uname)

# MAC OS
ifeq ($(OS), Darwin)
export CXX = clang-omp++
endif

include _cub/common.mk

export CC = $(NVCC) $(DEFINES) $(SM_TARGETS)
export LDFLAGS= -lm
export CFLAGS = $(NVCCFLAGS) -O3 -I_cub/ -I_json/src/ -std=c++11 -Xcompiler -fPIC -Xcompiler -O3 -Xcompiler -fopenmp -Xcompiler -Wall -Xcompiler -funroll-loops -Xcompiler -march=native -ccbin=$(CXX)
SLIB = python-wrapper/arboretum_wrapper.so
OBJ = io.o param.o garden.o

all: $(OBJ) $(SLIB)

param.o: src/core/param.cpp src/core/param.h

garden.o: src/core/garden.cu src/core/garden.h param.o io.o

io.o: src/io/io.cu src/io/io.h

python-wrapper/arboretum_wrapper.so: python-wrapper/arboretum_wrapper.cpp python-wrapper/arboretum_wrapper.h io.o garden.o param.o

$(OBJ) :
	$(CC) -c $(CFLAGS) -o $@ $(firstword $(filter %.cpp %.c %.cc %.cu, $^) )

$(SLIB) :
	$(CC) $(CFLAGS) -shared -o $@ $(filter %.cpp %.o %.c %.a %.cc %.cu, $^) $(LDFLAGS)


clean:
	$(RM) -rf $(OBJ) $(SLIB) *.o  */*.o */*/*.o *~ */*~ */*/*~
