export CXX  = g++-5
export LDFLAGS= -pthread -lm
export CFLAGS = -Wall -O1 -g -mtune=native -march=native  -Wno-unknown-pragmas -ftree-vectorize -std=c++11 -fPIC
SLIB = python-wrapper/arboretum_wrapper.so
OBJ = io.o param.o garden.o

all: $(OBJ) $(SLIB)

param.o: src/core/param.cpp src/core/param.h

garden.o: src/core/garden.cpp src/core/garden.h param.o io.o

io.o: src/io/io.cpp src/io/io.h src/core/objective.h

python-wrapper/arboretum_wrapper.so: python-wrapper/arboretum_wrapper.cpp python-wrapper/arboretum_wrapper.h io.o garden.o param.o

$(OBJ) :
	$(CXX) -c $(CFLAGS) -o $@ $(firstword $(filter %.cpp %.c %.cc, $^) )

$(SLIB) :
	$(CXX) $(CFLAGS) -fPIC -shared -o $@ $(filter %.cpp %.o %.c %.a %.cc, $^) $(LDFLAGS)


clean:
	$(RM) -rf $(OBJ) $(SLIB) *.o  */*.o */*/*.o *~ */*~ */*/*~
