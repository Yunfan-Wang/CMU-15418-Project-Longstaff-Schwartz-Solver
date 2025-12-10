# Makefile for Longstaff Schwartz GPU project

EXECUTABLE := lsm_gpu

NVCC      := nvcc
NVCCFLAGS := -O3 -m64 --gpu-architecture=compute_61 -ccbin /usr/bin/g++-11
LDFLAGS   := -L/usr/local/cuda-11.7/lib64/ -lcudart
CXXFLAGS  := -O3 -m64 -std=c++17

OBJS := main.o lsm.o

all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(LDFLAGS)

main.o: main.cpp lsm.hpp
	$(NVCC) $(NVCCFLAGS) -std=c++17 -c $< -o $@

lsm.o: lsm.cu lsm.hpp
	$(NVCC) $(NVCCFLAGS) -std=c++17 -c $< -o $@

clean:
	rm -f $(EXECUTABLE) *.o