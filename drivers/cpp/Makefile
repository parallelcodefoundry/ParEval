CXX = g++
MPICXX = mpicxx
CXX_FLAGS = -std=c++17 -O3
Kokkos_PATH ?= ../../tpl/kokkos/build

HAS_NVCC := $(shell command -v nvcc 2> /dev/null)
HAS_HIPCC := $(shell command -v hipcc 2> /dev/null)
HAS_KOKKOS := $(shell test -d $(Kokkos_PATH)/lib64/ && echo "true")

SERIAL_DRIVERS = $(patsubst %.cc,%.o, $(wildcard */*serial-driver.cc))
OMP_DRIVERS = $(patsubst %.cc,%.o, $(wildcard */*omp-driver.cc))
MPI_DRIVERS = $(patsubst %.cc,%.o, $(wildcard */*mpi-driver.cc))
MPI_OMP_DRIVERS = $(patsubst %.cc,%.o, $(wildcard */*mpi-omp-driver.cc))
CUDA_DRIVERS = $(patsubst %.cu,%.o, $(wildcard */*cuda-driver.cu))
HIP_DRIVERS = $(patsubst %.cu,%.o, $(wildcard */*hip-driver.cu))
KOKKOS_DRIVERS = $(patsubst %.cc,%.o, $(wildcard */*kokkos-driver.cc))

ALL_DRIVERS = $(SERIAL_DRIVERS) $(OMP_DRIVERS) $(MPI_DRIVERS) $(MPI_OMP_DRIVERS) $(CUDA_DRIVERS) $(HIP_DRIVERS) $(KOKKOS_DRIVERS)

all: $(ALL_DRIVERS)

%.o: %.cc
	$(CXX) $(CXX_FLAGS) -o $@ -c $<

%serial-driver.o: %serial-driver.cc
	$(CXX) $(CXX_FLAGS) -o $@ -c $<

%omp-driver.o: %omp-driver.cc
	$(CXX) $(CXX_FLAGS) -fopenmp -o $@ -c $<

%mpi-driver.o: %mpi-driver.cc
	$(MPICXX) $(CXX_FLAGS) -o $@ -c $<

%mpi-omp-driver.o: %mpi-omp-driver.cc
	$(MPICXX) $(CXX_FLAGS) -fopenmp -o $@ -c $<

%cuda-driver.o: %cuda-driver.cu
ifdef HAS_NVCC
	@echo "nvcc found, compiling $(HAS_NVCC)"
	nvcc -std=c++17 -O3 -o $@ -c $<
else
	@echo "nvcc not found, skipping $@"
endif

%hip-driver.o: %hip-driver.cu
ifdef HAS_HIPCC
	hipcc -std=c++17 -O3 -o $@ -c $< -Wno-unused-result
else
	@echo "hipcc not found, skipping $@"
endif

%kokkos-driver.o: %kokkos-driver.cc
ifdef HAS_KOKKOS
	$(CXX) $(CXX_FLAGS) -I$(Kokkos_PATH)/include -L$(Kokkos_PATH)/lib64 -fopenmp -o $@ -c $<
else
	@echo "Kokkos not found, skipping $@"
endif

clean:
	rm -f $(ALL_DRIVERS)
