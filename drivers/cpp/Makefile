CXX = g++
MPICXX = mpicxx
CXX_FLAGS = -std=c++17 -O3

SERIAL_DRIVERS = $(patsubst %.cc,%.o, $(wildcard */*serial-driver.cc))
OMP_DRIVERS = $(patsubst %.cc,%.o, $(wildcard */*omp-driver.cc))
MPI_DRIVERS = $(patsubst %.cc,%.o, $(wildcard */*mpi-driver.cc))
MPI_OMP_DRIVERS = $(patsubst %.cc,%.o, $(wildcard */*mpi-omp-driver.cc))
CUDA_DRIVERS = $(patsubst %.cu,%.o, $(wildcard */*cuda-driver.cu))

ALL_DRIVERS = $(SERIAL_DRIVERS) $(OMP_DRIVERS) $(MPI_DRIVERS) $(MPI_OMP_DRIVERS) $(CUDA_DRIVERS)

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
	nvcc -std=c++17 -O3 -o $@ -c $<

clean:
	rm -f $(ALL_DRIVERS)
