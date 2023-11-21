CXX = g++
MPICXX = mpicxx
CXX_FLAGS = -std=c++17 -O3

OMP_DRIVERS = $(patsubst %.cc,%.o, $(wildcard */*omp-driver.cc))
MPI_DRIVERS = $(patsubst %.cc,%.o, $(wildcard */*mpi-driver.cc))

all: $(OMP_DRIVERS) $(MPI_DRIVERS)

%.o: %.cc
	$(CXX) $(CXX_FLAGS) -o $@ -c $<

%omp-driver.o: %omp-driver.cc
	$(CXX) $(CXX_FLAGS) -fopenmp -o $@ -c $<

%mpi-driver.o: %mpi-driver.cc
	$(MPICXX) $(CXX_FLAGS) -o $@ -c $<

clean:
	rm -f $(OMP_DRIVERS) $(MPI_DRIVERS)
