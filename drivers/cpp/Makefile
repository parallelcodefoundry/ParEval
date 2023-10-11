CXX = g++
CXX_FLAGS = -std=c++17 -O3

BENCHMARK_DRIVERS = $(patsubst %.cc,%.o, $(wildcard benchmarks/*-driver.cc))
MODEL_DRIVERS = $(patsubst %.cc,%.o, $(wildcard models/*-driver.cc))

all: $(BENCHMARK_DRIVERS) $(MODEL_DRIVERS)

%.o: %.cc
	$(CXX) $(CXX_FLAGS) -o $@ -c $<

models/omp-driver.o: models/omp-driver.cc
	$(CXX) $(CXX_FLAGS) -fopenmp -o $@ -c $<


clean:
	rm -f $(BENCHMARK_DRIVERS) $(MODEL_DRIVERS)
