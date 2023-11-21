#pragma once
#include <cassert>
#include <type_traits>

// make sure some parallel model is defined
#if !defined(USE_SERIAL) && !defined(USE_OMP) && !defined(USE_MPI) && !defined(USE_MPI_OMP) && !defined(USE_KOKKOS) && !defined(USE_CUDA) && !defined(USE_HIP)
#error "No parallel model not defined"
#endif

// include the necessary libraries for the parallel model
#if defined(USE_OMP) || defined(USE_MPI_OMP)
#include <omp.h>
#elif defined(USE_MPI) || defined(USE_MPI_OMP)
#include <mpi.h>
#elif defined(USE_KOKKOS)
#include <Kokkos_Core.hpp>
#elif defined(USE_CUDA)
#include <cuda_runtime.h>
#endif

// some helper macros to unify CUDA and HIP interfaces
#if defined(USE_CUDA)
#define GRID_STRIDE_LOOP(i, n) for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)
#define ALLOC(ptr, size) cudaMalloc(&(ptr), (size))
#define COPY_H2D(dst, src, size) cudaMemcpy((dst), (src), (size), cudaMemcpyHostToDevice)
#define COPY_D2H(dst, src, size) cudaMemcpy((dst), (src), (size), cudaMemcpyDeviceToHost)
#define FREE(ptr) cudaFree((ptr))
#define SYNC() cudaDeviceSynchronize()
#elif defined(USE_HIP)
#define GRID_STRIDE_LOOP(i, n) for (int i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x; i < (n); i += hipBlockDim_x * hipGridDim_x)
#define ALLOC(ptr, size) hipMalloc(&(ptr), (size))
#define COPY_H2D(dst, src, size) hipMemcpy((dst), (src), (size), hipMemcpyHostToDevice)
#define COPY_D2H(dst, src, size) hipMemcpy((dst), (src), (size), hipMemcpyDeviceToHost)
#define FREE(ptr) hipFree((ptr))
#define SYNC() hipDeviceSynchronize()
#endif

// Kokkos utilities
#if defined(USE_KOKKOS)
template <typename DType>
void copyVectorToView(std::vector<DType> const& vec, Kokkos::View<DType*> view) {
    assert(vec.size() == view.size());
    for (int i = 0; i < vec.size(); i += 1) {
        view(i) = vec[i];
    }
}

template <typename DType>
void copyViewToVector(Kokkos::View<DType*> view, std::vector<DType>& vec) {
    assert(vec.size() == view.size());
    for (int i = 0; i < vec.size(); i += 1) {
        vec[i] = view(i);
    }
}

template <typename DType>
void fillRandKokkos(Kokkos::View<DType*> &x, DType min, DType max) {
    for (int i = 0; i < x.size(); i += 1) {
        DType val;
        if constexpr (std::is_floating_point_v<DType>) {
            val = (rand() / (double) RAND_MAX) * (max - min) + min;
        } else if constexpr (std::is_integral_v<DType>) {
            val = rand() % (max - min) + min;
        }
        x(i) = val;
    }
}
#endif


// MPI utilities
#if defined(USE_MPI) || defined(USE_MPI_OMP)
#define IS_ROOT(rank) ((rank) == 0)
#define BCAST(vec,dtype) MPI_Bcast((vec).data(), (vec).size(), MPI_##dtype, 0, MPI_COMM_WORLD)
#define SYNC() MPI_Barrier(MPI_COMM_WORLD)
#else
#define IS_ROOT(rank) true
#define BCAST(vec,dtype)
#define SYNC()
#endif



// utility functions
template <typename T, typename DType>
void fillRand(T &x, DType min, DType max) {
    for (int i = 0; i < x.size(); i += 1) {
        DType val;
        if constexpr (std::is_floating_point_v<DType>) {
            val = (rand() / (double) RAND_MAX) * (max - min) + min;
        } else if constexpr (std::is_integral_v<DType>) {
            val = rand() % (max - min) + min;
        }
        x[i] = val;
    }
}

// compare two vectors of floating point numbers
template <typename Vec, typename FType>
bool fequal(Vec const& a, Vec const& b, FType epsilon = 1e-6) {
    assert(a.size() == b.size());
    for (int i = 0; i < a.size(); i += 1) {
        if (std::abs(a[i] - b[i]) > epsilon) {
            return false;
        }
    }
    return true;
}