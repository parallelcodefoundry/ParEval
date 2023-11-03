#pragma once
#include <cassert>
#include <type_traits>


#if !defined(USE_SERIAL) && !defined(USE_OMP) && !defined(USE_MPI) && !defined(USE_MPI_OMP) && !defined(USE_KOKKOS) && !defined(USE_CUDA) && !defined(USE_HIP)
#error "No parallel model not defined"
#endif

#if defined(USE_OMP) || defined(USE_MPI_OMP)
#include <omp.h>
#elif defined(USE_MPI) || defined(USE_MPI_OMP)
#include <mpi.h>
#elif defined(USE_CUDA)
#include <cuda_runtime.h>
#endif

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