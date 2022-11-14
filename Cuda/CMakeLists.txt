cmake_minimum_required(VERSION 3.18.2)

project(CudaOp)


set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


# Python
find_package(Python3 COMPONENTS Interpreter Development)
message("====Pybind11====")
add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/pybind11 EXCLUDE_FROM_ALL)
message("================")

# OpenMP
find_package(OpenMP)
if(OPENMP_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# Cuda
find_package(CUDAToolkit)
if(CMAKE_CXX_COMPILE_VERSION VERSION_GREATER 5.0)
    add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
endif()

set(CUDA_ARCHS "75" CACHE STRING "Cuda Archs")
foreach(arch ${CUDA_ARCHS})
    list(APPEND CUDA_FLAG "-gencode arch=compute_${arch},code=sm_${arch} ")
endforeach()
message("Set CUDA FLAGS: " ${CUDA_FLAG})
set(CMAKE_CUDA_FLAGS ${CUDA_FLAG})

# Library
set(CUDART_LIBRARY CUDA::cudart)
set(CUBLAS_LIBRARY CUDA::cublas CUDA::cublasLt)
set(OMP_LIBRARY OpenMP::OpenMP_CXX)

# Subdirectory
add_subdirectory(Transpose)