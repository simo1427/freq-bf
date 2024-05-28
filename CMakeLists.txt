cmake_minimum_required(VERSION 3.25)
project(cuda_bf CUDA CXX)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD 20)

add_executable(cuda_bf main.cu utils.cu utils.cuh filtering.cu filtering.cuh rangeKernels.h rangeKernels.cpp)

set_target_properties(cuda_bf PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

find_package(OpenCV REQUIRED )
find_package(GSL REQUIRED )
find_package(CLI11 CONFIG REQUIRED)

target_link_libraries(cuda_bf PRIVATE ${OpenCV_LIBS})
target_link_libraries(cuda_bf PRIVATE GSL::gsl GSL::gslcblas)
target_link_libraries(cuda_bf PRIVATE CLI11::CLI11)