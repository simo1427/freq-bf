cmake_minimum_required(VERSION 3.25)
project(cuda_bf CUDA CXX)

add_executable(cuda_bf
	"main.cu"
	"utils/cuda_utils.cu"
	"utils/cuda_utils.cuh"
	"spatial/separableConvolution.cu"
	"spatial/separableConvolution.cuh"
	"rangeKernels.h"
	"rangeKernels.cpp"
	"refBilateral.h"
	"refBilateral.cpp"
	"fastBilateral.cu"
	"fastBilateral.cuh"
	 "utils/cuda_math.h"
		evaluators/psnrRunner.cpp
		evaluators/psnrRunner.h
		evaluators/perfRunner.cpp
		evaluators/perfRunner.h)
target_include_directories(cuda_bf PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_compile_features(cuda_bf PUBLIC cxx_std_20 cuda_std_20)
set_target_properties(cuda_bf PROPERTIES
	CUDA_SEPARABLE_COMPILATION ON
	CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_compile_options(cuda_bf PUBLIC "$<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G>")
target_compile_options(cuda_bf PUBLIC "$<$<AND:$<CONFIG:RelWithDebInfo>,$<COMPILE_LANGUAGE:CUDA>>:-lineinfo>")

find_package(OpenCV REQUIRED)
find_package(GSL REQUIRED)
find_package(CLI11 CONFIG REQUIRED)
target_link_libraries(cuda_bf PUBLIC ${OpenCV_LIBS} GSL::gsl GSL::gslcblas CLI11::CLI11)