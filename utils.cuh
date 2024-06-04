//
// Created by HP on 23/05/2024.
//

#ifndef CUDA_BF_UTILS_CUH
#define CUDA_BF_UTILS_CUH

#include <device_launch_parameters.h>
#include <iostream>
#include <format>
#include <cuda_runtime.h>

// Following methods taken from helper_cuda.h (part of cuda-samples)
template <typename T>
void check(T result, char const *const func, const char *const file,
           int const line) {
    if (result) {
//        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
//                static_cast<unsigned int>(result), _cudaGetErrorEnum(result), func);

        std::cerr << "CUDA error at " << std::string(file) << ":" << line
                  << " code=" << static_cast<unsigned int>(result) << "(" << std::string(cudaGetErrorName(result)) << ")"
                  << "\"" << std::string(func) << "\" \n";

        exit(EXIT_FAILURE);
    }
}

// This will output the proper CUDA error strings in the event
// that a CUDA host call returns an error
#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

// This will output the proper error string when calling cudaGetLastError
#define getLastCudaError(msg) __getLastCudaError(msg, __FILE__, __LINE__)

inline void __getLastCudaError(const char *errorMessage, const char *file,
                               const int line) {
    cudaError_t err = cudaGetLastError();

    if (cudaSuccess != err) {
        fprintf(stderr,
                "%s(%i) : getLastCudaError() CUDA error :"
                " %s : (%d) %s.\n",
                file, line, errorMessage, static_cast<int>(err),
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// This will only print the proper error string when calling cudaGetLastError
// but not exit program incase error detected.
#define printLastCudaError(msg) __printLastCudaError(msg, __FILE__, __LINE__)

inline void __printLastCudaError(const char *errorMessage, const char *file,
                                 const int line) {
    cudaError_t err = cudaGetLastError();

    if (cudaSuccess != err) {
        fprintf(stderr,
                "%s(%i) : getLastCudaError() CUDA error :"
                " %s : (%d) %s.\n",
                file, line, errorMessage, static_cast<int>(err),
                cudaGetErrorString(err));
    }
}


dim3 computeNumWorkGroups(const dim3& workGroupSize, int width, int height);

#endif //CUDA_BF_UTILS_CUH
