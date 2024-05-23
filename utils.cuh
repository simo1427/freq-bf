//
// Created by HP on 23/05/2024.
//

#ifndef CUDA_BF_UTILS_CUH
#define CUDA_BF_UTILS_CUH

#include <device_launch_parameters.h>
#include <iostream>

void errorCheck(cudaError_t err);

#define checkForErrors(val) errorCheck(val)

#endif //CUDA_BF_UTILS_CUH
