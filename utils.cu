//
// Created by HP on 23/05/2024.
//

#include "utils.cuh"

void errorCheck(cudaError_t err)
{
    if (err != cudaSuccess) {
        std::cerr << "CUDA error:" << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}