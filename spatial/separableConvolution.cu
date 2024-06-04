#include "separableConvolution.cuh"
#include "utils.cuh"

#include "cuda_math.h"
#include <cuda.h>
#include <iostream>
#include <math_constants.h>
#include <stdio.h> // for within CUDA kernels for debugging purposes

#define IMAGE_PATCH_COLS_WIDTH 32
#define IMAGE_PATCH_COLS_HEIGHT 32
#define IMAGE_PATCH_ROWS_WIDTH 512
#define IMAGE_PATCH_ROWS_HEIGHT 2

#define MAX_KS 64
#define MAX_HKS 32

__constant__ float d_Krn[MAX_KS];

void setConvolutionKernel(float* h_Krn, int krnSize)
{
    cudaMemcpyToSymbol(d_Krn, h_Krn, krnSize * sizeof(float));
}

__device__ int toAddress(int x, int y, int width)
{
    return y * width + x;
}

__global__ void sepFilterHorizontalF4(float4* d_Out, float4* d_Src, int width, int height, int krnSize, size_t pitch)
{
    const unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    int hks = krnSize / 2;
    float4 sum = make_float4(0, 0, 0, 0);
    for (int krnI = -hks; krnI <= hks; krnI++) {
        int neighbourX = x + krnI;
        if (neighbourX < 0)
            neighbourX = -neighbourX;
        else if (neighbourX >= width)
            neighbourX = 2 * width - neighbourX - 2;

        const int neighbourAddress = toAddress(neighbourX, y, pitch);
        const float spatialCoef = d_Krn[hks + krnI];
        sum = sum + make_float4(spatialCoef) * d_Src[neighbourAddress];
    }

    const int address = toAddress(x, y, pitch);
    d_Out[address] = sum;
}

__global__ void sepFilterVerticalF4(float4* d_Out, float4* d_Src, int width, int height, int krnSize, size_t pitch)
{
    const unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    int hks = krnSize / 2;
    float4 sum = make_float4(0, 0, 0, 0);
    for (int krnI = -hks; krnI <= hks; krnI++) {
        int neighbourY = y + krnI;
        if (neighbourY < 0)
            neighbourY = -neighbourY;
        else if (neighbourY >= height)
            neighbourY = 2 * height - neighbourY - 2;

        const int neighbourAddress = toAddress(x, neighbourY, pitch);
        const float spatialCoef = d_Krn[hks + krnI];
        sum = sum + make_float4(spatialCoef) * d_Src[neighbourAddress];
    }

    const int address = toAddress(x, y, pitch);
    d_Out[address] = sum;
}

void sepFilterf4(float4* d_Out, float4* d_Src, float4* d_Buf, int width, int height, int krnSize, size_t pitchInBytes)
{
    // TODO: take stream as a param?
    const dim3 verticalWorkGroupSize { 32, 16 };
    const dim3 horizontalWorkGroupSize { 128, 1 };
    const dim3 verticalWorkGroups = computeNumWorkGroups(verticalWorkGroupSize, width, height);
    const dim3 horizontalWorkGroups = computeNumWorkGroups(horizontalWorkGroupSize, width, height);
    const float pitch = pitchInBytes / sizeof(float4);

    sepFilterVerticalF4<<<verticalWorkGroups, verticalWorkGroupSize>>>(d_Buf, d_Src, width, height, krnSize, pitch);
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) { // something's gone wrong
        // print out the CUDA error as a string
        printf("CUDA Error: %s\n", cudaGetErrorString(error));
    }
    sepFilterHorizontalF4<<<horizontalWorkGroups, horizontalWorkGroupSize>>>(d_Out, d_Buf, width, height, krnSize, pitch);
    cudaDeviceSynchronize();
    cudaGetLastError();
    if (error != cudaSuccess) {
        // something's gone wrong
        // print out the CUDA error as a string
        printf("CUDA Error: %s\n", cudaGetErrorString(error));
    }
}

__global__ void sepFilterRows(float* d_Out, float* d_Src, int width, int height, int krnSize, size_t pitch)
{
    unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int j = threadIdx.y + blockIdx.y * blockDim.y;

    if (i >= height || j >= width)
        return;

    // TODO: introduce shared memory

    int hks = krnSize / 2;

    float sum = 0;

    float* d_SrcRow = (float*)((char*)d_Src + i * pitch);
    for (int krnI = -hks; krnI <= hks; krnI++) {
        int tmpJ = j + krnI;

        if (tmpJ < 0)
            tmpJ = -tmpJ;
        else if (tmpJ >= width)
            tmpJ = 2 * width - tmpJ - 2;

        sum += d_Krn[hks + krnI] * d_SrcRow[tmpJ]; // TODO: introduce shared memory
    }

    float* d_OutPtr = (float*)((char*)d_Out + i * pitch);
    d_OutPtr[j] = sum;
}

__global__ void sepFilterCols(float* d_Out, float* d_Src, int width, int height, int krnSize, size_t pitch)
{
    unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int j = threadIdx.y + blockIdx.y * blockDim.y;

    if (i >= height || j >= width)
        return;

    // TODO: introduce shared memory

    int hks = krnSize / 2;

    float sum = 0;

    for (int krnI = -hks; krnI <= hks; krnI++) {
        int tmpI = i + krnI;
        if (tmpI < 0)
            tmpI = -tmpI;
        else if (tmpI >= height)
            tmpI = 2 * height - tmpI - 2;

        float* d_SrcRow = (float*)((char*)d_Src + tmpI * pitch);

        sum += d_Krn[hks + krnI] * d_SrcRow[j]; // TODO: introduce shared memory
        // also, this probably will lead to maaaany uncoalesced accesses, rethink how to handle this part
    }

    float* d_OutPtr = (float*)((char*)d_Out + i * pitch);
    d_OutPtr[j] = sum;
}

void sepFilter(float* d_Out, float* d_Src, float* d_Buf, int width, int height, int krnSize, size_t pitch)
{
    //    dim3 threads(1, 1);
    //    dim3 blocks(1, 1);
    dim3 threadsRows(IMAGE_PATCH_ROWS_HEIGHT, IMAGE_PATCH_ROWS_WIDTH);
    dim3 blocksRows(height / threadsRows.x + (height % threadsRows.x ? 1 : 0), width / threadsRows.y + (width % threadsRows.y ? 1 : 0));

    dim3 threadsCols(IMAGE_PATCH_COLS_HEIGHT, IMAGE_PATCH_COLS_WIDTH);
    dim3 blocksCols(height / threadsCols.x + (height % threadsCols.x ? 1 : 0), width / threadsCols.y + (width % threadsCols.y ? 1 : 0));

    // Create events for measuring the performance of each of the kernels
    cudaEvent_t start, buf1, buf2, finish;
    float timeRows, timeCols;

    cudaEventCreate(&start);
    cudaEventCreate(&buf1);
    cudaEventCreate(&buf2);
    cudaEventCreate(&finish);

    // Run the kernels
    cudaEventRecord(start, 0);
    sepFilterRows<<<blocksRows, threadsRows>>>(d_Buf, d_Src, width, height, krnSize, pitch);
    cudaEventRecord(buf1, 0);
    cudaEventSynchronize(buf1);

    /////
    cudaEventRecord(buf2, 0);
    sepFilterCols<<<blocksCols, threadsCols>>>(d_Out, d_Buf, width, height, krnSize, pitch);
    cudaEventRecord(finish, 0);
    cudaEventSynchronize(finish);

    // Check for errors
    //    cudaError_t error = cudaGetLastError();
    //    if (error != cudaSuccess)
    //    {
    //        // something's gone wrong
    //        // print out the CUDA error as a string
    //        printf("CUDA Error: %s\n", cudaGetErrorString(error));
    //        return;
    //    }

    // Calculate elapsed time

    cudaEventElapsedTime(&timeRows, start, buf1);
    cudaEventElapsedTime(&timeCols, buf2, finish);

    cudaEventDestroy(start);
    cudaEventDestroy(buf1);
    cudaEventDestroy(buf2);
    cudaEventDestroy(finish);

    std::cout << "frame times: rows: " << timeRows << "ms, cols: " << timeCols << "ms\n";
    std::cout << "fps: rows: " << 1000.0 / timeRows << "fps, cols: " << 1000.0 / timeCols << "fps\n";

    // TODO: make use of local memory & cooperative groups
    // TODO: copy convolution kernel to constant memory
}
