#include "separableConvolution.cuh"

#include <math_constants.h>
#include <stdio.h> // for within CUDA kernels for debugging purposes
#include <iostream>

#define IMAGE_PATCH_COLS_WIDTH 64
#define IMAGE_PATCH_COLS_HEIGHT 16
#define IMAGE_PATCH_ROWS_WIDTH 64
#define IMAGE_PATCH_ROWS_HEIGHT 16

#define MAX_KS 64
#define MAX_HKS (MAX_KS / 2)

__constant__ float d_Krn[MAX_KS];

void setConvolutionKernel(float* h_Krn, int krnSize)
{
    cudaMemcpyToSymbol(d_Krn, h_Krn, krnSize * sizeof(float));
}

__global__ void sepFilterRowsf4(float4* d_Out, float4* d_Src, int width, int height, int krnSize) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if (i >= height || j >= width)
        return;

    // TODO: introduce shared memory

    int hks = krnSize / 2;

    float4 sum = (float4)(0, 0, 0, 0);


    for (int krnI = -hks; krnI <= hks; krnI++)
    {
        int tmpJ = j + krnI;

        if (tmpJ < 0)
            tmpJ = -tmpJ;
        else if (tmpJ >= width)
            tmpJ = 2 * width - tmpJ - 2;

//        printf("%d %d;;", j, tmpJ);
        //TODO: for debug purpose, remove after debugged!

        int addr = i * width + tmpJ;
        float spatialCoef = d_Krn[hks + krnI];
        sum.w += spatialCoef * d_Src[addr].w;
        sum.x += spatialCoef * d_Src[addr].x;
        sum.y += spatialCoef * d_Src[addr].y;
        sum.z += spatialCoef * d_Src[addr].z; // TODO: introduce shared memory

//        printf("(%d) %f * %f  -> %f\n", d_Src[addr], curPx, d_Krn[hks + krnI], sum);
    }


    d_Out[i * width + j] = sum;


}

__global__ void sepFilterColsf4(float4* d_Out, float4* d_Src, int width, int height, int krnSize) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if (i >= height || j >= width)
        return;

    // TODO: introduce shared memory

    int hks = krnSize / 2;

    float4 sum = make_float4(0, 0, 0, 0);


    for (int krnI = -hks; krnI <= hks; krnI++)
    {
        int tmpI = i + krnI;
        if (tmpI < 0)
            tmpI = -tmpI;
        else if (tmpI >= height)
            tmpI = 2 * height - tmpI - 2;

//        printf("%d %d;;", j, tmpJ);
        //TODO: for debug purpose, remove after debugged!

        int addr = tmpI * width + j;
        float spatialCoef = d_Krn[hks + krnI];
        sum.w += spatialCoef * d_Src[addr].w;
        sum.x += spatialCoef * d_Src[addr].x;
        sum.y += spatialCoef * d_Src[addr].y;
        sum.z += spatialCoef * d_Src[addr].z; // TODO: introduce shared memory

//        printf("(%d) %f * %f  -> %f\n", d_Src[addr], curPx, d_Krn[hks + krnI], sum);
    }


    d_Out[i * width + j] = sum;

}

__global__ void sepFilterRows(float* d_Out, float* d_Src, int width, int height, int krnSize) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if (i >= height || j >= width)
        return;

    // TODO: introduce shared memory

    int hks = krnSize / 2;

    float sum = 0;


    for (int krnI = -hks; krnI <= hks; krnI++)
    {
        int tmpJ = j + krnI;

        if (tmpJ < 0)
            tmpJ = -tmpJ;
        else if (tmpJ >= width)
            tmpJ = 2 * width - tmpJ - 2;

//        printf("%d %d;;", j, tmpJ);
        //TODO: for debug purpose, remove after debugged!

        int addr = i * width + tmpJ;
        sum += d_Krn[hks + krnI] * d_Src[addr]; // TODO: introduce shared memory

//        printf("(%d) %f * %f  -> %f\n", d_Src[addr], curPx, d_Krn[hks + krnI], sum);
    }


    d_Out[i * width + j] = sum;


}

__global__ void sepFilterCols(float* d_Out, float* d_Src, int width, int height, int krnSize) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if (i >= height || j >= width)
        return;

    // TODO: introduce shared memory

    int hks = krnSize / 2;

    float sum = 0;


    for (int krnI = -hks; krnI <= hks; krnI++)
    {
        int tmpI = i + krnI;
        if (tmpI < 0)
            tmpI = -tmpI;
        else if (tmpI >= height)
            tmpI = 2 * height - tmpI - 2;

//        printf("%d %d;;", j, tmpJ);
        //TODO: for debug purpose, remove after debugged!

        int addr = tmpI * width + j;
        sum += d_Krn[hks + krnI] * d_Src[addr]; // TODO: introduce shared memory

//        printf("(%d) %f * %f  -> %f\n", d_Src[addr], curPx, d_Krn[hks + krnI], sum);
    }


    d_Out[i * width + j] = sum;

}

void sepFilter(float* d_Out, float* d_Src, float* d_Buf, int width, int height, int krnSize)
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
    sepFilterRows <<<blocksRows, threadsRows>>> (d_Buf, d_Src, width, height, krnSize);
    cudaEventRecord(buf1, 0);
    cudaEventSynchronize( buf1 );

    /////
    cudaEventRecord(buf2, 0);
    sepFilterCols <<<blocksCols, threadsCols>>> (d_Out, d_Buf, width, height, krnSize);
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
    std::cout << "fps: rows: " << 1000.0/timeRows << "fps, cols: " << 1000.0/timeCols << "fps\n";


    // TODO: make use of local memory & cooperative groups
    // TODO: copy convolution kernel to constant memory
}

void sepFilterf4(float4* d_Out, float4* d_Src, float4* d_Buf, int width, int height, int krnSize)
{
    // TODO: take stream as a param?


    dim3 threadsRows(IMAGE_PATCH_ROWS_HEIGHT, IMAGE_PATCH_ROWS_WIDTH);
    dim3 blocksRows(height / threadsRows.x + (height % threadsRows.x ? 1 : 0), width / threadsRows.y + (width % threadsRows.y ? 1 : 0));

    dim3 threadsCols(IMAGE_PATCH_COLS_HEIGHT, IMAGE_PATCH_COLS_WIDTH);
    dim3 blocksCols(height / threadsCols.x + (height % threadsCols.x ? 1 : 0), width / threadsCols.y + (width % threadsCols.y ? 1 : 0));

    sepFilterRowsf4 <<<blocksRows, threadsRows>>> (d_Buf, d_Src, width, height, krnSize);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        // something's gone wrong
        // print out the CUDA error as a string
        printf("CUDA Error: %s\n", cudaGetErrorString(error));

    }
    sepFilterColsf4 <<<blocksCols, threadsCols>>> (d_Out, d_Buf, width, height, krnSize);

    cudaGetLastError();
    if (error != cudaSuccess)
    {
        // something's gone wrong
        // print out the CUDA error as a string
        printf("CUDA Error: %s\n", cudaGetErrorString(error));

    }
}
