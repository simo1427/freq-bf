#include "filtering.cuh"

#include <math_constants.h>
#include <stdio.h>
#define IMAGE_PATCH_WIDTH 64
#define IMAGE_PATCH_HEIGHT 16
#define MAX_HKS 63

__global__ void sepFilterRows(float* d_Out, float* d_Src, float* d_Krn, int width, int height, int krnSize) {
    unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int j = threadIdx.y + blockIdx.y * blockDim.y;

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

__global__ void sepFilterCols(float* d_Out, float* d_Src, float* d_Krn, int width, int height, int krnSize) {
    unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int j = threadIdx.y + blockIdx.y * blockDim.y;

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

void sepFilter(float* d_Out, float* d_Src, float* d_Buf, float* d_Krn, int width, int height, int krnSize)
{
//    dim3 threads(1, 1);
//    dim3 blocks(1, 1);
    dim3 threadsRows(IMAGE_PATCH_HEIGHT, IMAGE_PATCH_WIDTH);
    dim3 blocksRows(height / threadsRows.x + (height % threadsRows.x ? 1 : 0), width / threadsRows.y + (width % threadsRows.y ? 1 : 0));

    dim3 threadsCols(IMAGE_PATCH_WIDTH, IMAGE_PATCH_HEIGHT);
    dim3 blocksCols(height / threadsCols.x + (height % threadsCols.x ? 1 : 0), width / threadsCols.y + (width % threadsCols.y ? 1 : 0));

    sepFilterRows <<<blocksRows, threadsRows>>> (d_Buf, d_Src, d_Krn, width, height, krnSize);
    sepFilterCols <<<blocksCols, threadsCols>>> (d_Out, d_Buf, d_Krn, width, height, krnSize);
    
    // TODO: make use of local memory & cooperative groups
    // TODO: copy convolution kernel to constant memory
}
