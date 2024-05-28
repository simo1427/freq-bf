#include "filtering.cuh"

#include <math_constants.h>
#include <stdio.h>
#define IMAGE_PATCH_WIDTH 64
#define IMAGE_PATCH_HEIGHT 16
#define MAX_HKS 64

__global__ void sepFilterRows(float* d_Out, uint8_t* d_Src, float* d_Krn, int width, int height, int krnSize) {
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
//        if (tmpI < 0)
//            tmpI += height;
//        else if (tmpI >= height)
//            tmpI -= height;
        if (tmpJ < 0)
            tmpJ = -tmpJ;
        else if (tmpJ >= width)
            tmpJ -= (hks + 1);

//        printf("%d %d;;", j, tmpJ);
        //TODO: for debug purpose, remove after debugged!

        int addr = i * width + tmpJ;
        float curPx = d_Src[addr] / 255.0f;
        sum += d_Krn[hks + krnI] * curPx; // TODO: introduce shared memory

//        printf("(%d) %f * %f  -> %f\n", d_Src[addr], curPx, d_Krn[hks + krnI], sum);
    }


    d_Out[i * width + j] = sum;


}



void sepFilter(float* d_Out, uint8_t* d_Src, float* d_Krn, int width, int height, int krnSize)
{
//    dim3 threads(1, 1);
//    dim3 blocks(1, 1);
    dim3 threads(IMAGE_PATCH_HEIGHT, IMAGE_PATCH_WIDTH);
    dim3 blocks(height / threads.x + (height % threads.x ? 1 : 0), width / threads.y + (width % threads.y ? 1 : 0));

    sepFilterRows <<<blocks, threads>>> (d_Out, d_Src, d_Krn, width, height, krnSize);
}
