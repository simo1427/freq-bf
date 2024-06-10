#include <opencv2/opencv.hpp>
#include <stdio.h>

#include "../spatial/separableConvolution.cuh"
#include "../rangeKernels.h"
#include "sepFilterEval.cuh"
#include "utils/cuda_utils.cuh"

void sepConvEval(cv::Mat &input, cv::Mat &output, cv::Mat &spatialKernel, int numOfRuns) {
    // Initialization
    assert(input.type() == CV_8U);

    int width = input.cols;
    int height = input.rows;

    cv::Mat inputF32(input.rows, input.cols, CV_32F);
    input.convertTo(inputF32, CV_32F, 1.0f/255, 0);

    setConvolutionKernel(spatialKernel.ptr<float>(), spatialKernel.rows);

    // Allocate arrays for intermediate images

    int frameSize = input.rows * input.cols;

    size_t floatPitch;

    float *d_Inp;
    checkCudaErrors(cudaMallocPitch(&d_Inp, &floatPitch,
                                    input.cols * sizeof(float), input.rows));

    float *d_Out;
    checkCudaErrors(cudaMallocPitch(&d_Out, &floatPitch,
                                    input.cols * sizeof(float), input.rows));


    float *d_Buf;
    checkCudaErrors(cudaMallocPitch(&d_Buf, &floatPitch,
                                    input.cols * sizeof(float), input.rows));

    // copy the image to the GPU

    checkCudaErrors(cudaMemcpy2D(d_Inp, floatPitch,
                                 input.ptr<uint8_t>(), input.cols * sizeof(uint8_t),
                                 input.cols * sizeof(uint8_t), input.rows,
                                 cudaMemcpyHostToDevice));

    std::vector<float> runs(numOfRuns);
    float elapsedTime;
    // Run the filter
    for (int run = 0; run < numOfRuns; run++) {
        cudaEvent_t start, finish;

        cudaEventCreate(&start);
        cudaEventCreate(&finish);

        cudaEventRecord(start, 0);

        sepFilter(d_Out, d_Inp, d_Buf, width, height, spatialKernel.rows,floatPitch);

        cudaEventRecord(finish, 0); // Beware of streams if they are going to be added later!
        cudaEventSynchronize(finish);
        cudaEventElapsedTime(&elapsedTime, start, finish);

        runs[run] = elapsedTime;

        cudaEventDestroy(start);
        cudaEventDestroy(finish);
    }

    // Print
    printf("Average of %d runs\n", numOfRuns);
    double totalElapsedTime = 0;
    for (int run = 0; run < numOfRuns; run++) {
        totalElapsedTime += runs[run];
        printf("%d: %f ms\n", run + 1, runs[run]);
    }
    totalElapsedTime /= numOfRuns; // [in milliseconds]

    double standardDev = 0;
    if (numOfRuns == 1) {
        standardDev = NAN;
        printf("Average elapsed time: %f ms\n", totalElapsedTime);
    } else {
        for (int run = 0; run < numOfRuns; run++) {
            double diff = runs[run] - totalElapsedTime;
            standardDev += diff * diff;
        }
        standardDev = sqrt(standardDev / numOfRuns);
        printf("Average elapsed time: %f ms +/- %f ms\n", totalElapsedTime, standardDev);
    }

    double elapsedTimeSec = totalElapsedTime * 1e-3; // [in seconds]


    printf("Throughput: %f MP/s\n", width * height / (elapsedTimeSec * 1e6));

    // Cleanup

    cudaFree(d_Inp);
    cudaFree(d_Out);
    cudaFree(d_Out);
    cudaFree(d_Buf);
}