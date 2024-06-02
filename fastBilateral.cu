//
// Created by HP on 31/05/2024.
//

#include "fastBilateral.cuh"
#include "utils.cuh"
#include "spatial/separableConvolution.cuh"

#define BF_POPULATE_WIDTH 16
#define BF_POPULATE_HEIGHT 16


#define MAX_COEFS_NUM 20 // TODO: check for a better value, possibly derived from the amount of VRAM

__constant__ float d_Coefs[MAX_COEFS_NUM];
__constant__ float2 d_trigLut[256][MAX_COEFS_NUM];
// TODO: is this an efficient memory layout?
// shouldn't be a problem, as constant memory should take care of that according to the CUDA C++ Programming Guide

void setCoefficients(float* h_Coefs, int n)
{
    cudaMemcpyToSymbol(d_Coefs, h_Coefs, n);
}

void populateLut(int numberOfCoefficients, float T)
{
    float2 h_trigLut[256][MAX_COEFS_NUM];
    for (int k = 0; k < numberOfCoefficients; k++)
    {
        for (int j = 0; j < 256; j++)
        {
            h_trigLut[j][k].x = cosf(static_cast<float>(j / 255.0f) * 2.0f * M_PI * k / T);
            h_trigLut[j][k].y = sinf(static_cast<float>(j / 255.0f) * 2.0f * M_PI * k / T);
            if (j == 186)
                printf("LUT vals for 186: %f %f\n", h_trigLut[j][k].x, h_trigLut[j][k].y);
        }
    }

    cudaMemcpyToSymbol(d_trigLut, h_trigLut, 256 * MAX_COEFS_NUM * sizeof(float2));
}

__global__ void fastBFPopulate(uint8_t* d_Inp, float4* d_Buf, int width, int height, int numberOfCoefficients)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

//    if (i == 0 && j == 0)
//    {
//        printf("coefficients from const mem:\n");
//        for (int k = 0; k < numberOfCoefficients; k++)
//            printf("%f,", d_Coefs[k]);
//        printf("number of coefs: %d\n", numberOfCoefficients);
//    }

    if (i >= height || j >= width)
        return;

    // TODO: shared memory? maybe first pitched memory, then shared...
    uint8_t px = d_Inp[i * width + j];

    float pxScaled = static_cast<float>(px) / 255.0f;
    for (int k = 0; k < numberOfCoefficients; k++)
    {
        // order: x:cos y:sin z:cosIntensity w:sinIntensity

        float2 vals = d_trigLut[px][k];
        float4 tmp = make_float4(vals.x, vals.y, pxScaled * vals.x, pxScaled * vals.y);
        d_Buf[k * width * height + i * width + j] = tmp;
//        if (i == 0 && j == 0)
        if (i == 185 && j == 25)
        {
//            printf("%d: (%f %f),", k, vals.x, vals.y);
            printf("%d/%f %d: (%f %f %f %f)", px, pxScaled, k, tmp.x, tmp.y, tmp.z, tmp.w);

        }
        // TODO: maybe this is causing some errors?
    }
}

__global__ void collectResults(float4* d_OutNonSummed, uint8_t* d_Inp, float* d_Out, int width, int height, int numOfCoefs)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if (i >= height || j >= width)
        return;

    float sum = 0;
    float W = 0; // normalization factor
    uint8_t px = d_Inp[i * width + j];

    for (int k = 0; k < numOfCoefs; k++)
    {
        float4 tmp = d_OutNonSummed[k * width * height + i * width + j]; // TODO: maybe this is causing some errors?
        float2 sinCosVals = d_trigLut[px][k];
        W += d_Coefs[k] * (sinCosVals.x * tmp.x + sinCosVals.y * tmp.y);
        sum += d_Coefs[k] * (sinCosVals.x * tmp.z + sinCosVals.y * tmp.w);
        if (i == 185 && j == 25)
            printf("%d %d: (sum=%f, W=%f);", px, k, sum, W);
    }

    d_Out[i * width + j] = sum / W;
}

void BF_approx_gpu(cv::Mat &input, cv::Mat &output, cv::Mat &spatialKernel, double sigmaRange, range_krn_t rangeKrn, int numberOfCoefficients, float T)
{
    assert(input.type() == CV_8U);

    int width = input.cols;
    int height = input.rows;

    if (numberOfCoefficients == 0)
        // modified heuristic compared to Honours project
        numberOfCoefficients =(int)ceil(3 * 2 / (6 * sigmaRange)) + 1;

    auto doubleCoefs = getFourierCoefficients(sigmaRange, T, numberOfCoefficients, rangeKrn);
    std::vector<float> coefsVec{doubleCoefs.begin(), doubleCoefs.end()};



#ifdef DEBUG_PRINT_FOURIER
    std::cout << "Fourier coefs:\n";
    for (int i = 0; i < coefs.size(); i++)
    {
        std::cout << coefs[i] << " ";
    }
    std::cout << std::endl;
#endif

    // Copy the coefficients to constant memory
    setCoefficients(coefsVec.data(), coefsVec.size() * sizeof(float));

    populateLut(numberOfCoefficients, T);
    setConvolutionKernel(spatialKernel.ptr<float>(), spatialKernel.rows);

    // Allocate arrays for intermediate images

    int frameSize = input.rows * input.cols;

    uint8_t* d_Inp;
    checkCudaErrors(cudaMalloc(&d_Inp, frameSize * sizeof(uint8_t)));

    float* d_OutSummed;
    checkCudaErrors(cudaMalloc(&d_OutSummed, frameSize * sizeof(float)));

    float4* d_BfBuf;
    checkCudaErrors(cudaMalloc(&d_BfBuf, frameSize * numberOfCoefficients * sizeof(float4)));
    float4* d_OutNonSummed;
    checkCudaErrors(cudaMalloc(&d_OutNonSummed, frameSize * numberOfCoefficients * sizeof(float4)));
    float4* d_OutNonSummedBuf;
    checkCudaErrors(cudaMalloc(&d_OutNonSummedBuf, frameSize * numberOfCoefficients * sizeof(float4)));

    // copy the image to the GPU

    checkCudaErrors(cudaMemcpy(d_Inp, input.ptr<uint8_t>(), frameSize * sizeof(uint8_t), cudaMemcpyHostToDevice));

    // execute kernels

    dim3 populateThreads(BF_POPULATE_HEIGHT, BF_POPULATE_WIDTH);
    dim3 populateBlocks(height / populateThreads.x + (height % populateThreads.x ? 1 : 0), width / populateThreads.y + (width % populateThreads.y ? 1 : 0));

    printf("Number of coefficients: %d\n", numberOfCoefficients);
    fastBFPopulate<<<populateBlocks, populateThreads>>>(d_Inp, d_BfBuf, width, height, numberOfCoefficients);

    // TODO: enqueue convolutions for each of the images in memory
    printf("bufSize: %d\n", frameSize * numberOfCoefficients * sizeof(float4));
    for (int i = 0; i < numberOfCoefficients; i++) {
        printf("Enqueued %d\n", i);
        sepFilterf4(d_OutNonSummed + i * frameSize,
                    d_BfBuf + i * frameSize,
                    d_OutNonSummedBuf + i * frameSize, // i * frameSize * sizeof(float4)
                    width,
                    height,
                    spatialKernel.rows);
    }

    cudaDeviceSynchronize(); //might not be necessary if the calls are on the same stream
    printf("successfully sync'ed\n");

    collectResults<<<populateBlocks, populateThreads>>>(d_OutNonSummed, d_Inp, d_OutSummed, width, height, numberOfCoefficients);
    // copy result back from the GPU

    cudaDeviceSynchronize(); //might not be necessary if the calls are on the same stream
    printf("successfully sync'ed 2!\n");

    float4* h_BfBuf = (float4*) malloc(frameSize * numberOfCoefficients * sizeof(float4));

    checkCudaErrors(cudaMemcpy(h_BfBuf, d_BfBuf, numberOfCoefficients * frameSize * sizeof(float4), cudaMemcpyDeviceToHost));

    auto debugOut = cv::Mat(height, width, CV_32F);
    for (int i = 0; i < height; i++)
    {
        float* ptrDebugOut = debugOut.ptr<float>(i);
        for (int j = 0; j < width; j++)
        {
            ptrDebugOut[j] = h_BfBuf[2 * width * height + i * width + j].x;
        }
    }
    cv::imwrite("./cosImg.tif", debugOut, {cv::ImwriteFlags::IMWRITE_TIFF_COMPRESSION, 0});

    for (int i = 0; i < height; i++)
    {
        float* ptrDebugOut = debugOut.ptr<float>(i);
        for (int j = 0; j < width; j++)
        {
            ptrDebugOut[j] = h_BfBuf[2 * width * height + i * width + j].y;
        }
    }
    cv::imwrite("./sinImg.tif", debugOut, {cv::ImwriteFlags::IMWRITE_TIFF_COMPRESSION, 0});

    for (int i = 0; i < height; i++)
    {
        float* ptrDebugOut = debugOut.ptr<float>(i);
        for (int j = 0; j < width; j++)
        {
            ptrDebugOut[j] = h_BfBuf[2 * width * height + i * width + j].z;
        }
    }
    cv::imwrite("./cosIntensityImg.tif", debugOut, {cv::ImwriteFlags::IMWRITE_TIFF_COMPRESSION, 0});

    for (int i = 0; i < height; i++)
    {
        float* ptrDebugOut = debugOut.ptr<float>(i);
        for (int j = 0; j < width; j++)
        {
            ptrDebugOut[j] = h_BfBuf[2 * width * height + i * width + j].w;
        }
    }
    cv::imwrite("./sinIntensityImg.tif", debugOut, {cv::ImwriteFlags::IMWRITE_TIFF_COMPRESSION, 0});

    checkCudaErrors(cudaMemcpy(output.ptr<float>(), d_OutSummed, frameSize * sizeof(float), cudaMemcpyDeviceToHost));


    cudaFree(d_Inp);
    cudaFree(d_OutSummed);
    cudaFree(d_OutNonSummed);
    cudaFree(d_BfBuf);
    cudaFree(d_OutNonSummedBuf);
    free(h_BfBuf);
    return;

}
