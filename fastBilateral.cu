//
// Created by HP on 31/05/2024.
//

#include "fastBilateral.cuh"
#include "utils.cuh"

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
            h_trigLut[j][k].x = cos(2 * M_PI * k / T);
            h_trigLut[j][k].y = sin(2 * M_PI * k / T);
        }
    }

    cudaMemcpyToSymbol(d_trigLut, h_trigLut, 256 * MAX_COEFS_NUM * sizeof(float));
}

__global__ void fastBFPopulate(uint8_t* d_Inp, float4* d_Buf, int width, int height, int numberOfCoefficients)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if (i >= height || j >= width)
        return;

    // TODO: shared memory? maybe first pitched memory, then shared...
    uint8_t px = d_Inp[i * width + j];

    for (int k = 0; k < numberOfCoefficients; k++)
    {
        float4 tmp; // order: w:cos x:sin ycosIntensity z:sinIntensity

        float2 vals = d_trigLut[px][k];
        tmp.w = vals.x;
        tmp.x = vals.y;
        tmp.y = px * vals.x;
        tmp.z = px * vals.y;
        d_Buf[i * width + j] = tmp;
    }
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

    // Allocate arrays for intermediate images

    int frameSize = input.rows * input.cols;

    uint8_t* d_Inp;
    checkCudaErrors(cudaMalloc(&d_Inp, frameSize * sizeof(uint8_t)));

    float* d_Out;
    checkCudaErrors(cudaMalloc(&d_Out, frameSize * sizeof(float)));

    float4* d_Buf;
    checkCudaErrors(cudaMalloc(&d_Buf, frameSize * numberOfCoefficients * sizeof(float4)));

    // copy the image to the GPU

    checkCudaErrors(cudaMemcpy(d_Inp, input.ptr<uint8_t>(), frameSize * sizeof(uint8_t), cudaMemcpyHostToDevice));

    // execute kernels

    dim3 populateThreads(BF_POPULATE_HEIGHT, BF_POPULATE_WIDTH);
    dim3 populateBlocks(height / populateThreads.x + (height % populateThreads.x ? 1 : 0), width / populateThreads.y + (width % populateThreads.y ? 1 : 0));
    fastBFPopulate<<<populateBlocks, populateThreads>>>(d_Inp, d_Buf, width, height, numberOfCoefficients);

    // TODO: enqueue convolutions for each of the images in memory


    // copy result back from the GPU

    checkCudaErrors(cudaMemcpy(output.ptr<float>(), d_Out, frameSize * sizeof(float), cudaMemcpyDeviceToHost));

    return;

}
