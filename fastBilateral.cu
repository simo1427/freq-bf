//
// Created by HP on 31/05/2024.
//

#include "fastBilateral.cuh"
#include "utils.cuh"
#include "spatial/separableConvolution.cuh"

#define BF_POPULATE_WIDTH 32
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
        }
    }

    cudaMemcpyToSymbol(d_trigLut, h_trigLut, 256 * MAX_COEFS_NUM * sizeof(float2));
}

__global__ void fastBFPopulate(uint8_t* d_Inp, float4* d_Buf, int width, int height, int k, size_t srcPitch, size_t bufPitch)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if (i >= height || j >= width)
        return;

    // TODO: shared memory? maybe first pitched memory, then shared...
    uint8_t* d_InpRow = d_Inp + i * srcPitch;

    uint8_t px = d_InpRow[j];
    // TODO: the trick for accessing many uint8_t's at the same time? I saw that recently in an NVIDIA presentation

    float pxScaled = static_cast<float>(px) / 255.0f;

    // order: x:cos y:sin z:cosIntensity w:sinIntensity

    float2 vals = d_trigLut[px][k];
    float4 tmp = make_float4(vals.x, vals.y, pxScaled * vals.x, pxScaled * vals.y);

    float4* d_BufRow = (float4*) ((char*) d_Buf + i * bufPitch);
    d_BufRow[j] = tmp;

}

__global__ void collectResults(float4* d_OutNonSummed, uint8_t* d_Inp,
                               float2* d_Out, int width, int height,
                               int k, size_t inpPitch, size_t bufPitch, size_t outPitch)
{
    // TODO: make d_Out a float2, then process on the cpu?
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if (i >= height || j >= width)
        return;

    float sum = 0;
    float W = 0; // normalization factor

    uint8_t* d_InpRow = d_Inp + i * inpPitch;
    uint8_t px = d_InpRow[j];

    float4* d_OutNonSummedRow = (float4*)((char*) d_OutNonSummed + i * bufPitch);
    float4 tmp = d_OutNonSummedRow[j];
    sum = tmp.x;
    W = tmp.y;
    float2 sinCosVals = d_trigLut[px][k];

    // PSNR 51.7611 dB
//        W += d_Coefs[k] * (sinCosVals.x * tmp.x + sinCosVals.y * tmp.y);
//        sum += d_Coefs[k] * (sinCosVals.x * tmp.z + sinCosVals.y * tmp.w);

    // PSNR 51.7611 dB
    W = __fmaf_rn(d_Coefs[k], sinCosVals.x * tmp.x + sinCosVals.y * tmp.y, W);
    sum = __fmaf_rn(d_Coefs[k], sinCosVals.x * tmp.z + sinCosVals.y * tmp.w, sum);

    float2* d_OutRow = (float2*) ((char*)d_Out + i * outPitch);
    d_OutRow[j] = make_float2(sum, W);
}

__global__ void obtainFinalImage(float2* d_OutSummed,
                               float* d_Out, int width, int height,
                               size_t inpPitch, size_t outPitch)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if (i >= height || j >= width)
        return;


    float2* d_OutSummedRow = (float2*) ((char*)d_OutSummed + i * inpPitch);
    float2 tmp = d_OutSummedRow[j];

    float* d_OutRow = (float*) ((char*)d_Out + i * outPitch);
    d_OutRow[j] = tmp.x / tmp.y;
}

void debugOutBuf(float4* h_BfBuf, int rows, int cols)
{
    cv::Mat dbgOut = cv::Mat(rows, cols, CV_32F);
    std::string filenames[] = {"./cosImg.tif", "./sinImg.tif", "./cosIntensityImg.tif", "./sinIntensityImg.tif"};

    for (int k = 0; k < 4; k++)
    {
        for (int i = 0; i < rows; i++)
        {
            float* ptrDbgOut = dbgOut.ptr<float>(i);

            union {
                float4 oneWord;
                float fourFloats[4];
            } tmp;

            for (int j = 0; j < cols; j++)
            {
                tmp.oneWord = h_BfBuf[i * cols + j];
                ptrDbgOut[j] = tmp.fourFloats[k];
            }
        }

        cv::imwrite(filenames[k], dbgOut);

    }

    dbgOut.release();
}

void BF_approx_gpu(cv::Mat &input, cv::Mat &output, cv::Mat &spatialKernel, double sigmaRange, range_krn_t rangeKrn, int numberOfCoefficients, float T)
{
    assert(input.type() == CV_8U);

    int width = input.cols;
    int height = input.rows;

    if (numberOfCoefficients == 0)
        // modified heuristic compared to Honours project
        numberOfCoefficients =(int)ceil(1.5 * 2 / (6 * sigmaRange)) + 1;

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

    size_t uint8Pitch, floatPitch, float2Pitch, float4Pitch;

    uint8_t* d_Inp;
    checkCudaErrors(cudaMallocPitch(&d_Inp, &uint8Pitch,
                                    input.cols * sizeof(uint8_t), input.rows));

    float2* d_OutSummed;
    checkCudaErrors(cudaMallocPitch(&d_OutSummed, &float2Pitch,
                                    input.cols * sizeof(float2), input.rows));

    float* d_Out;
    checkCudaErrors(cudaMallocPitch(&d_Out, &floatPitch,
                                    input.cols * sizeof(float2), input.rows));

    float4* d_BfBuf;
//    checkCudaErrors(cudaMalloc(&d_BfBuf, frameSize * numberOfCoefficients * sizeof(float4)));
    checkCudaErrors(cudaMallocPitch(&d_BfBuf, &float4Pitch,
                                    input.cols * sizeof(float4), input.rows));
    float4* d_OutNonSummed;
//    checkCudaErrors(cudaMalloc(&d_OutNonSummed, frameSize * numberOfCoefficients * sizeof(float4)));
    checkCudaErrors(cudaMallocPitch(&d_OutNonSummed, &float4Pitch,
                                    input.cols * sizeof(float4), input.rows));
    float4* d_OutNonSummedBuf;
//    checkCudaErrors(cudaMalloc(&d_OutNonSummedBuf, frameSize * numberOfCoefficients * sizeof(float4)));
    checkCudaErrors(cudaMallocPitch(&d_OutNonSummedBuf, &float4Pitch,
                                    input.cols * sizeof(float4), input.rows));

    // copy the image to the GPU

    checkCudaErrors(cudaMemcpy2D(d_Inp, uint8Pitch,
                                 input.ptr<uint8_t>(), input.cols * sizeof(uint8_t),
                                 input.cols * sizeof(uint8_t), input.rows,
                                 cudaMemcpyHostToDevice));

    // create events for measuring execution time
    cudaEvent_t start, finish;
    float elapsedTime;
    cudaEventCreate(&start);
    cudaEventCreate(&finish);

    // execute kernels

    dim3 populateThreads(BF_POPULATE_HEIGHT, BF_POPULATE_WIDTH);
    dim3 populateBlocks(height / populateThreads.x + (height % populateThreads.x ? 1 : 0), width / populateThreads.y + (width % populateThreads.y ? 1 : 0));

    // for debug image output
    float4* h_BfBuf = (float4*) malloc(frameSize * sizeof(float4));

//    printf("Number of coefficients: %d\n", numberOfCoefficients);
    cudaEventRecord(start, 0);


    // TODO: enqueue convolutions for each of the images in memory
//    printf("bufSize: %d\n", frameSize * numberOfCoefficients * sizeof(float4));
    for (int i = 0; i < numberOfCoefficients; i++) {
//        printf("Enqueued %d\n", i);
        fastBFPopulate<<<populateBlocks, populateThreads>>>(d_Inp,
                                                            d_BfBuf, width, height, i, uint8Pitch, float4Pitch);

        sepFilterf4(d_OutNonSummed,
                    d_BfBuf,
                    d_OutNonSummedBuf, // i * frameSize * sizeof(float4)
                    width,
                    height,
                    spatialKernel.rows,
                    float4Pitch);

//        checkCudaErrors(cudaMemcpy2D(h_BfBuf, input.cols * sizeof(float4),
//                                     d_OutNonSummed, float4Pitch,
//                                     input.cols * sizeof(float4), input.rows,
//                                     cudaMemcpyDeviceToHost));
//        debugOutBuf(h_BfBuf, input.rows, input.cols);

        collectResults<<<populateBlocks, populateThreads>>>(d_OutNonSummed,
                                                            d_Inp, d_OutSummed,
                                                            width, height,
                                                            i, uint8Pitch,
                                                            float4Pitch, float2Pitch);

    }

    obtainFinalImage<<<populateBlocks, populateThreads>>>(d_OutSummed, d_Out, width, height, float2Pitch, floatPitch);

    cudaEventRecord(finish, 0); // Beware of streams if they are going to be added later!
    cudaEventSynchronize(finish);
    cudaEventElapsedTime(&elapsedTime, start, finish);
    // copy result back from the GPU


//    checkCudaErrors(cudaMemcpy(output.ptr<float>(), d_OutSummed, frameSize * sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy2D(output.ptr<float>(), input.cols * sizeof(float),
                                 d_OutSummed, floatPitch,
                                 input.cols * sizeof(float), input.rows,
                                 cudaMemcpyDeviceToHost));


    free(h_BfBuf);

    printf("Elapsed time: %f ms\n", elapsedTime);

    // cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(finish);
    cudaFree(d_Inp);
    cudaFree(d_OutSummed);
    cudaFree(d_OutNonSummed);
    cudaFree(d_BfBuf);
    cudaFree(d_OutNonSummedBuf);
    cudaFree(d_Out);
    return;

}
