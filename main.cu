#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

#include "utils.cuh"
#include <CLI/CLI.hpp>
#include "rangeKernels.h"
#include "spatial/separableConvolution.cuh"
#include "fastBilateral.cuh"
//#include <assert.h>

#define DEBUG_OUT(x) std::cout << #x << "= " << x << "\n"

int main(int argc, char** argv) {

    CLI::App app{"Runner for the CUDA implementation of the fast bilateral filter using Fourier series"};


    std::string filename;

    app.add_option("file", filename, "Path to the input image");


    CLI11_PARSE(app, argc, argv);

    double sigmaSpatial = 5;
    int spatialKernelSize = static_cast<int>(round(sigmaSpatial * 1.5f) * 2 + 1);
    double sigmaRange = 0.1;
    float T = 2;
    int numberOfCoefficients = 10;

    std::cout << "Params: \n";
    DEBUG_OUT(sigmaRange);
    DEBUG_OUT(T);
    DEBUG_OUT(sigmaSpatial);

    cv::Mat frame = cv::imread(filename, cv::IMREAD_GRAYSCALE);
    cv::Mat intFrame = frame.clone();
    frame.convertTo(frame, CV_32F, 1.0 / 255, 0);
    std::cout << frame.rows << " " << frame.cols << " " << frame.isContinuous() << std::endl;
    if (!frame.isContinuous())
    {
        frame = frame.clone();
        if (!frame.isContinuous())
        {
            std::cerr << "Non-continuous cv::Mat!\n";
            return EXIT_FAILURE;
        }
    }
    cv::Mat filterOut(frame.rows, frame.cols, CV_32F);
    assert(filterOut.isContinuous());

    int w = frame.cols; // placeholder width value
    int h = frame.rows; // placeholder height value

    int frameSize = w * h;

    // Image filtering kernel
    cv::Mat kernel, kernel2D;

    //TODO: use parameters instead of hardcoded ones
    kernel = cv::getGaussianKernel(spatialKernelSize, sigmaSpatial, CV_32F);

    // call the BF function
    BF_approx_gpu(intFrame, filterOut, kernel, sigmaRange, gaussian, 0, T);

    std::cout << "Gaussian kernel:\n";
    for (int i = 0; i < kernel.rows; i++) {
        std::cout << kernel.ptr<float>()[i] << " ";
    }
    std::cout << std::endl;

    cv::imwrite("./gpuOut.tif", filterOut);
//
//    // CUDA buffers declaration
//
//    // - Buffers for the component images GPU kernels
//    // This has to be better thought out - the buffer should be contiguous in memory
//    // to ensure the last kernel can sum all components. This will be handled once
//    // the bilateral filter approximation will be implemented.
//
//
//    float* dInp;
//    checkCudaErrors(cudaMalloc(&dInp, frameSize * sizeof(float)));
//    float* dBuf;
//    checkCudaErrors(cudaMalloc(&dBuf, frameSize * sizeof(float)));
//    float* dOut;
//    checkCudaErrors(cudaMalloc(&dOut, frameSize * sizeof(float)));
//
//    // Load image into the inp buf
//    checkCudaErrors(cudaMemcpy(dInp, frame.ptr<float>(), frameSize * sizeof(float), cudaMemcpyHostToDevice));
//    setConvolutionKernel(kernel.ptr<float>(), kernel.rows);
//
//    // Execute kernel
//    sepFilter(dOut, dInp, dBuf, w, h, spatialKernelSize);
//    // Copy back from GPU
//
//    checkCudaErrors(cudaMemcpy(filterOut.ptr(), dOut, frameSize * sizeof(float), cudaMemcpyDeviceToHost));
//
////    cv::imshow("in", frame);
////    cv::imshow("out", filterOut);
//    cv::imwrite("./out.tif", filterOut);
//
//    auto dstGold = cv::Mat(frame.rows, frame.cols, CV_32F);
//
//    cv::sepFilter2D(frame, dstGold, CV_32F, kernel, kernel);
//
//    cv::Mat diff = dstGold - filterOut;
//
//    float mse = 0;
//
//    for (int i = 0; i < diff.rows ; i++) {
//        float* errPtr = diff.ptr<float>(i);
//        for (int j = 0; j < diff.cols; j++) {
//            mse += errPtr[j] * errPtr[j];
//        }
//    }
//
//    mse /= ((diff.rows) * (diff.cols));
//
//    std::cout << "PSNR: " << 10 * log10(1 / mse) << " dB\n";
//    cv::imwrite("./cv.tif", dstGold);
//    cv::imwrite("./diff.tif", diff);
//
////    cv::waitKey(0);
//    // Deallocate
//
//    cudaFree(dInp);
//    cudaFree(dOut);
//    cudaFree(dBuf);

    frame.release();
    intFrame.release();
    filterOut.release();
    kernel.release();


//    cv::destroyAllWindows();
    return 0;
}


