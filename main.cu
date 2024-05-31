#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

#include "utils.cuh"
#include <CLI/CLI.hpp>
#include "rangeKernels.h"
#include "spatial/separableConvolution.cuh"
//#include <assert.h>

#define DEBUG_OUT(x) std::cout << #x << "= " << x << "\n"

int main(int argc, char** argv) {

    CLI::App app{"Runner for the CUDA implementation of the fast bilateral filter using Fourier series"};


    std::string filename;

    app.add_option("file", filename, "Path to the input image");


    CLI11_PARSE(app, argc, argv);

    double sigmaSpatial = 8;
    int spatialKernelSize = static_cast<int>(round(sigmaSpatial * 1.5f) * 2 + 1);
    DEBUG_OUT(spatialKernelSize);
    double sigmaRange = 0.35;
    double T = 2;
    int numberOfCoefficients = 10;
    cv::Mat frame = cv::imread(filename, cv::IMREAD_GRAYSCALE);
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

    int w = frame.cols; // placeholder width value
    int h = frame.rows; // placeholder height value

    int frameSize = w * h;

    // Image filtering kernel
    cv::Mat kernel, kernel2D;

    //TODO: use parameters instead of hardcoded ones
    kernel = cv::getGaussianKernel(spatialKernelSize, sigmaSpatial, CV_32F);

    std::cout << "Gaussian kernel:\n";
    for (int i = 0; i < kernel.rows; i++) {
        std::cout << kernel.ptr<float>()[i] << " ";
    }
    std::cout << std::endl;

    // CUDA buffers declaration

    // - Buffers for the component images GPU kernels
    // This has to be better thought out - the buffer should be contiguous in memory
    // to ensure the last kernel can sum all components. This will be handled once
    // the bilateral filter approximation will be implemented.

    size_t pitch;
    float* dInp;
    checkCudaErrors(cudaMallocPitch(&dInp, &pitch, frame.cols * sizeof(float), frame.rows));
    float* dBuf;
    checkCudaErrors(cudaMallocPitch(&dBuf, &pitch, frame.cols * sizeof(float), frame.rows));
    float* dOut;
    checkCudaErrors(cudaMallocPitch(&dOut, &pitch, frame.cols * sizeof(float), frame.rows));

    // Load image into the inp buf
    checkCudaErrors(cudaMemcpy2D(dInp, pitch,
                                 frame.ptr<float>(), frame.cols * sizeof(float),
                                 frame.cols * sizeof(float), frame.rows,
                                 cudaMemcpyHostToDevice));
    setConvolutionKernel(kernel.ptr<float>(), kernel.rows);

    // Execute kernel
    sepFilter(dOut, dInp, dBuf, w, h, spatialKernelSize, pitch);
    sepFilter(dOut, dInp, dBuf, w, h, spatialKernelSize, pitch); // use these timings

    // Copy back from GPU

    //TODO: uncomment that, important!!!!
//    checkCudaErrors(cudaMemcpy(filterOut.ptr(), dOut, frameSize * sizeof(float), cudaMemcpyDeviceToHost));

//    checkCudaErrors(cudaMemcpy(filterOut.ptr(), dBuf , frameSize * sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy2D(filterOut.ptr<float>(), frame.cols * sizeof(float),
                                 dOut, pitch,
                                 frame.cols * sizeof(float), frame.rows,
                                 cudaMemcpyDeviceToHost));
//    cv::imshow("in", frame);
//    cv::imshow("out", filterOut);
    cv::imwrite("./out.tif", filterOut);

    auto dstGold = cv::Mat(frame.rows, frame.cols, CV_32F);

    cv::sepFilter2D(frame, dstGold, CV_32F, kernel, kernel);

    cv::Mat diff = dstGold - filterOut;

    float mse = 0;

    for (int i = 0; i < diff.rows ; i++) {
        float* errPtr = diff.ptr<float>(i);
        for (int j = 0; j < diff.cols; j++) {
            mse += errPtr[j] * errPtr[j];
        }
    }

    mse /= ((diff.rows) * (diff.cols));

    std::cout << "PSNR: " << 10 * log10(1 / mse) << " dB\n";
    cv::imwrite("./cv.tif", dstGold);
    cv::imwrite("./diff.tif", diff);

//    cv::waitKey(0);
    // Deallocate

    cudaFree(dInp);
    cudaFree(dOut);
    cudaFree(dBuf);

    frame.release();
    filterOut.release();
    kernel.release();


//    cv::destroyAllWindows();
    return 0;
}


