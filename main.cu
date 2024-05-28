#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

#include "utils.cuh"
#include <CLI/CLI.hpp>
#include "rangeKernels.h"
#include "filtering.cuh"
//#include <assert.h>



int main(int argc, char** argv) {

    CLI::App app{"Runner for the CUDA implementation of the fast bilateral filter using Fourier series"};


    std::string filename;

    app.add_option("file", filename, "Path to the input image");


    CLI11_PARSE(app, argc, argv);

    double sigmaSpatial = 8;
    int spatialKernelSize = round(sigmaSpatial * 1.5f) * 2 + 1;;
    double sigmaRange = 0.35;
    double T = 2;
    int numberOfCoefficients = 10;
    cv::Mat frame = cv::imread(filename, cv::IMREAD_GRAYSCALE);
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
    kernel = cv::getGaussianKernel(33, 8, CV_32F);

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


    uint8_t* dInp;
    checkCudaErrors(cudaMalloc(&dInp, frameSize * sizeof(uint8_t)));
    float* dOut;
    checkCudaErrors(cudaMalloc(&dOut, frameSize * sizeof(float)));

    float* dKrn;
    checkCudaErrors(cudaMalloc(&dKrn, kernel.rows * sizeof(float)));

    // Load image into the inp buf
    checkCudaErrors(cudaMemcpy(dInp, frame.ptr(), frameSize * sizeof(uint8_t), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dKrn, kernel.ptr<float>(), kernel.rows * sizeof(float),  cudaMemcpyHostToDevice));

    // Execute kernel

    sepFilter(dOut, dInp, dKrn, w, h, 33);

    // Copy back from GPU

    checkCudaErrors(cudaMemcpy(filterOut.ptr(), dOut, frameSize * sizeof(float), cudaMemcpyDeviceToHost));

    cv::imshow("in", frame);
    cv::imshow("out", filterOut);
    cv::imwrite("./out.tif", filterOut);
    cv::waitKey(0);
    // Deallocate

    cudaFree(dInp);
    cudaFree(dOut);

    cv::destroyAllWindows();
    return 0;
}


