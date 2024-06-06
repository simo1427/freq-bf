#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

#include "utils.cuh"
#include <CLI/CLI.hpp>
#include "rangeKernels.h"
#include "spatial/separableConvolution.cuh"
#include "fastBilateral.cuh"
#include "refBilateral.h"
//#include <assert.h>

#define DEBUG_OUT(x) std::cout << #x << "= " << x << "\n"

int main(int argc, char** argv) {

    CLI::App app{"Runner for the CUDA implementation of the fast bilateral filter using Fourier series"};


    std::string filename;

    app.add_option("file", filename, "Path to the input image");


    CLI11_PARSE(app, argc, argv);

    double sigmaSpatial = 5;
    int spatialKernelSize = static_cast<int>(round(sigmaSpatial * 1.5f) * 2 + 1);
    DEBUG_OUT(spatialKernelSize);
    double sigmaRange = 0.1;
    float T = 2;
    int numberOfCoefficients = 10;
    range_krn_t rangeKrn = gaussian;

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
    kernel2D = kernel * kernel.t(); // for the CPU method, the kernel needs to be a 2D array!!!

    // call the BF function
    BF_approx_gpu(intFrame, filterOut, kernel, sigmaRange, rangeKrn, 0, T);

    std::cout << "Gaussian kernel:\n";
    for (int i = 0; i < kernel.rows; i++) {
        std::cout << kernel.ptr<float>()[i] << " ";
    }
    std::cout << std::endl;

    cv::imwrite("./gpuOut.tif", filterOut);

#ifdef PSNR_OUT
    // Measure PSNR compared with CPU slow BF
    auto bfGold = cv::Mat(frame.rows, frame.cols, CV_32F);
//
    BF(frame, bfGold, kernel2D, sigmaRange, rangeKrn);

//
    cv::Mat diff = bfGold - filterOut;

    float mse = 0;
    int hks = kernel.rows / 2;

    for (int i = hks; i < diff.rows - hks ; i++) {
        float* errPtr = diff.ptr<float>(i);
        for (int j = hks; j < diff.cols - hks; j++) {
            mse += errPtr[j] * errPtr[j];
        }
    }

    mse /= ((diff.rows) * (diff.cols));

    std::cout << "PSNR: " << 10 * log10(1 / mse) << " dB\n";
    cv::imwrite("./slow.tif", bfGold);
    cv::imwrite("./diff.tif", diff);

#endif
//    cv::waitKey(0);
    // Deallocate
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


