#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

#include "utils.cuh"
#include <CLI/CLI.hpp>
//#include <assert.h>


__global__ void testKernel(uint8_t* src, uint8_t* dst, int imageW, int imageH)
{
    int i = blockIdx.x * 16 + threadIdx.x;
    int j = blockIdx.y * 16 + threadIdx.y;
    if (i >= imageH || j >= imageW)
        return;
    dst[i * imageW + j] = 255 - src[i * imageW + j];
}


int main(int argc, char** argv) {

    CLI::App app{"Runner for the CUDA implementation of the fast bilateral filter using Fourier series"};


    std::string filename;

    app.add_option("file", filename, "Path to the input image");


    CLI11_PARSE(app, argc, argv);

    double sigmaSpatial = 8;
    int spatialKernelSize = -1;
    double sigmaRange = 0.35;
//    range_krn_t rangeKrn = gaussian;
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
    cv::Mat filterOut(frame.rows, frame.cols, CV_8U);

    int w = frame.cols; // placeholder width value
    int h = frame.rows; // placeholder height value

    int frameSize = w * h;

    // CUDA buffers declaration

    // - Buffers for the component images GPU kernels
    // This has to be better thought out - the buffer should be contiguous in memory
    // to ensure the last kernel can sum all components. This will be handled once
    // the bilateral filter approximation will be implemented.

    uint8_t* dInp;
    checkCudaErrors(cudaMalloc(&dInp, frameSize * sizeof(uint8_t)));
    uint8_t* dOut;
    checkCudaErrors(cudaMalloc(&dOut, frameSize * sizeof(uint8_t)));

    // Load image into the inp buf
    checkCudaErrors(cudaMemcpy(dInp, frame.ptr(), frameSize * sizeof(uint8_t), cudaMemcpyHostToDevice));

    // Execute kernel

    dim3 blocks(h / 16 + (h % 16 ? 1 : 0), w / 16 + (w % 16 ? 1 : 0));
    std::cout << blocks.x << " " << blocks.y << std::endl;

    dim3 threads(16, 16);

    testKernel<<<blocks, threads>>>(dInp, dOut, w, h);

    // Copy back from GPU

    checkCudaErrors(cudaMemcpy(filterOut.ptr(), dOut, frameSize * sizeof(uint8_t), cudaMemcpyDeviceToHost));

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


