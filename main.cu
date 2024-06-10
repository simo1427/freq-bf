#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

#include "utils/cuda_utils.cuh"
#include <CLI/CLI.hpp>
#include "rangeKernels.h"
#include "spatial/separableConvolution.cuh"
#include "fastBilateral.cuh"
#include "refBilateral.h"
#include "evaluators/psnrRunner.h"
//#include <assert.h>

#define PSNR_OUT

#define DEBUG_OUT(x) std::cout << #x << "= " << x << "\n"

int main(int argc, char **argv) {

    CLI::App app{"Runner for the CUDA implementation of the fast bilateral filter using Fourier series"};

    double sigmaSpatial = 8;
    int spatialKernelSize = static_cast<int>(round(sigmaSpatial * 1.5f) * 2 + 1);
    DEBUG_OUT(spatialKernelSize);
    double sigmaRange = 0.1;
    float T = 2;
    int numberOfCoefficients = 0;
    std::string rangeKrnName;
    range_krn_t rangeKrn = gaussian;

    // Variables for the perf evaluation
    double startSigmaRange = 0.1;
    double endSigmaRange = 1.0;
    int numOfVals = 10;

    int startKernelSize = 9;
    int endKernelSize = 11;

    int numberOfRuns = 10; // excluding the first one

    app.add_option("--spatial", sigmaSpatial, "The sigma of the spatial kernel");
    app.add_option("--range", sigmaRange, "The sigma of the range kernel");
    app.add_option("--rangeKrn", rangeKrnName,
                   "The range kernel in use (gaussian - default, gaussianUnscaled, huber, tukey, lorentz)");
    app.add_option("--kernelSize", spatialKernelSize,
                   "The size of the spatial kernel; if not provided, it is computed based on the spatial sigma");
    app.add_option("--coefs", numberOfCoefficients, "The number of coefficients to use for the filtering");

    CLI::Option *colArg = app.add_flag("--colour", "Specifies whether the filtering is done on a colour image");
    CLI::Option *boxArg = app.add_flag("--box",
                                       "Use a box spatial kernel instead of a Gaussian. The size will be computed from ");


    CLI::Option *perfArg = app.add_flag("--perf-eval",
                                        "Run a performance evaluation on the bilateral filter. "
                                        "Will run different combination of params a number of times and save a CSV");

    app.add_option("--perf-sigma-start", startSigmaRange, "Start value for sigmaRange")->needs(perfArg);
    app.add_option("--perf-sigma-end", endSigmaRange, "End value for sigmaRange")->needs(perfArg);
    app.add_option("--perf-sigma-num", numOfVals, "Number of values between the start and end of sigmaRange")->needs(
            perfArg);

    app.add_option("--perf-kernel-start", startKernelSize, "Start value for the kernel size")->needs(perfArg);
    app.add_option("--perf-kernel-end", endKernelSize, "End value for the kernel size")->needs(perfArg);
    app.add_option("--perf-runs", numberOfRuns,
                   "Number of runs per combination of sigmaRange and kernel size (does not count the first run that is discarded)")->needs(
            perfArg);

    CLI::Option *psnrArg = app.add_flag("--psnr",
                                        "Run an evaluation on the approximation based on PSNR"
                                        "Will run different combination of params, measure PSNR and save a CSV")->excludes(
            perfArg);
    perfArg = perfArg->excludes(psnrArg); // mutually exclude the perf runner and the psnr analysis

    app.add_option("--psnr-sigma-start", startSigmaRange, "Start value for sigmaRange")->needs(psnrArg);
    app.add_option("--psnr-sigma-end", endSigmaRange, "End value for sigmaRange")->needs(psnrArg);
    app.add_option("--psnr-sigma-num", numOfVals, "Number of values between the start and end of sigmaRange")->needs(
            psnrArg);

    app.add_option("--psnr-kernel-start", startKernelSize, "Start value for the kernel size")->needs(psnrArg);
    app.add_option("--psnr-kernel-end", endKernelSize, "End value for the kernel size")->needs(psnrArg);

    std::string filename;

    app.add_option("file", filename, "Path to the input image");


    CLI11_PARSE(app, argc, argv);

    rangeKrn = rangeKrnProvider(rangeKrnName);


    std::cout << "Params: \n";
    DEBUG_OUT(sigmaRange);
    DEBUG_OUT(T);
    DEBUG_OUT(sigmaSpatial);

    cv::Mat frame = cv::imread(filename, cv::IMREAD_GRAYSCALE);
    cv::Mat intFrame = frame.clone();
    frame.convertTo(frame, CV_32F, 1.0 / 255, 0);
    std::cout << frame.rows << " " << frame.cols << " " << frame.isContinuous() << std::endl;
    if (!frame.isContinuous()) {
        frame = frame.clone();
        if (!frame.isContinuous()) {
            std::cerr << "Non-continuous cv::Mat!\n";
            return EXIT_FAILURE;
        }
    }
    cv::Mat filterOut(frame.rows, frame.cols, CV_32F);
    assert(filterOut.isContinuous());

    // Check if the program is in evaluation mode for PSNR
    if (psnrArg->count()) {
        std::string kernels[] = {"gaussian", "tukey", "huber", "lorentz"};
        for (std::string &kernelName: kernels) {
            std::cout << "Evaluating " << kernelName << "\n";

            runPsnrMeasure(intFrame,
                           startSigmaRange, endSigmaRange, numOfVals,
                           startKernelSize, endKernelSize,
                           kernelName, numberOfCoefficients);
        }
        exit(EXIT_SUCCESS);
    }

    int w = frame.cols; // placeholder width value
    int h = frame.rows; // placeholder height value

    int frameSize = w * h;

    // Image filtering kernel
    cv::Mat kernel, kernel2D;

    //TODO: use parameters instead of hardcoded ones
    kernel = cv::getGaussianKernel(spatialKernelSize, sigmaSpatial, CV_32F);
    kernel2D = kernel * kernel.t(); // for the CPU method, the kernel needs to be a 2D array!!!

    // call the BF function
    BF_approx_gpu(intFrame, filterOut, kernel, sigmaRange, rangeKrn, numberOfCoefficients, T);

    cv::imwrite("./gpuOut.tif", filterOut);

#ifdef PSNR_OUT
    // Measure PSNR compared with CPU slow BF
    auto bfGold = cv::Mat(frame.rows, frame.cols, CV_32F);
//
    BF(frame, bfGold, kernel2D, sigmaRange, rangeKrn);

//
    double psnr = psnrCompute(bfGold, filterOut, spatialKernelSize, true);
    std::cout << "PSNR: " << psnr << " dB\n";

    cv::imwrite("./slow.tif", bfGold);

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


