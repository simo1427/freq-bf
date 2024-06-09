#include <opencv2/opencv.hpp>
#include <fstream>

#include "psnrRunner.h"
#include "rangeKernels.h"
#include "refBilateral.h"
#include "fastBilateral.cuh"

double psnrCompute(cv::Mat &ref, cv::Mat &our, int kernelSize, bool writeDiffImg = false) {
    cv::Mat err = ref - our;

    if (writeDiffImg)
        cv::imwrite("./diffPsnrCompute.tif", err, {cv::ImwriteFlags::IMWRITE_TIFF_COMPRESSION, 0});

    float mse = 0;
    int hks = kernelSize / 2;

    for (int i = hks; i < err.rows - hks; i++) {
        float *errPtr = err.ptr<float>(i);
        for (int j = hks; j < err.cols - hks; j++) {
            mse += errPtr[j] * errPtr[j];
        }
    }

    mse /= ((err.rows - 2 * hks) * (err.cols - 2 * hks));

    return 10 * log10(1 / mse); // we work with float images, so max is 1^2, not 255^2

}

// TODO: reorder arguments
void runPsnrMeasure(cv::Mat image, bool autoCoefs,
                    std::string rangeKrnName, int numOfVals,
                    double startSigmaRange, double endSigmaRange,
                    int startKernelSize, int endKernelSize, float T = 2) {
    /*
     * Here the precision of the filtering in terms of PSNR will be evaluated.
     * OpenCV will _not_ be used.
     * Note that from empirical observations the largest contributor to the error is still the borders
     *
     * For each combination of sigmaRange and kernel size:
     * Run the slow BF and the fast BF once each, then compute PSNR, print
     * The number of coefficients should be either a large constant, or should vary according to the heuristic
     * Note: This code has been copied from my Honours project codebase.
     */
    std::ofstream csvOut("psnr-" + rangeKrnName + ".csv");

    assert(image.type() == CV_8U); // necessary for using the LUT in the GPU implementation

    int spatialKernelSize;
    double sigmaSpatial;
    double sigmaRange;
    cv::Mat kernel, kernel2D;

    range_krn_t rangeKrn = rangeKrnProvider(rangeKrnName);

    // CSV header
    csvOut << "sigmaRange,spatialKernelSize,numOfCoefs,T,PSNR\n";

    double step = (endSigmaRange - startSigmaRange) / (numOfVals - 1); // step for the sigmaRange step

    cv::Mat slowBFGold = cv::Mat(image.size[0], image.size[1], CV_32F);
    cv::Mat fastBF = cv::Mat(image.size[0], image.size[1], CV_32F);

    cv::Mat imageF32 = cv::Mat(image.size[0], image.size[1], CV_32F);

    for (spatialKernelSize = startKernelSize; spatialKernelSize <= endKernelSize; spatialKernelSize += 2) {
        sigmaSpatial = ((double) spatialKernelSize - 1) / 3.0;// inverse of round(sigmaSpatial * 1.5f) * 2 + 1;
        kernel = cv::getGaussianKernel(spatialKernelSize, sigmaSpatial, CV_32F);
        kernel2D = kernel * kernel.t(); // needed for the slow BF as it doesn't do separable convolution

        for (sigmaRange = startSigmaRange; sigmaRange <= endSigmaRange; sigmaRange += step) {
            int numOfCoefs = (int) ceil(8 * 2 / (6 * sigmaRange)) + 6;

            std::cout << "sigmaRange= " << sigmaRange << " spatialKernelSize= " << spatialKernelSize << "\n";

            // Run the two filters
            BF_approx_gpu(image, fastBF, kernel, sigmaRange, rangeKrn, numOfCoefs, T);
            BF(imageF32, slowBFGold, kernel2D, sigmaRange, rangeKrn);


            auto fastFname = std::format("./psnr-eval/{}-{}-{}-fast.tif", spatialKernelSize, sigmaRange, rangeKrnName);
            auto slowFname = std::format("./psnr-eval/{}-{}-{}-slow.tif", spatialKernelSize, sigmaRange, rangeKrnName);
            auto diffFname = std::format("./psnr-eval/{}-{}-{}-diff.tif", spatialKernelSize, sigmaRange, rangeKrnName);

            double psnr = psnrCompute(slowBFGold, fastBF, spatialKernelSize);

            if (psnr < 65) {
                cv::imwrite(fastFname, fastBF, {cv::ImwriteFlags::IMWRITE_TIFF_COMPRESSION, 34661});
                cv::imwrite(slowFname, slowBFGold, {cv::ImwriteFlags::IMWRITE_TIFF_COMPRESSION, 34661});
                cv::imwrite(diffFname, slowBFGold - fastBF, {cv::ImwriteFlags::IMWRITE_TIFF_COMPRESSION, 34661});
            }
            csvOut << sigmaRange << ","
                   << spatialKernelSize << ","
                   << numOfCoefs << ","
                   << T << ","
                   << psnr << "\n";
            fastBF *= 0;
            slowBFGold *= 0;
        }
    }

    csvOut.close();

}