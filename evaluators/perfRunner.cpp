//
// Created by HP on 10/06/2024.
//
#include <opencv2/opencv.hpp>
#include <fstream>

#include "perfRunner.h"
#include "rangeKernels.h"
#include "fastBilateral.cuh"

void runPerfMeasureFast(cv::Mat image, int numberOfRuns,
                        double startSigmaRange, double endSigmaRange, int numOfVals,
                        int startKernelSize, int endKernelSize,
                        std::string rangeKrnName, int numberOfCoefficients,
                        float T) {
    double step = (endSigmaRange - startSigmaRange) / (numOfVals - 1); // step for the sigmaRange step

    if (image.type() != CV_8U) { // necessary for using the LUT in the GPU implementation
        std::cerr << "The image provided to the performance evaluator must be of type CV_8U\n";
        exit(EXIT_FAILURE);
    }

    int spatialKernelSize;
    double sigmaSpatial;
    double sigmaRange;
    cv::Mat kernel, kernel2D;

    range_krn_t rangeKrn = rangeKrnProvider(rangeKrnName);

    std::ofstream csvOutCV("./perf-eval/bf-perf-eval-slow.csv");
    std::ofstream csvOutFast("./perf-eval/bf-perf-eval-fast.csv");
    // Create the CSV's header
    csvOutCV << "sigmaRange, numOfCoefs, spatialKernelSize, warmup";
    csvOutFast << "sigmaRange, numOfCoefs, spatialKernelSize, warmup";

    for (int i = 0; i < numberOfRuns; i++) {
        csvOutCV << ",time" << i;
        csvOutFast << ",time" << i;
    }

    csvOutCV << "\n";
    csvOutFast << "\n";

    cv::Mat slowBFRef = cv::Mat(image.size[0], image.size[1], CV_32F);
    cv::Mat fastBF = cv::Mat(image.size[0], image.size[1], CV_32F);


//    std::cout << "Evaluating CV BF\n";
//    // Slow BF eval
//    for (sigmaRange = startSigmaRange;  sigmaRange <= endSigmaRange; sigmaRange += step)
//    {
//        for (spatialKernelSize = startKernelSize; spatialKernelSize <= endKernelSize; spatialKernelSize += 2)
//        {
//
//            sigmaSpatial = ((double) spatialKernelSize - 1) / 3.0;// inverse of round(sigmaSpatial * 1.5f) * 2 + 1;
//            std::cout << "sigmaRange= " << sigmaRange << " spatialKernelSize= " << spatialKernelSize <<" ";
//            csvOutCV << sigmaRange << "," << spatialKernelSize;
//            // discard the first run
//            long long cvMillis = cvBFRunner(slowBFRef, img32, spatialKernelSize, sigmaRange, sigmaSpatial, isColour);
//            csvOutCV << "," << cvMillis;
//
//            for (int i = 0; i < numberOfRuns; i++)
//            {
//                // runner for CV BF
//                cvMillis = cvBFRunner(slowBFRef, img32, spatialKernelSize, sigmaRange, sigmaSpatial, isColour);
//                csvOutCV << "," << cvMillis;
//                std::cout << ".";
//            }
//            csvOutCV << "\n";
//            std::cout << "\n";
//        }
//    }

    std::cout << "Evaluating fast BF\n";
    // Fast BF eval
    for (int i = 1; i <= numOfVals; i++) {
        sigmaRange = startSigmaRange; //+ i * step;
        /*
         * The number of coefficients is subject to tuning, hence it is additionally calculated here
         * and passed to subsequent methods
         */
        //int numOfCoefs = numberOfCoefficients == 0 ? (int) ceil(4 * T / (6 * sigmaRange)) + 1 : numberOfCoefficients;
        int numOfCoefs = i;
        std::cout << "Number of coefficients: " << numOfCoefs << "\n";
        for (spatialKernelSize = startKernelSize; spatialKernelSize <= endKernelSize; spatialKernelSize += 2) {
            sigmaSpatial = ((double) spatialKernelSize - 1) / 3.0;// inverse of round(sigmaSpatial * 1.5f) * 2 + 1;
            cv::Mat kernel{cv::getGaussianKernel(spatialKernelSize, sigmaSpatial, CV_32F)};
            std::cout << "numOfCoefs= " << numOfCoefs << " spatialKernelSize= " << spatialKernelSize
                      << " "; // << " sigmaRange= " << sigmaSpatial
            csvOutFast << sigmaRange << "," << numOfCoefs << "," << spatialKernelSize;
            // discard the first run
            std::vector<float> runs = BF_approx_gpu_perf(image, fastBF, kernel, sigmaRange, rangeKrn, numOfCoefs, T, numberOfRuns + 1);


            for (int run = 0; run < runs.size(); run++) {
                // runner for CV BF
                csvOutFast << "," << runs[run];
            }
            csvOutFast << "\n";
            std::cout << "\n";
        }
    }


    csvOutCV.close();
    csvOutFast.close();
    return;
}
