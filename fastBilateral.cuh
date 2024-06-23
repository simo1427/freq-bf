//
// Created by HP on 31/05/2024.
//

#include <opencv2/opencv.hpp>
#include "rangeKernels.h"

#ifndef CUDA_BF_FASTBILATERAL_CUH
#define CUDA_BF_FASTBILATERAL_CUH


#define MAX_COEFS_NUM 28 // TODO: check for a better value, possibly derived from the amount of VRAM

void BF_approx_gpu(cv::Mat &input, cv::Mat &output, cv::Mat &spatialKernel, double sigmaRange, range_krn_t rangeKrn,
                   int numberOfCoefficients, float T);

std::vector<float>
BF_approx_gpu_perf(cv::Mat &input, cv::Mat &output, cv::Mat &spatialKernel, double sigmaRange, range_krn_t rangeKrn,
                   int numberOfCoefficients, float T, int numOfRuns);

#endif //CUDA_BF_FASTBILATERAL_CUH
