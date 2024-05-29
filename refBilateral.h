//
// Created by Simeon Atanasov on 27-4-23.
//

#ifndef FAST_BILATERAL_PERFORMANCE_ANALYSIS_REFBILATERAL_H
#define FAST_BILATERAL_PERFORMANCE_ANALYSIS_REFBILATERAL_H

#define _USE_MATH_DEFINES

#include <opencv2/opencv.hpp>
#include "rangeKernels.h"


void BF(cv::Mat &input, cv::Mat &output, cv::Mat &spatialKernel, double sigmaRange, range_krn_t rangeKrn);

#endif //FAST_BILATERAL_PERFORMANCE_ANALYSIS_REFBILATERAL_H
