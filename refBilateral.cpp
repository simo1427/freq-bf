//
// Created by Simeon Atanasov on 27-4-23.
//

#include "refBilateral.h"

#include <cmath>


void BF_grs(cv::Mat &input, cv::Mat &output, cv::Mat &spatialKernel, double sigmaRange, range_krn_t rangeKrn) {

    cv::MatSize imgSize = input.size;
    cv::MatSize krnSize = spatialKernel.size;

    int ksz = krnSize[0];
    int hks = krnSize[0] / 2;

    const double SQRT_2_PI = sqrt(2 * M_PI);

    cv::Mat padded = cv::Mat(imgSize[0] + 2 * hks, imgSize[1] + 2 * hks, input.type());
    cv::MatSize paddedSize = padded.size;

    cv::copyMakeBorder(input, padded, hks, hks, hks, hks, cv::BORDER_REFLECT);
//    x = (center_of_filtering - img[i - y + hks, j - x + hks])
//    calculation_terms = spatial_kernel[y, x] *
//            (np.exp((-0.5 * x * x)/ (sigma_range_2))) / (sigma_range * np.sqrt(2 * np.pi))

    for(int i = hks; i < paddedSize[0] - hks; i++) {

        float* paddedPtr = padded.ptr<float>(i);
        float* outputPtr = output.ptr<float>(i - hks);
        for (int j = hks; j < paddedSize[1] - hks; j++) {

            float tmp_sum = 0.0;
            float tmp_Wp = 0.0;

            float center_of_filtering = paddedPtr[j];

            for(int krnI = 0; krnI < ksz; krnI++) {
                float* paddedPtr2 = padded.ptr<float>(i - hks + krnI);
                float* spatialKrnPtr = spatialKernel.ptr<float>(krnI);
                for (int krnJ = 0; krnJ < ksz; krnJ++) {
                    float x = (center_of_filtering - paddedPtr2[j - hks + krnJ]);
                    float spatialKrnWeight = spatialKrnPtr[krnJ];
                    float calculation_terms = spatialKrnWeight * rangeKrn(x, sigmaRange);
                    tmp_sum += paddedPtr2[j - hks + krnJ] *  calculation_terms;
                    tmp_Wp += calculation_terms;
                }
            }
            outputPtr[j - hks] = tmp_sum / tmp_Wp;
        }
    }
}

void BF_col(cv::Mat &input, cv::Mat &output, cv::Mat &spatialKernel, double sigmaRange, range_krn_t rangeKrn) {

    cv::MatSize imgSize = input.size;
    cv::MatSize krnSize = spatialKernel.size;

    int ksz = krnSize[0];
    int hks = krnSize[0] / 2;

    const double SQRT_2_PI = sqrt(2 * M_PI);

    cv::Mat padded = cv::Mat(imgSize[0] + 2 * hks, imgSize[1] + 2 * hks, input.type());
    cv::MatSize paddedSize = padded.size;

    cv::copyMakeBorder(input, padded, hks, hks, hks, hks, cv::BORDER_REFLECT);

    for(int i = hks; i < paddedSize[0] - hks; i++) {

        cv::Vec3f* paddedPtr = padded.ptr<cv::Vec3f>(i);
        cv::Vec3f* outputPtr = output.ptr<cv::Vec3f>(i - hks);
        for (int j = hks; j < paddedSize[1] - hks; j++) {

            cv::Vec3f tmp_sum = 0.0;
            float tmp_Wp = 0.0;

            cv::Vec3f center_of_filtering = paddedPtr[j];

            for(int krnI = 0; krnI < ksz; krnI++) {
                cv::Vec3f* paddedPtr2 = padded.ptr<cv::Vec3f>(i - hks + krnI);
                float* spatialKrnPtr = spatialKernel.ptr<float>(krnI);
                for (int krnJ = 0; krnJ < ksz; krnJ++) {
                    float x = cv::norm(center_of_filtering - paddedPtr2[j - hks + krnJ]);
                    float spatialKrnWeight = spatialKrnPtr[krnJ]; // Gaussian is separable, so this should be valid
                    float calculation_terms = spatialKrnWeight * rangeKrn(x, sigmaRange);
                    tmp_sum += paddedPtr2[j - hks + krnJ] *  calculation_terms;
                    tmp_Wp += calculation_terms;
                }
            }
            outputPtr[j - hks] = tmp_sum / tmp_Wp;
        }
    }
}

void
BF(cv::Mat &input, cv::Mat &output, cv::Mat &spatialKernel, double sigmaRange, range_krn_t rangeKrn) {
    if (input.type() != output.type())
    {
        std::cerr
                << "Types of the input and output images differ! input: "
                << input.type()
                << " output: "
                << output.type()
                << "\n";
        return;
    }

    if (input.type() != CV_32F && input.type() != CV_32FC3)
    {
        std::cerr
                << "Type "
                << input.type()
                << " is not supported, only CV_32F ("
                << CV_32F
                << ") and CV_32FC3 ("
                << CV_32FC3
                << ")\n";
        return;
    }

    cv::MatSize inp {input.size};
    cv::MatSize out {output.size};

    if (input.size() != output.size()) {
        std::cerr
                << "Input image of size "
                << input.size()
                << " differs from the output buffer of size "
                << output.size()
                << ", they need to be equal\n";
        return;
    }

    if (input.type() == CV_32F)
    {
        BF_grs(input, output, spatialKernel, sigmaRange, rangeKrn);
        return;
    }
    // the only other possible type is CV_32FC3

    BF_col(input, output, spatialKernel, sigmaRange, rangeKrn);
}