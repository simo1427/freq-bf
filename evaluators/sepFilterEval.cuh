#ifndef CUDA_BF_SEPFILTEREVAL_CUH
#define CUDA_BF_SEPFILTEREVAL_CUH

void sepConvEval(cv::Mat &input, cv::Mat &output, cv::Mat &spatialKernel, int numOfRuns);

#endif //CUDA_BF_SEPFILTEREVAL_CUH
