#ifndef CUDA_BF_PSNRRUNNER_H
#define CUDA_BF_PSNRRUNNER_H

//void runPsnrMeasure();

double psnrCompute(cv::Mat &ref, cv::Mat &our, int kernelSize, bool writeDiffImg);

#endif //CUDA_BF_PSNRRUNNER_H
