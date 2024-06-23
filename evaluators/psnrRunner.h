#ifndef CUDA_BF_PSNRRUNNER_H
#define CUDA_BF_PSNRRUNNER_H

void runPsnrMeasure(cv::Mat image,
                    double startSigmaRange, double endSigmaRange, int numOfVals,
                    int startKernelSize, int endKernelSize,
                    std::string rangeKrnName, int numberOfCoefficients = 0,
                    float T = 2);

double psnrCompute(cv::Mat &ref, cv::Mat &our, int kernelSize, bool writeDiffImg);

#endif //CUDA_BF_PSNRRUNNER_H
