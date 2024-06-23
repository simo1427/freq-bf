//
// Created by HP on 10/06/2024.
//

#ifndef CUDA_BF_PERFRUNNER_H
#define CUDA_BF_PERFRUNNER_H

void runPerfMeasureFast(cv::Mat image, int numberOfRuns,
                    double startSigmaRange, double endSigmaRange, int numOfVals,
                    int startKernelSize, int endKernelSize,
                    std::string rangeKrnName, int numberOfCoefficients = 0,
                    float T = 2);

#endif //CUDA_BF_PERFRUNNER_H
