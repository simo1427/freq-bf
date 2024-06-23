#include <gsl/gsl_integration.h>
#include <cmath>
#include <vector>
#include <string>
#include <iostream>


#ifndef CUDA_BF_RANGEKERNELS_H
#define CUDA_BF_RANGEKERNELS_H

#define PROVIDE_KERNEL(KRN_NAME) if(name == #KRN_NAME) return KRN_NAME;


typedef struct integrand_params {
    int k;
    double sigma;
    double T;
    double normConst;
    double (*rangeKrn)(double, double);
} integrand_params_t;

typedef double (*range_krn_t)(double, double); // define the type of function pointer to a range kernel

const int GSL_INTEGRATION_WORKSPACE_LIMIT = 1000;
const double SQRT_2_PI = sqrt(2 * M_PI);

double gaussian(double x, double sigma);

double gaussianUnscaled(double x, double sigma);

double huber(double x, double sigma);

double tukey(double x, double sigma);

double lorentz(double x, double sigma);

range_krn_t
rangeKrnProvider(std::string& name);

std::vector<double>
getFourierCoefficients(double sigmaRange, double T, int numberOfCoefficients, range_krn_t rangeKrn);



#endif //CUDA_BF_RANGEKERNELS_H
