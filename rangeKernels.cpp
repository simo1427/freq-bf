#include "rangeKernels.h"

double
gaussian(double x, double sigma) {
    double f = exp(-x * x / (2 * sigma * sigma)) / (sigma * SQRT_2_PI);
    return f;
}

double
gaussianUnscaled(double x, double sigma) {
    double f = exp(-x * x / (2 * sigma * sigma));
    return f;
}

double
huber(double x, double sigma) {
    if (x <= sigma && x >= -sigma)
        return 1 / sigma;
    else
        return 1 / (x > 0 ? x : -x);
}

double
tukey(double x, double sigma) {
    sigma *= sqrt(5); // to ensure consistent scale, Durand paper
    if (x <= sigma && x >= -sigma){
        double xSigma = x / sigma;
        double toSquare = 1 - (xSigma * xSigma);
        return toSquare * toSquare / 2;
    }
    else
        return 0;
}

double
lorentz(double x, double sigma) {
    sigma /= sqrt(2); // to ensure consistent scale, Durand paper
    return 2 / (2 + (x * x / (sigma * sigma)));
}


double
fourierCoef(double x, void* params) {
    auto * integrandParams = (integrand_params_t * ) params;

    int k = (integrandParams->k);
    double T = (integrandParams->T);
    double sigma = (integrandParams->sigma);
    double normConst = (integrandParams->normConst);
    range_krn_t rangeKrn = (integrandParams -> rangeKrn);


    double f = 2 * rangeKrn(x, sigma) * cos(2 * M_PI * k * x / T) ;
    return f;
}

range_krn_t
rangeKrnProvider(std::string& name) {
    PROVIDE_KERNEL(huber)
    PROVIDE_KERNEL(tukey)
    PROVIDE_KERNEL(lorentz)
    PROVIDE_KERNEL(gaussianUnscaled)

    return gaussian; // the default
}

std::vector<double>
getFourierCoefficients(double sigmaRange, double T, int numberOfCoefficients, range_krn_t rangeKrn) {

    std::vector<double> ret = std::vector<double>();

    gsl_integration_workspace* workspace = gsl_integration_workspace_alloc(GSL_INTEGRATION_WORKSPACE_LIMIT);

    double result, error;

    double normConst = (T * sigmaRange * SQRT_2_PI);

    for(int k = 0; k < numberOfCoefficients; k++) {

        auto params = integrand_params_t {k, sigmaRange, T, normConst, rangeKrn};

        gsl_function F;
        F.function = fourierCoef;
        F.params = &params;

        gsl_integration_qags (&F, -T/2, T/2, 1e-7, 1e-7,
                              GSL_INTEGRATION_WORKSPACE_LIMIT, workspace, &result, &error);

        ret.push_back(result);
#ifdef CUTOFF_COEFS
        if (std::abs(result) < 5e-5) {
            std::cerr << "Threshold reached, number of coefs is "
                << ret.size()
                << "instead of "
                << numberOfCoefficients
                << "\n";
            break;
        }
#endif
//        std::cout << result << " " << error << "\n";
    }
    gsl_integration_workspace_free(workspace);
    ret[0] /= 2;
    return ret;
}