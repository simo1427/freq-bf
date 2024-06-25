# Frequency-based Bilateral Filter on Graphics Cards

This is the implementation of the frequency-based bilateral filter in using the CUDA API.
It decomposes the range kernel of the bilateral filter using Fourier series,
in order to obtain a series of linear filters, which are summed together.
Doing so decreases the time complexity significantly.

Development was done on Windows, Linux support would be provided at a later stage.

## Setup

Prerequisites:
* MSVC
* CUDA 12.3 (higher versions are not supported because of the dependency OpenCV)
* CMake

This project uses `vcpkg` for managing libraries.

## Usage

The project uses command-line flags, use `--help` to obtain a summary

```
Runner for the CUDA implementation of the fast bilateral filter using Fourier series
Usage: .\cuda_bf.exe [OPTIONS] [file]

Positionals:
  file TEXT                   Path to the input image

Options:
  -h,--help                   Print this help message and exit
  --spatial FLOAT             The sigma of the spatial kernel
  --range FLOAT               The sigma of the range kernel
  --rangeKrn TEXT             The range kernel in use (gaussian - default, gaussianUnscaled, huber, tukey, lorentz)
  --kernelSize INT            The size of the spatial kernel; if not provided, it is computed based on the spatial sigma
  --coefs INT                 The number of coefficients to use for the filtering
  --colour                    Specifies whether the filtering is done on a colour image
  --box                       Use a box spatial kernel instead of a Gaussian. The size will be computed from 
  --perf-eval Excludes: --psnr
                              Run a performance evaluation on the bilateral filter. Will run different combination of params a number of times and save a CSV
  --perf-sigma-start FLOAT Needs: --perf-eval
                              Start value for sigmaRange
  --perf-sigma-end FLOAT Needs: --perf-eval
                              End value for sigmaRange
  --perf-sigma-num INT Needs: --perf-eval
                              Number of values between the start and end of sigmaRange
  --perf-kernel-start INT Needs: --perf-eval
                              Start value for the kernel size
  --perf-kernel-end INT Needs: --perf-eval
                              End value for the kernel size
  --perf-runs INT Needs: --perf-eval
                              Number of runs per combination of sigmaRange and kernel size (does not count the first run that is discarded)
  --psnr Excludes: --perf-eval
                              Run an evaluation on the approximation based on PSNRWill run different combination of params, measure PSNR and save a CSV
  --psnr-sigma-start FLOAT Needs: --psnr
                              Start value for sigmaRange
  --psnr-sigma-end FLOAT Needs: --psnr
                              End value for sigmaRange
  --psnr-sigma-num INT Needs: --psnr
                              Number of values between the start and end of sigmaRange
  --psnr-kernel-start INT Needs: --psnr
                              Start value for the kernel size
  --psnr-kernel-end INT Needs: --psnr
                              End value for the kernel size
  --experiment                Used for development purposes

```

There are currently three ways of using this project: applying a filtering operation, 
doing a performance benchmark, and doing a PSNR evaluation.

### Applying a filtering

Applicable params:

`--spatial`, `--range`, `--rangeKrn`, `--kernelSize`, `--coefs`
The kernel size, unless specified, is computed automatically from the value of `--spatial`.
The number of coefficients is also computed automatically unless specified.
Kernel sizes specify the diameter of the kernel, thus need to be _odd_ numbers!

Options for `--rangeKrn` (used in the bachelor thesis):
* gaussian 
* gaussianUnscaled (the Gaussian range kernel, but not divided by $\sigma \sqrt(2 \pi)$ )
* huber
* tukey
* lorentz

Example usage:
`.\cuda_bf.exe --spatial 8 --range 0.1 [filepath]`

`.\cuda_bf.exe --spatial 16.6 --range 0.1 --coefs 10 [filepath]`

### PSNR evaluation

Applicable params:

`--psnr*`, `--coefs`

In order to enter this mode, first specify the `--psnr` flag, then provide the other parameters
You need to make two folders: `./psnr-eval` and `./psnr-eval/outputs`.
Images with a PSNR < 50 dB will be saved with the respective parameters in their filename.
The output is 4 CSV files with an evaluation for all four supported range kernels.

Example usage:

`.\cuda_bf.exe --psnr --psnr-kernel-start 3 --psnr-kernel-end 63 --psnr-sigma-start 0.05 --psnr-sigma-end 1 --psnr-sigma-num 20 [filename]` - runs a PSNR evaluation by doing a grid search over the parameters within the specified bounds.
The number of coefficients is determined automatically.

`.\cuda_bf.exe --coefs 10 --psnr --psnr-kernel-start 3 --psnr-kernel-end 63 --psnr-sigma-start 0.05 --psnr-sigma-end 1 --psnr-sigma-num 20 [filename]` - similarly to the command above, does a grid search over the parameter space, with
the difference that the number of coefficients is _fixed_ to 10 for all tested parameter combinations.

### Performance evaluation

Applicable params:

`--perf*`

Similarly to the PSNR evaluation, create a folder `./perf-eval/`. It will contain a CSV file with runtimes.
Apart from the specified number of runs, there is also a warmup run which should not be taken into account when calculating
the mean and standard deviation of execution times.

**Important note:** The runtime performance of the filter does not depend on the value of sigmaRange directly, 
but rather on the number of coefficients. The current version of the proejct uses the value `--perf-sigma-num` as means to
provide the upper bound of the value for the number of coefficients. Therefore, benchmarks will be tested for 
coefficient values ranging from 1 to the number specified in `--perf-sigma-num`.

Example usage:

`.\cuda_bf.exe --perf-eval --perf-sigma-start 0.1 --perf-sigma-num 28 --perf-kernel-start 3 --perf-kernel-end 127 [filename]`

### Unsupported flags

`--colour` - the project only filters images in grayscale at the current stage.

`--box` - a box filter is not supported yet. 

`--experiment` - can be used as a placeholder for quick experiments, a developer environment is needed to use it properly.