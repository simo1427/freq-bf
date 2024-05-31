#ifndef CUDA_BF_SEPARABLECONVOLUTION_CUH
#define CUDA_BF_SEPARABLECONVOLUTION_CUH

void setConvolutionKernel(float* h_Krn, int krnSize);

void sepFilter(float* d_Out, float* d_Src, float* d_Buf, int width, int height, int krnSize, size_t pitch);

#endif //CUDA_BF_SEPARABLECONVOLUTION_CUH
