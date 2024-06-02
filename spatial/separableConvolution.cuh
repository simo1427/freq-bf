#ifndef CUDA_BF_SEPARABLECONVOLUTION_CUH
#define CUDA_BF_SEPARABLECONVOLUTION_CUH

void setConvolutionKernel(float* h_Krn, int krnSize);

void sepFilter(float* d_Out, float* d_Src, float* d_Buf, int width, int height, int krnSize);

void sepFilterf4(float4* d_Out, float4* d_Src, float4* d_Buf, int width, int height, int krnSize);

#endif //CUDA_BF_SEPARABLECONVOLUTION_CUH
