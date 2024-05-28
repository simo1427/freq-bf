#ifndef CUDA_BF_FILTERING_CUH
#define CUDA_BF_FILTERING_CUH


void sepFilter(float* d_Out, uint8_t* d_Src, float* d_Krn, int width, int height, int krnSize);

#endif //CUDA_BF_FILTERING_CUH
