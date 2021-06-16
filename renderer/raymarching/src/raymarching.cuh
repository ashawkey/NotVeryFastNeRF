#ifndef _RAYMARCHING_CUDA_CUH
#define _RAYMARCHING_CUDA_CUH

void raymarching_cuda_wrapper(int H, int W, int D, int N, float near, float far, float xmin, float xmax, float ymin, float ymax, float zmin, float zmax, const float* volume, const float* rays_o, const float* rays_d, float* rgb);
void raymarching_fastnerf_cuda_wrapper(int H, int W, int D, int K, int N, float near, float far, float xmin, float xmax, float ymin, float ymax, float zmin, float zmax, const float* volume, const float* rays_o, const float* rays_d, float* rgb);
void raymarching_fastnerf_sparse_cuda_wrapper(int H, int W, int D, int M, int K, int N, float near, float far, float xmin, float xmax, float ymin, float ymax, float zmin, float zmax, const int* inds, const float* uvws, const float* beta, const float* rays_o, const float* rays_d, float* rgb, bool spherical);

#endif
