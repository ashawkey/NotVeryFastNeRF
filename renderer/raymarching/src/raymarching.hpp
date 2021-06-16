#ifndef _RAYMARCHING_CUDA_H
#define _RAYMARCHING_CUDA_H

#include <torch/extension.h>

at::Tensor raymarching_cuda(at::Tensor volume, at::Tensor rays_o, at::Tensor rays_d, float near, float far, float xmin, float xmax, float ymin, float ymax, float zmin, float zmax);
at::Tensor raymarching_fastnerf_cuda(at::Tensor volume, at::Tensor rays_o, at::Tensor rays_d, float near, float far, float xmin, float xmax, float ymin, float ymax, float zmin, float zmax);
at::Tensor raymarching_fastnerf_sparse_cuda(at::Tensor inds, at::Tensor uvws, at::Tensor beta, at::Tensor rays_o, at::Tensor rays_d, float near, float far, float xmin, float xmax, float ymin, float ymax, float zmin, float zmax, bool spherical);

#endif
