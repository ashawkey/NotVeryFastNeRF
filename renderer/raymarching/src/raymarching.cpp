#include "raymarching.hpp"
#include "raymarching.cuh"

#include <iostream>
#include "utils.hpp"

at::Tensor raymarching_cuda(at::Tensor volume, at::Tensor rays_o, at::Tensor rays_d, float near, float far, float xmin, float xmax, float ymin, float ymax, float zmin, float zmax) {
    // input:
    // -- volume: [H, W, D, C], float32, RGBsigma
    // -- rays_o: [N, 3], float32
    // -- rays_d: [N, 3], float32
    // -- near/far: float32, ray range
    // -- x/y/z min/max: float32, volume bounding box

    CHECK_CONTIGUOUS(volume);
    CHECK_CONTIGUOUS(rays_o);
    CHECK_CONTIGUOUS(rays_d);
    
    int H = volume.size(0);
    int W = volume.size(1);
    int D = volume.size(2);

    int N = rays_o.size(0);

    at::Tensor rgb = torch::zeros({N, 3}, at::device(volume.device()).dtype(at::ScalarType::Float));

    //std::cout << rays_o << std::endl;
    //std::cout << rays_d << std::endl;

    raymarching_cuda_wrapper(H, W, D, N, near, far, xmin, xmax, ymin, ymax, zmin, zmax, volume.data_ptr<float>(), rays_o.data_ptr<float>(), rays_d.data_ptr<float>(), rgb.data_ptr<float>());

    return rgb;
}

at::Tensor raymarching_fastnerf_cuda(at::Tensor volume, at::Tensor rays_o, at::Tensor rays_d, float near, float far, float xmin, float xmax, float ymin, float ymax, float zmin, float zmax) {
    // input:
    // -- volume: [H, W, D, C], float32, RGBsigma
    // -- rays_o: [N, 3], float32
    // -- rays_d: [N, 3], float32
    // -- near/far: float32, ray range
    // -- x/y/z min/max: float32, volume bounding box

    CHECK_CONTIGUOUS(volume);
    CHECK_CONTIGUOUS(rays_o);
    CHECK_CONTIGUOUS(rays_d);
    
    int H = volume.size(0);
    int W = volume.size(1);
    int D = volume.size(2);
    int K = (volume.size(3) - 1) / 4; // 8

    int N = rays_o.size(0);

    at::Tensor rgb = torch::zeros({N, 3}, at::device(volume.device()).dtype(at::ScalarType::Float));

    //std::cout << rays_o << std::endl;
    //std::cout << rays_d << std::endl;

    raymarching_fastnerf_cuda_wrapper(H, W, D, K, N, near, far, xmin, xmax, ymin, ymax, zmin, zmax, volume.data_ptr<float>(), rays_o.data_ptr<float>(), rays_d.data_ptr<float>(), rgb.data_ptr<float>());

    return rgb;
}


at::Tensor raymarching_fastnerf_sparse_cuda(at::Tensor inds, at::Tensor uvws, at::Tensor beta, at::Tensor rays_o, at::Tensor rays_d, float near, float far, float xmin, float xmax, float ymin, float ymax, float zmin, float zmax, bool spherical) {
    // input:
    // -- inds: [H, W, D], int32
    // -- uvws: [nnz, 25], float32
    // -- beta: [M, 2M, 8], float32
    // -- rays_o: [N, 3], float32
    // -- rays_d: [N, 3], float32
    // -- near/far: float32, ray range
    // -- x/y/z min/max: float32, volume bounding box

    CHECK_CONTIGUOUS(inds);
    CHECK_CONTIGUOUS(uvws);
    CHECK_CONTIGUOUS(beta);
    CHECK_CONTIGUOUS(rays_o);
    CHECK_CONTIGUOUS(rays_d);
    
    int H = inds.size(0);
    int W = inds.size(1);
    int D = inds.size(2);
    int M = beta.size(0);

    int K = (uvws.size(1) - 1) / 3; // 8

    int N = rays_o.size(0);

    at::Tensor rgb = torch::zeros({N, 3}, at::device(inds.device()).dtype(at::ScalarType::Float));

    //std::cout << rays_o << std::endl;
    //std::cout << rays_d << std::endl;

    raymarching_fastnerf_sparse_cuda_wrapper(H, W, D, M, K, N, near, far, xmin, xmax, ymin, ymax, zmin, zmax, inds.data_ptr<int>(), uvws.data_ptr<float>(), beta.data_ptr<float>(), rays_o.data_ptr<float>(), rays_d.data_ptr<float>(), rgb.data_ptr<float>(), spherical);

    return rgb;
}