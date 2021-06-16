#include <pybind11/pybind11.h>

#include "raymarching.hpp"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("raymarching_cuda", &raymarching_cuda, "ray marching forward (CUDA)");
    m.def("raymarching_fastnerf_cuda", &raymarching_fastnerf_cuda, "fastnerf ray marching forward (CUDA)");
    m.def("raymarching_fastnerf_sparse_cuda", &raymarching_fastnerf_sparse_cuda, "fastnerf sparse ray marching forward (CUDA)");
}
