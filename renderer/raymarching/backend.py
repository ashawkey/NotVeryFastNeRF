import os
from torch.utils.cpp_extension import load

_src_path = os.path.dirname(os.path.abspath(__file__))

_backend = load(name='_raymarching_backend',
                extra_cflags=['-g', '-O3', '-fopenmp', '-lgomp'],
                extra_cuda_cflags=['-arch=compute_30', '-O3'],
                sources=[os.path.join(_src_path, 'src', f) for f in [
                    'raymarching.cpp',
                    'raymarching.cu',
                    'bindings.cpp',
                ]]
                )

__all__ = ['_backend']