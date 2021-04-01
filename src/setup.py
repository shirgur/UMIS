from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='morphpool_cuda',
    ext_modules=[
        CUDAExtension('morphpool_cuda', [
            'morphpool_cuda.cpp',
            'morphpool_cuda_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })