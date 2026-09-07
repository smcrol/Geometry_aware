from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='grouping',
    ext_modules=[
        CUDAExtension('grouping_cuda', [
            'src/grouping_api.cpp',
            'src/grouping.cpp',
            'src/grouping_gpu.cu',
        ],
        extra_compile_args={'cxx': ['-std=c++14','-g'],
                            'nvcc': ['-O2','-std=c++14', '-arch=sm_60']})
    ],
    cmdclass={'build_ext': BuildExtension}
)
