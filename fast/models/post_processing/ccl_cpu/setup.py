from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='ccl_cpu',
    ext_modules=[
        CppExtension('ccl_cpu', [
            'ccl.cpp',  # Your C++ source file
            'ccl_cpu.cpp',  # Your ported CPU version of the .cu file
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
