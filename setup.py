import platform
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy
import torch
from torch.utils import cpp_extension
import os
from os.path import join
from setuptools.command.build_ext import build_ext as _build_ext


def find_in_path(name, path):
    """Find a file in a search path"""
    for dir in path.split(os.pathsep):
        binpath = join(dir, name)
        if os.path.exists(binpath):
            return os.path.abspath(binpath)
    return None

def locate_cuda():
    """Locate the CUDA environment on the system"""
    if 'CUDA_HOME' in os.environ:
        home = os.environ['CUDA_HOME']
    elif 'CUDA_PATH' in os.environ:
        home = os.environ['CUDA_PATH']
    else:
        home = '/usr/local/cuda'

    cudaconfig = {
        'home': home,
        'nvcc': os.path.join(home, 'bin', 'nvcc'),
        'include': os.path.join(home, 'include'),
        'lib64': os.path.join(home, 'lib64'),
    }

    for k, v in cudaconfig.items():
        if not os.path.exists(v):
            raise EnvironmentError(f'The CUDA {k} path could not be located in {v}')

    return cudaconfig

class custom_build_ext(_build_ext):
    def build_extensions(self):
        self.compiler.src_extensions.append('.cu')
        original_compile = self.compiler._compile

        def new_compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
            if src.endswith('.cu'):
                self.set_executables(compiler_so='nvcc')
                postargs = ['-Xcompiler', '-fPIC'] + (extra_postargs or [])
                postargs = extra_postargs['nvcc']
            else:
                postargs = extra_postargs or []
                postargs = extra_postargs['gcc']
            original_compile(obj, src, ext, cc_args, postargs, pp_opts)

        self.compiler._compile = new_compile
        _build_ext.build_extensions(self)

def determine_ccl_dir():
    os_name = platform.system()
    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except ImportError:
        has_cuda = False

    if os_name == 'Darwin':
        return 'ccl_cpu'
    elif has_cuda:
        return 'ccl'
    else:
        return 'ccl_cpu'

torch_include_dirs = torch.utils.cpp_extension.include_paths()
torch_library_dirs = torch.utils.cpp_extension.library_paths()

def get_ccl_extension():
    ccl_dir = determine_ccl_dir()

    if ccl_dir == 'ccl':
        from torch.utils.cpp_extension import CUDAExtension
        return CUDAExtension(
            'ccl_cuda',
            sources=[
                'fast/models/post_processing/ccl/ccl.cpp',
                'fast/models/post_processing/ccl/ccl_cuda.cu',
            ],
            include_dirs=[numpy.get_include()] + torch_include_dirs,
            library_dirs=torch_library_dirs,
            libraries=["torch", "c10", "torch_cuda"],  # Ensure the libraries are correct
            extra_compile_args={'gcc': ['-fPIC'], 'nvcc': ['-Xcompiler', '-fPIC']},
        )
    else:
        return Extension(
            'ccl',
            sources=['fast/models/post_processing/ccl_cpu/ccl.cpp'],
            language='c++',
            include_dirs=[numpy.get_include()] + torch_include_dirs,
            library_dirs=torch_library_dirs,
            libraries=["torch", "torch_cpu", "c10"],
            extra_compile_args={'gcc': ['-O3']},
        )

extensions = [
    Extension(
        'pa',
        sources=['fast/models/post_processing/pa/pa.pyx'],
        language='c++',
        include_dirs=[numpy.get_include()],
        extra_compile_args=['-O3'],
    ),
    Extension(
        'pse',
        sources=["fast/models/post_processing/pse/pse.pyx"],
        language='c++',
        include_dirs=[numpy.get_include()],
        extra_compile_args=['-O3'],
    ),
    get_ccl_extension(),
]

with open('requirements.txt', encoding="utf-8-sig") as f:
    requirements = f.readlines()

def readme():
    with open('README.md', encoding="utf-8-sig") as f:
        return f.read()

setup(
    name='fast-ocr',
    packages=['fast'],
    include_package_data=True,
    cmdclass={'build_ext': custom_build_ext},
    version='0.0.1',
    install_requires=requirements,
    license='Apache License 2.0',
    description='End-to-End Multi-Lingual Optical Character Recognition (OCR) Solution',
    long_description=readme(),
    long_description_content_type="text/markdown",
    url='https://github.com/voxlyqhq/fast',
    download_url='https://github.com/voxlyhq/fast.git',
    keywords=['FAST - OCR realtime ocr library'],
    classifiers=[],
    ext_modules=cythonize(extensions),
)
