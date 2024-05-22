import platform
from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import numpy
import torch
from torch.utils import cpp_extension
from os.path import join
import os
from setuptools.command.build_ext import build_ext as _build_ext


# CUDA compilation is adapted from the source
# https://github.com/rmcgibbo/npcuda-example
# CUDA functions for compilation
def find_in_path(name, path):
    """
    Find a file in a search path
    """
    
    #adapted fom http://code.activestate.com/recipes/52224-find-a-file-given-a-search-path/
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

def customize_compiler_for_nvcc(self):
    """
    inject deep into distutils to customize how the dispatch
    to gcc/nvcc works.
    If you subclass UnixCCompiler, it's not trivial to get your subclass
    injected in, and still have the right customizations (i.e.
    distutils.sysconfig.customize_compiler) run on it. So instead of going
    the OO route, I have this. Note, it's kindof like a wierd functional
    subclassing going on.
    """
    
    # tell the compiler it can processes .cu
    self.src_extensions.append('.cu')

    # save references to the default compiler_so and _comple methods
    default_compiler_so = self.compiler_so
    super = self._compile

    # now redefine the _compile method. This gets executed for each
    # object but distutils doesn't have the ability to change compilers
    # based on source extension: we add it.
    def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
        if os.path.splitext(src)[1] == '.cu':
            # Locate CUDA paths
            CUDA = locate_cuda()

            # use the cuda for .cu files
            self.set_executable('compiler_so', CUDA['nvcc'])
            # use only a subset of the extra_postargs, which are 1-1 translated
            # from the extra_compile_args in the Extension class
            postargs = extra_postargs['nvcc']
        else:
            postargs = extra_postargs

        super(obj, src, ext, cc_args, postargs, pp_opts)
        # reset the default compiler_so, which we might have changed for cuda
        self.compiler_so = default_compiler_so

    # inject our redefined _compile method into the class
    self._compile = _compile

# run the customize_compiler
class cuda_build_ext(_build_ext):
    def build_extensions(self):
        customize_compiler_for_nvcc(self.compiler)
        _build_ext.build_extensions(self)

# Function to determine the correct directory for CCL
def determine_ccl_dir():
    os_name = platform.system()
    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except ImportError:
        has_cuda = False

    if os_name == 'Darwin':  # macOS
        return 'ccl_cpu'
    elif has_cuda:
        return 'ccl'  # CUDA GPU available
    else:
        return 'ccl_cpu'  # Default to cpu

# Define include and library directories for PyTorch
torch_include_dirs = torch.utils.cpp_extension.include_paths()
torch_library_dirs = torch.utils.cpp_extension.library_paths()

def get_ccl_extension():
    # Determine the CCL directory
    ccl_dir = determine_ccl_dir()

    print(f"ccl_dir -#{ccl_dir}")
    if ccl_dir == 'ccl':
        print("CUDA is available, compiling CCL with CUDA support")
        from torch.utils.cpp_extension import BuildExtension, CUDAExtension
        return CUDAExtension(
            'ccl_cuda',
            sources=[
                'fast/models/post_processing/ccl/ccl.cpp',
                'fast/models/post_processing/ccl/ccl_cuda.cu',
            ],
            extra_compile_args={'gcc': ['-fPIC'], 'nvcc': ['-Xcompiler', '-fPIC']},
        )
    else:
        return Extension(
            'ccl',
            sources=['fast/models/post_processing/ccl_cpu/ccl.cpp'],
            language='c++',
            include_dirs=[numpy.get_include()] + torch_include_dirs,
            library_dirs=torch_library_dirs,
            libraries=["torch", "torch_cpu", "c10"],  # List necessary PyTorch libraries
            extra_compile_args=['-O3'],
            extra_link_args=[]
        )

# Define extensions
extensions = [
    Extension(
        'pa',
        sources=['fast/models/post_processing/pa/pa.pyx'],
        language='c++',
        include_dirs=[numpy.get_include()],
        library_dirs=[],
        libraries=[],
        extra_compile_args=['-O3'],
        extra_link_args=[]
    ),
    Extension(
        'pse',
        sources=["fast/models/post_processing/pse/pse.pyx"],
        language='c++',
        include_dirs=[numpy.get_include()],
        library_dirs=[],
        libraries=[],
        extra_compile_args=['-O3'],
        extra_link_args=[]
    ),
    get_ccl_extension(),
]



with open('requirements.txt', encoding="utf-8-sig") as f:
    requirements = f.readlines()

def readme():
    with open('README.md', encoding="utf-8-sig") as f:
        README = f.read()
    return README



setup(
    name='fast-ocr',
    packages=['fast'],
    include_package_data=True,
    cmdclass={'build_ext': cuda_build_ext,},
    version='0.0.1',
    install_requires=requirements, #TODO how to compile cython code???
    license='Apache License 2.0',
    description='End-to-End Multi-Lingual Optical Character Recognition (OCR) Solution',
    long_description=readme(),
    long_description_content_type="text/markdown",
    url='https://github.com/voxlyqhq/fast',
    download_url='https://github.com/voxlyhq/fast.git',
    keywords=['FAST - OCR realtime ocr library'],
    classifiers=[
    ],
    ext_modules=extensions,
)