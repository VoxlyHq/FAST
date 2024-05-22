import platform
from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import numpy
import torch
from torch.utils import cpp_extension

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
            extra_compile_args={'cxx': ['-g'],
                                'nvcc': ['-O2']}
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