from setuptools import setup, find_packages, Command
from setuptools.command.build_ext import build_ext as _build_ext
import subprocess
import os

class CustomBuildExtCommand(_build_ext):
    """A custom command to run setup.py files in subdirectories."""
    description = 'run setup.py in subdirectories'
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    """Custom build command that runs setup.py in subdirectories."""
    def run(self):
        print("building custom modules for fast-ocr")
        # Check for OS and CUDA GPU
        os_name = platform.system()
        has_cuda = torch.cuda.is_available()

        if os_name == 'Darwin':  # macOS
            ccl_dir = 'ccl_cpu' #TODO in future add metal
        elif has_cuda:
            ccl_dir = 'ccl'  # CUDA GPU available
        else:
            ccl_dir = 'ccl_cpu'  # Default to your_ccl_dir

        subdir_setup_scripts = [
            "./models/post_processing/pa/setup.py",
            "./models/post_processing/pse/setup.py",
            f"./models/post_processing/{ccl_dir}/setup.py",
        ]

        for script in subdir_setup_scripts:
            subprocess.check_call([os.sys.executable, script, "build_ext", "--inplace"])

        # Run the original build_ext command
        super().run()


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
    cmdclass={
         'build_ext': CustomBuildExtCommand,
    }
)