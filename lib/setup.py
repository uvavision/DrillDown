from setuptools import setup, Extension
import numpy as np

# To compile and install locally run "python setup.py build_ext --inplace"
# To install library to Python site-packages run "python setup.py build_ext install"

ext_modules = [
    Extension(
        "nms.cpu_nms",
        sources=["nms/cpu_nms.pyx"],
        extra_compile_args=['-Wno-cpp', '-Wno-unused-function', '-std=c99'],
        include_dirs = [np.get_include()]
    ),
    Extension(
        "box_intersections_cpu.bbox",
        sources=["box_intersections_cpu/bbox.pyx"],
        extra_compile_args=['-Wno-cpp', '-Wno-unused-function', '-std=c99'],
        include_dirs = [np.get_include()]
    ),
    Extension(
        'pycocotools._mask',
        sources=['pycocotools/maskApi.c', 'pycocotools/_mask.pyx'],
        extra_compile_args=['-Wno-cpp', '-Wno-unused-function', '-std=c99'],
        include_dirs = [np.get_include(), 'pycocotools'],
    )
]

setup(
    name='external_tools',
    packages=['external_tools'],
    package_dir = {'external_tools': 'external_tools'},
    install_requires=[
        'setuptools>=18.0',
        'cython>=0.27.3',
        'matplotlib>=2.1.0'
    ],
    version='2.0',
    ext_modules= ext_modules
)
