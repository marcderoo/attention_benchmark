from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import sys
import os

# Détection automatique du système
if sys.platform == "win32":
    compile_args = ['/O2', '/openmp']
    link_args = []
else:
    compile_args = ['-O3', '-fopenmp']
    link_args = ['-fopenmp']

# Extension Cython avec OpenMP
ext = Extension(
    name="cython_impl",
    sources=["cython_impl.pyx"],
    extra_compile_args=compile_args,
    extra_link_args=link_args,
    include_dirs=[np.get_include()],
)

# Setup classique avec cythonize
setup(
    name="cython_impl",
    ext_modules=cythonize([ext], language_level="3"),
)