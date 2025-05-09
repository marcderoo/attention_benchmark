# setup.py
import numpy
from setuptools import setup, Extension
from Cython.Build import cythonize

ext = Extension(
    name="attention.cython_impl",
    sources=["attention/cython_impl.pyx"], 
    include_dirs=[numpy.get_include()],
    language="c++",
    define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    extra_compile_args=["/O2", "/openmp", "/arch:AVX2"],
    extra_link_args=["-openmp"],
)

setup(
    name="attention_benchmark",
    ext_modules=cythonize(ext, language_level="3")
)
