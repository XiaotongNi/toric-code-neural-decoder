from setuptools import setup
from setuptools import Extension
from Cython.Build import cythonize
import numpy
setup(ext_modules=cythonize('bp.pyx'), include_dirs=[numpy.get_include()])
