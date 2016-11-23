from distutils.core import setup, Extension

from distutils.core import *
from distutils import sysconfig

import numpy as np
try:
	numpy_include = np.get_include()
except AttributeError:
	numpy_include = np.get_numpy_include()

extension_mod = Extension("_test_kernel_swig", 
						["test_kernel_swig_wrap.cxx", "test_kernel.cpp"]
						)

setup(name = "test_kernel", ext_modules=[extension_mod])