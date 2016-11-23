from distutils.core import setup, Extension

from distutils.core import *
from distutils import sysconfig

import numpy as np
try:
	numpy_include = np.get_include()
except AttributeError:
	numpy_include = np.get_numpy_include()

extension_mod = Extension("_reinforceEdges", 
						["reinforceEdges_wrap.cxx", "reinforceEdges.cpp"]
						)

setup(name = "reinforceEdges", ext_modules=[extension_mod])
