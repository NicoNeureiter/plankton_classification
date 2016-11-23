from distutils.core import setup, Extension

from distutils.core import *
from distutils import sysconfig

import numpy as np
try:
	numpy_include = np.get_include()
except AttributeError:
	numpy_include = np.get_numpy_include()

extension_mod = Extension("_distToTree", ["distToTree_wrap.cxx", "distToTree.cpp"])

setup(name = "distToTree", ext_modules=[extension_mod])
