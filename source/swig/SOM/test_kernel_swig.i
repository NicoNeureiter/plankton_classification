%module test_kernel_swig

%{
#define SWIG_FILE_WITH_INIT
#include "test_kernel.hpp"
%}

%include "numpy.i"

%init %{
import_array();
%}

%apply (double* IN_ARRAY2, int DIM1, int DIM2) {(double* img_pointer, int rows, int cols)}

%apply (long* ARGOUT_ARRAY1, int DIM1) {(long* res, int k2)}

%include "test_kernel.hpp"

