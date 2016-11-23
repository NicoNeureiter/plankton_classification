%module reinforceEdges

%{
#define SWIG_FILE_WITH_INIT
#include "reinforceEdges.hpp"
%}

%include "numpy.i"

%init %{
import_array();
%}

%apply (double* INPLACE_ARRAY2, int DIM1, int DIM2) {(double* img_pointer, int rows, int cols)}

%include "reinforceEdges.hpp"

