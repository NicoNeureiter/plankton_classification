%module distToTree

%{
#define SWIG_FILE_WITH_INIT
#include "distToTree.hpp"
%}

%include "numpy.i"

%init %{
import_array();
%}

%apply (double* IN_ARRAY1, int DIM1) {(double* centroids_pointer, int k2)}
%apply (double* INPLACE_ARRAY2, int DIM1, int DIM2) {(double* dist_array_pointer, int rows, int cols)}

%include "distToTree.hpp"

