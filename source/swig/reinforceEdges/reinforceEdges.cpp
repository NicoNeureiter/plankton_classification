#include "reinforceEdges.hpp"
#include <vector>
#include <cmath>

using namespace std;


void reinforceEdges(double* img_pointer, int rows, int cols){
    vector<vector<double> > I(rows,vector<double>(cols));
    
    for (int it=0; it<2; ++it){

        for (int i=0;i<rows;++i){
            for (int j=0;j<cols;++j){
                I[i][j] = pow(1-img_pointer[cols*i + j],0.8);
            }
        }
        for (int i=2;i<cols-2;++i){
            for (int j=2;j<rows-2;++j){
                double top, bottom;
                double val = 0;
                for (int k=-2;k<2;++k){
                     top = (I[j-2][i+k] + I[j-2][i+k+1] + I[j-1][i+k] + I[j-1][i+k+1])/4;
                     bottom = (I[j+1][i-k] + I[j+1][i-k-1] + I[j+2][i-k] + I[j+2][i+k-1])/4;
                     val = max(val,top*bottom);
                }
                top = (I[j-1][i-2] + I[j-1][i-1] + I[j][i-2] + I[j][i-1])/4;
                bottom = (I[j][i+1] + I[j][i+2] + I[j+1][i+1] + I[j+1][i+2])/4;
                val = max(val,top*bottom);

                top = (I[j-1][i+2] + I[j-1][i+1] + I[j][i+2] + I[j][i+1])/4;
                bottom = (I[j][i-1] + I[j][i-2] + I[j+1][i-1] + I[j+1][i-2])/4;
                val = max(val,top*bottom);

                double mean = 0;
                for (int k=j-2;k<j+3;++k){
                    for (int l=i-2;l<i+3;++l){
                        mean += I[k][l];
                    }
                }
                mean /= 25;
                val = sqrt(val) + I[j][i] - 0.5*mean;
                val = max(0.0,min(1.0,val));
                img_pointer[cols*j + i] = 1 - val;
            }
        }
    }
}
