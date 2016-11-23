#include "test_kernel.hpp"
#include <vector>
#include <cmath>

using namespace std;

struct Point {
    double x;
    double y;

    Point(double x = 0.0, double y=0.0) : x(x), y(y) {}

    Point add(Point b){
     return Point(x + b.x, y + b.y);
    }

    Point sub(Point b){
     return Point(x - b.x, y - b.y);
    }

    double norm(){
        return sqrt(x*x + y*y);
    }
};

double sqr(double x){
    return x*x;
}

double squared_dist(Point p1, Point p2){
    return sqr(p1.x - p2.x) + sqr(p1.y - p2.y);
}

void som(double* img_pointer, int rows, int cols, long* res, int k2){
    int k = (int) k2/2;
    int iterations = 30;
    double diag = sqrt(sqr(rows) + sqr(cols));
    vector<vector<double> > img(rows,vector<double>(cols));
    for (int i=0;i<rows;++i){
        for (int j=0;j<cols;++j){
            img[i][j] = 1 - img_pointer[cols*i + j];
        }
    }
    vector<Point> centroids(k,Point(0,0));
    for (int i=0;i<k;++i){
        centroids[i].x = round(1.0*i/k * rows/2 + rows/4);
        centroids[i].y = round(1.0*i/k * cols/2 + cols/4);
    }

    for (int it=0;it<iterations; ++it){
        vector<Point> old_centroids = centroids;
        centroids = vector<Point>(k,Point(0,0));
        vector<double> norm_const(k,0);
        vector<double> neighbourhood(k-1,0);
        for (int c=0; c<k-1; ++c){
            double sqr_dist = squared_dist(old_centroids[c],old_centroids[c+1]);
            //double sigma = 1/diag;
            neighbourhood[c] = 0.7*exp(- 2*sqr_dist / diag);
        }

        for (int x=0; x<rows; ++x){
            for (int y=0; y<cols; ++y){
                double dist;
                int c_opt = 0;
                double min_dist = 1000000;
                for (int c=0; c<k; ++c){ 
                    dist = sqr(x - old_centroids[c].x) + sqr(y - old_centroids[c].y);
                    if (dist < min_dist){
                        c_opt = c;
                        min_dist = dist;
                    }
                }
                centroids[c_opt].x += img[x][y]*x;
                centroids[c_opt].y += img[x][y]*y;
                norm_const[c_opt] += img[x][y];
                if (c_opt > 0) {
                    centroids[c_opt-1].x += img[x][y]*x * neighbourhood[c_opt-1];
                    centroids[c_opt-1].y += img[x][y]*y * neighbourhood[c_opt-1];
                    norm_const[c_opt-1] += img[x][y] * neighbourhood[c_opt-1];
                }
                if (c_opt < k-1) {
                    centroids[c_opt+1].x += img[x][y]*x * neighbourhood[c_opt];
                    centroids[c_opt+1].y += img[x][y]*y * neighbourhood[c_opt];
                    norm_const[c_opt+1] += img[x][y] * neighbourhood[c_opt];
                }
            }
        }
        for (int c=0; c<k; ++c){
            if (norm_const[c] > 0){
                centroids[c].x = round(centroids[c].x/norm_const[c]);
                centroids[c].y = round(centroids[c].y/norm_const[c]);
            } else {
                centroids[c].x = old_centroids[c].x;
                centroids[c].y = old_centroids[c].y;
            }
        }
    }

    for (int c=0;c<k;++c){
        res[2*c] = (int) centroids[c].x;
        res[2*c + 1] = (int) centroids[c].y;
    }
}
