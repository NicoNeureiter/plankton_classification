#include "distToTree.hpp"
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

double dot(Point p0, Point p1){
    return p0.x*p1.x + p0.y*p1.y;
}


double pointSegmentDistance(Point p,
                            Point s0,
                            Point s1){
    Point s = s1.sub(s0);
    p = p.sub(s0);
    double s_len = s.norm();
    s.x /= s_len;
    s.y /= s_len;
    Point s_orth(-s.y,s.x);
    double t = dot(p,s);
    double u = dot(p,s_orth);
    if (t < 0) {
        return u*u + t*t;
    }else{ 
        if (t > s_len)
            return u*u + (t-s_len)*(t-s_len);
        else
            return u*u;
    }
}

void distToTree(double* centroids_pointer, int k2, double* dist_array_pointer, int rows, int cols){
    double diag = rows*rows + cols*cols;
    int k = (int) k2/2;
    vector<Point> centroids(k);
    for (int c=0; c<k; ++c){
        centroids[c] = Point(centroids_pointer[2*c],centroids_pointer[2*c+1]);
    }

    vector<vector<double> > dist_array(rows,vector<double>(cols));
 
    for (int i=0;i<rows;++i){
        for (int j=0;j<cols;++j){
            Point p(i,j);
            double dist = 1000000000;
            for (int c=0; c<k-1; ++c){
                Point s0 = centroids[c];
                Point s1 = centroids[c+1];
                dist = min(dist, pointSegmentDistance(p, s0, s1) );
            }
            dist = sqrt(dist/diag);
            dist_array[i][j] = dist;
        }
    }

    for (int i=0;i<rows;++i){
        for (int j=0;j<cols;++j){
            dist_array_pointer[cols*i + j] = dist_array[i][j];
        }
    }

}
