#include <stdio.h>
#include <math.h>

//#define char16_t uint16_T

// #include "mex.h"
#include "EnfConn.h"
#include <vector>
#include <algorithm>
#include <random>
#include <unordered_set>
#include <ctime>
//g++ SLICSP_new_match_v0929.cpp -fPIC -shared -o SLICSP_v0929.so
//-------------------------------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------------------------------

extern "C" {
    void SLICSP(double *L, double *A, double *B, int WIDTH, int HEIGHT,
                int K, double M, double prob, double *labels, float *X_mask);
}

void SLICSP(double *L, double *A, double *B, int WIDTH, int HEIGHT,
            int K, double M, double prob, double *labels, float *X_mask)
{
    // 1. 初始化中心点
    int N = WIDTH * HEIGHT;
    int S = int(sqrt(double(N) / K));
    std::vector<double> CX, CY, CL, CA, CB;
    for (int h = S / 2; h < HEIGHT; h += S) {
        for (int w = S / 2; w < WIDTH; w += S) {
            // 2. 3x3邻域内找梯度最小点
            double min_grad = 1e20;
            int min_h = h, min_w = w;
            for (int dh = -1; dh <= 1; ++dh) {
                for (int dw = -1; dw <= 1; ++dw) {
                    int _h = h + dh, _w = w + dw;
                    if (_h < 0 || _h >= HEIGHT - 1 || _w < 0 || _w >= WIDTH - 1) continue;
                    double grad = fabs(L[(_h+1)*WIDTH+(_w+1)] - L[_h*WIDTH+_w])
                                + fabs(A[(_h+1)*WIDTH+(_w+1)] - A[_h*WIDTH+_w])
                                + fabs(B[(_h+1)*WIDTH+(_w+1)] - B[_h*WIDTH+_w]);
                    if (grad < min_grad) {
                        min_grad = grad;
                        min_h = _h;
                        min_w = _w;
                    }
                }
            }
            CX.push_back(min_w);
            CY.push_back(min_h);
            int idx = min_h * WIDTH + min_w;
            CL.push_back(L[idx]);
            CA.push_back(A[idx]);
            CB.push_back(B[idx]);
        }
    }
    int SeedsNum = CX.size();
    // ...existing code...
    // 其余部分用 CX, CY, CL, CA, CB 替代原参数，聚类流程不变
    double  MAXDISTANCE=9999999999.99999;
    double *sigmal,*sigmaa,*sigmab,*sigmax,*sigmay,*clustersize;
    sigmal = ( double * ) calloc ( SeedsNum, sizeof ( double ) );
    sigmaa = ( double * ) calloc ( SeedsNum, sizeof ( double ) );
    sigmab = ( double * ) calloc ( SeedsNum, sizeof ( double ) );
    sigmax = ( double * ) calloc ( SeedsNum, sizeof ( double ) );
    sigmay = ( double * ) calloc ( SeedsNum, sizeof ( double ) );
    clustersize = ( double * ) calloc ( SeedsNum, sizeof ( double ) );
    int DisNum=WIDTH*HEIGHT;
    double *distvec;
    distvec = ( double * ) calloc ( DisNum, sizeof ( double ) );
    for(int ir=0;ir<HEIGHT;ir++)
        for(int ic=0;ic<WIDTH;ic++)
        {
            distvec[ir*WIDTH+ic]=MAXDISTANCE;
            labels[ir*WIDTH+ic]=-1;
        }
    double invwt=(M*M)/(S*S);
    int offset=2*S;
    for(int iter=0;iter<10;iter++)
    {
        for(int k=0;k<SeedsNum;k++)
        {
            int x1=(0>(CX[k]-offset))?0:(CX[k]-offset);
            int x2=((WIDTH-1)<(CX[k]+offset))?(WIDTH-1):(CX[k]+offset);
            int y1=(0>(CY[k]-offset))?0:(CY[k]-offset);
            int y2=((HEIGHT-1)<(CY[k]+offset))?(HEIGHT-1):(CY[k]+offset);
            for(int ir=y1;ir<=y2;ir++)
                for(int ic=x1;ic<=x2;ic++)
                {
                    double distLAB = (CL[k]-L[ir*WIDTH+ic])*(CL[k]-L[ir*WIDTH+ic]) +
                                               (CA[k]-A[ir*WIDTH+ic])*(CA[k]-A[ir*WIDTH+ic]) +
                                               (CB[k]-B[ir*WIDTH+ic])*(CB[k]-B[ir*WIDTH+ic]) ;
                    double distXY = (double(ir)-CY[k])*(double(ir)-CY[k]) + (double(ic)-CX[k])*(double(ic)-CX[k]) ;
                    double dist = distLAB+distXY*invwt;
                    if ( dist<distvec[ir*WIDTH+ic] )
                    {
                        distvec[ir*WIDTH+ic]=dist;
                        labels[ir*WIDTH+ic]=k;
                    }
                }
        }
        for(int ir=0;ir<HEIGHT;ir++)
            for(int ic=0;ic<WIDTH;ic++)
            {
                int index=labels[ir*WIDTH+ic];
                if (index != -1)
                {
                    sigmal[index]+=L[ir*WIDTH+ic];
                    sigmaa[index]+=A[ir*WIDTH+ic];
                    sigmab[index]+=B[ir*WIDTH+ic];
                    sigmax[index]+=double(ic);
                    sigmay[index]+=double(ir);
                    clustersize[index]+=1;
                }
            }
        for(int k=0;k<SeedsNum;k++)
        {
            if ( clustersize[k]<=0 )
                clustersize[k]=1;
            CL[k]=sigmal[k]/clustersize[k];
            CA[k]=sigmaa[k]/clustersize[k];
            CB[k]=sigmab[k]/clustersize[k];
            CX[k]=sigmax[k]/clustersize[k];
            CY[k]=sigmay[k]/clustersize[k];
        }
        for(int k=0;k<SeedsNum;k++)
        {
            clustersize[k]=0;sigmal[k]=0;sigmaa[k]=0;sigmab[k]=0;sigmax[k]=0;sigmay[k]=0;
        }
    }
    free(sigmal);
    free(sigmaa);
    free(sigmab);
    free(sigmax);
    free(sigmay);
    free(clustersize);
    free(distvec);
    int num_labels = SeedsNum;
    std::vector<int> all_labels(num_labels);
    for (int i = 0; i < num_labels; ++i) all_labels[i] = i;
    std::mt19937 rng((unsigned int)time(0));
    std::shuffle(all_labels.begin(), all_labels.end(), rng);
    int mask_count = int(num_labels * (1.0 - prob));
    std::unordered_set<int> mask_labels(all_labels.begin(), all_labels.begin() + mask_count);
    for (int n = 0; n < 1; ++n) {
        for (int y = 0; y < HEIGHT; ++y) {
            for (int x = 0; x < WIDTH; ++x) {
                int label = (int)labels[y * WIDTH + x];
                for (int c = 0; c < 3; ++c) {
                    if (mask_labels.count(label)) {
                        X_mask[n * HEIGHT * WIDTH * 3 + y * WIDTH * 3 + x * 3 + c] = static_cast<float>(rng() % 10000 / 10000.0);
                    } else {
                        X_mask[n * HEIGHT * WIDTH * 3 + y * WIDTH * 3 + x * 3 + c] = 1.0f;
                    }
                }
            }
        }
    }
}

