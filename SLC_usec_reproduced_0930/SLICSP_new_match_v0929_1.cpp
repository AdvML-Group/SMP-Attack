#include <stdio.h>
#include <math.h>
#include <ctime>

//#define char16_t uint16_T

// #include "mex.h"
#include "EnfConn.h"
#include <vector>
#include <algorithm>
#include <random>
#include <unordered_set>
//g++ SLICSP_new_match_v0929_1.cpp -fPIC -shared -o SLICSP_v0929_1.so
//-------------------------------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------------------------------

extern "C" {
    void SLICSP(double *L, double *A, double *B, int WIDTH, int HEIGHT,
                int K, double M, double prob, double *labels, float *X_mask);
}

void SLICSP(double *L, double *A, double *B, int WIDTH, int HEIGHT,
            int K, double M, double prob, double *labels, float *X_mask)
{
    int N = WIDTH * HEIGHT;
    int S = int(sqrt(double(N) / K));
    int max_seeds = ((HEIGHT + S - 1) / S) * ((WIDTH + S - 1) / S);
    double *CX = (double*)malloc(max_seeds * sizeof(double));
    double *CY = (double*)malloc(max_seeds * sizeof(double));
    double *CL = (double*)malloc(max_seeds * sizeof(double));
    double *CA = (double*)malloc(max_seeds * sizeof(double));
    double *CB = (double*)malloc(max_seeds * sizeof(double));
    int SeedsNum = 0;
    // 行优先初始化中心点
    //time
    //clock_t t_start = clock();
    for (int h = S / 2; h < HEIGHT; h += S) {
        for (int w = S / 2; w < WIDTH; w += S) {
            double min_grad = 1e20;
            int min_h = h, min_w = w;
            for (int dh = -1; dh <= 1; ++dh) {
                for (int dw = -1; dw <= 1; ++dw) {
                    int _h = h + dh, _w = w + dw;
                    int _hw1=(_h+1)*WIDTH+(_w+1);
                    int _hw2=_h*WIDTH+_w;
                    if (_h < 0 || _h >= HEIGHT - 1 || _w < 0 || _w >= WIDTH - 1) continue;
                    double grad = (L[_hw1] - L[_hw2])*(L[_hw1] - L[_hw2])
                                + (A[_hw1] - A[_hw2])*(A[_hw1] - A[_hw2])
                                + (B[_hw1] - B[_hw2])*(B[_hw1] - B[_hw2]);   //fabs
                    if (grad < min_grad) {
                        min_grad = grad;
                        min_h = _h;
                        min_w = _w;
                    }
                }
            }
            CX[SeedsNum] = min_w;
            CY[SeedsNum] = min_h;
            int idx = min_h * WIDTH + min_w;
            CL[SeedsNum] = L[idx];
            CA[SeedsNum] = A[idx];
            CB[SeedsNum] = B[idx];
            SeedsNum++;
        }
    }
    //clock_t t_end = clock();
    //printf("Seed initialization time: %.6f seconds\n", (double)(t_end - t_start) / CLOCKS_PER_SEC);
    double  MAXDISTANCE=9999999999.99999;
    double *sigmal = (double*)calloc(SeedsNum, sizeof(double));
    double *sigmaa = (double*)calloc(SeedsNum, sizeof(double));
    double *sigmab = (double*)calloc(SeedsNum, sizeof(double));
    double *sigmax = (double*)calloc(SeedsNum, sizeof(double));
    double *sigmay = (double*)calloc(SeedsNum, sizeof(double));
    double *clustersize = (double*)calloc(SeedsNum, sizeof(double));
    int DisNum = WIDTH * HEIGHT;
    double *distvec = (double*)calloc(DisNum, sizeof(double));
    for(int ir=0;ir<HEIGHT;ir++)
        for(int ic=0;ic<WIDTH;ic++) {
            distvec[ir*WIDTH+ic]=MAXDISTANCE;
            labels[ir*WIDTH+ic]=-1;
        }
    double invwt=(M*M)/(S*S);
    int offset=2*S;
    for(int iter=0;iter<5;iter++) {
        for(int k=0;k<SeedsNum;k++) {
            int x1=(0>(CX[k]-offset))?0:(int)(CX[k]-offset);
            int x2=((WIDTH-1)<(CX[k]+offset))?(WIDTH-1):(int)(CX[k]+offset);
            int y1=(0>(CY[k]-offset))?0:(int)(CY[k]-offset);
            int y2=((HEIGHT-1)<(CY[k]+offset))?(HEIGHT-1):(int)(CY[k]+offset);
            for(int ir=y1;ir<=y2;ir++)
                for(int ic=x1;ic<=x2;ic++) {
                    int _iw=ir*WIDTH+ic;
                    double distLAB = (CL[k]-L[_iw])*(CL[k]-L[_iw]) +
                                     (CA[k]-A[_iw])*(CA[k]-A[_iw]) +
                                     (CB[k]-B[_iw])*(CB[k]-B[_iw]);
                    double distXY = (double(ir)-CY[k])*(double(ir)-CY[k]) + (double(ic)-CX[k])*(double(ic)-CX[k]);
                    double dist = distLAB+distXY*invwt;
                    if (dist < distvec[_iw]) {
                        distvec[_iw]=dist;
                        labels[_iw]=k;
                    }
                }
        }
        for(int ir=0;ir<HEIGHT;ir++)
            for(int ic=0;ic<WIDTH;ic++) {
                int index=labels[ir*WIDTH+ic];
                if (index != -1) {
                    sigmal[index]+=L[ir*WIDTH+ic];
                    sigmaa[index]+=A[ir*WIDTH+ic];
                    sigmab[index]+=B[ir*WIDTH+ic];
                    sigmax[index]+=ic;
                    sigmay[index]+=ir;
                    clustersize[index]+=1;
                }
            }
        for(int k=0;k<SeedsNum;k++) {
            if (clustersize[k]<=0) clustersize[k]=1;
            CL[k]=sigmal[k]/clustersize[k];
            CA[k]=sigmaa[k]/clustersize[k];
            CB[k]=sigmab[k]/clustersize[k];
            CX[k]=sigmax[k]/clustersize[k];
            CY[k]=sigmay[k]/clustersize[k];
        }
        for(int k=0;k<SeedsNum;k++) {
            clustersize[k]=0;sigmal[k]=0;sigmaa[k]=0;sigmab[k]=0;sigmax[k]=0;sigmay[k]=0;
        }
    }
    free(sigmal); free(sigmaa); free(sigmab); free(sigmax); free(sigmay); free(clustersize); free(distvec);
    free(CX); free(CY); free(CL); free(CA); free(CB);
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

