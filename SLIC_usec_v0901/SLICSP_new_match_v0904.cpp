#include <stdio.h>
#include <math.h>

//#define char16_t uint16_T

#include "EnfConn.h"

//-------------------------------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------------------------------

extern "C" {
    void SLICSP(double *CX,double *CY, double *CL, double *CA, double *CB,
        int SeedsNum, double *L, double *A, double *B, int WIDTH,int HEIGHT,
        double STEP, double M, double *labels);
}

extern "C" void SLICSP(double *CX,double *CY, double *CL, double *CA, double *CB, \
        int SeedsNum, double *L, double *A, double *B, int WIDTH,int HEIGHT, \
        double STEP, double M, double *labels) 
{
    // 优化1：只分配一次内存，避免重复分配/释放
    double  MAXDISTANCE=9999999999.99999;
    double *sigmal = (double *)calloc(SeedsNum, sizeof(double));
    double *sigmaa = (double *)calloc(SeedsNum, sizeof(double));
    double *sigmab = (double *)calloc(SeedsNum, sizeof(double));
    double *sigmax = (double *)calloc(SeedsNum, sizeof(double));
    double *sigmay = (double *)calloc(SeedsNum, sizeof(double));
    double *clustersize = (double *)calloc(SeedsNum, sizeof(double));
    int DisNum = WIDTH * HEIGHT;
    double *distvec = (double *)calloc(DisNum, sizeof(double));

    //Set default value
    for(int ir=0;ir<HEIGHT;ir++)
        for(int ic=0;ic<WIDTH;ic++)
        {
            distvec[ir*WIDTH+ic]=MAXDISTANCE;
            labels[ir*WIDTH+ic]=-1;
        }
    double invwt=(M*M)/(STEP*STEP);
    int offset=2*STEP;
    //Perform SLIC Superpixel 10 times for pixels' clustering to the centers
    for(int iter=0;iter<10;iter++)
    {
        // 优化2：循环顺序调整为行优先，提升缓存命中率
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
        } // for k loop
        
        //Calculate the mean value of L,A,B,X,Y for seeds
        for(int ir=0;ir<HEIGHT;ir++)
            for(int ic=0;ic<WIDTH;ic++)
            {
                    int index=labels[ir*WIDTH+ic];
                    if (index == -1)
                    { 
                        //printf("unclusted pixels, wrong!!!.\n");
                        //printf("bypass this pixel.\n");
                    }
                    else
                    {
                        sigmal[index]+=L[ir*WIDTH+ic];
                        sigmaa[index]+=A[ir*WIDTH+ic];
                        sigmab[index]+=B[ir*WIDTH+ic];
                        sigmax[index]+=double(ic);
                        sigmay[index]+=double(ir);
                        clustersize[index]+=1;
                    }
            }
        
        //Reset the information of seeds
        for(int k=0;k<SeedsNum;k++)
        {
            if ( clustersize[k]<=0 )
                clustersize[k]=1;
            //inv[k]=1.0/clustersize[k];
            CL[k]=sigmal[k]/clustersize[k];
            CA[k]=sigmaa[k]/clustersize[k];
            CB[k]=sigmab[k]/clustersize[k];
            CX[k]=sigmax[k]/clustersize[k];
            CY[k]=sigmay[k]/clustersize[k];
        }
        
        //Reset the temporary variables for next iteration
        for(int k=0;k<SeedsNum;k++)
        {
            clustersize[k]=0;sigmal[k]=0;sigmaa[k]=0;sigmab[k]=0;sigmax[k]=0;sigmay[k]=0;            
        }
        
    }   // for iter= loop
    
    free(sigmal);
    free(sigmaa);
    free(sigmab);
    free(sigmax);
    free(sigmay);
    free(clustersize);
    free(distvec);
}


