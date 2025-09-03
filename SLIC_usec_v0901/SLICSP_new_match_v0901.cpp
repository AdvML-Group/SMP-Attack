#include <stdio.h>
#include <math.h>

//#define char16_t uint16_T

#include "EnfConn.h"

//-------------------------------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------------------------------

void SLICSP(double *CX,double *CY, double *CL, double *CA, double *CB, \
        int SeedsNum, double *L, double *A, double *B, int WIDTH,int HEIGHT, \
        double STEP, double M, double *labels) 
{
    // 在SLICSP函数最前面添加
    // 1. 均匀初始化中心点
    int S = (int)sqrt((double)(WIDTH * HEIGHT) / SeedsNum);
    int k = 0;
    for (int h = S/2; h < HEIGHT; h += S) {
        for (int w = S/2; w < WIDTH; w += S) {
            // 2. 在3x3邻域内寻找梯度最小点
            double min_grad = 1e10;
            int min_h = h, min_w = w;
            for (int dh = -1; dh <= 1; dh++) {
                for (int dw = -1; dw <= 1; dw++) {
                    int _h = h + dh, _w = w + dw;
                    if (_h >= 0 && _h < HEIGHT-1 && _w >= 0 && _w < WIDTH-1) {
                        double grad = 
                            (L[(_h+1)*WIDTH+(_w+1)] - L[_h*WIDTH+_w]) +
                            (A[(_h+1)*WIDTH+(_w+1)] - A[_h*WIDTH+_w]) +
                            (B[(_h+1)*WIDTH+(_w+1)] - B[_h*WIDTH+_w]);
                        if (grad < min_grad) {
                            min_grad = grad;
                            min_h = _h;
                            min_w = _w;
                        }
                    }
                }
            }
            CX[k] = min_w;
            CY[k] = min_h;
            CL[k] = L[min_h*WIDTH + min_w];
            CA[k] = A[min_h*WIDTH + min_w];
            CB[k] = B[min_h*WIDTH + min_w];
            k++;
            if (k >= SeedsNum) break;
        }
        if (k >= SeedsNum) break;
    }
    //printf("wid=%d,hei=%d.\n",WIDTH,HEIGHT);
    double  MAXDISTANCE=9999999999.99999;
    //Temporary variables for saving the information(X,Y,L,A,B) of Seeds
    double *sigmal,*sigmaa,*sigmab,*sigmax,*sigmay,*clustersize;
    sigmal = ( double * ) calloc ( SeedsNum, sizeof ( double ) );
    sigmaa = ( double * ) calloc ( SeedsNum, sizeof ( double ) );
    sigmab = ( double * ) calloc ( SeedsNum, sizeof ( double ) );
    sigmax = ( double * ) calloc ( SeedsNum, sizeof ( double ) );
    sigmay = ( double * ) calloc ( SeedsNum, sizeof ( double ) );
    clustersize = ( double * ) calloc ( SeedsNum, sizeof ( double ) );
    
    //Distance Variables
    int DisNum=WIDTH*HEIGHT;
    double *distvec;
    distvec = ( double * ) calloc ( DisNum, sizeof ( double ) );
   
    //Set default value
    int cnt=0;
    for(int ir=0;ir<HEIGHT;ir++)
        for(int ic=0;ic<WIDTH;ic++)
        {
            distvec[ir*WIDTH+ic]=MAXDISTANCE;
            labels[ir*WIDTH+ic]=-1;
        }
    
    double invwt=(M*M)/(STEP*STEP);
    int offset=2*STEP;  ////int offset=STEP;
    
    //Perform SLIC Superpixel 10 times for pixels' clustering to the centers
    for(int iter=0;iter<10;iter++)
    {
        //printf("Perform SLIC Superpixel: Iteration-%d\n",iter+1);
        
        for(int k=0;k<SeedsNum;k++)
        {
            //Generate Neighbourhood around the Seed(x,y)
            int x1=(0>(CX[k]-offset))?0:(CX[k]-offset);
            int x2=((WIDTH-1)<(CX[k]+offset))?(WIDTH-1):(CX[k]+offset);
            int y1=(0>(CY[k]-offset))?0:(CY[k]-offset);
            int y2=((HEIGHT-1)<(CY[k]+offset))?(HEIGHT-1):(CY[k]+offset);
            
            //Calculate the defined distance
            for(int ir=y1;ir<=y2;ir++)
                for(int ic=x1;ic<=x2;ic++)
                {
                    //int index=Y[k]*WIDTH+X[k];
                    double distLAB = (CL[k]-L[ir*WIDTH+ic])*(CL[k]-L[ir*WIDTH+ic]) +
                                               (CA[k]-A[ir*WIDTH+ic])*(CA[k]-A[ir*WIDTH+ic]) +
                                               (CB[k]-B[ir*WIDTH+ic])*(CB[k]-B[ir*WIDTH+ic]) ;
                    double distXY = (double(ir)-CY[k])*(double(ir)-CY[k]) + (double(ic)-CX[k])*(double(ic)-CX[k]) ;
                    //double dist = sqrt(distLAB)+sqrt(distXY*invwt);
                    //double dist = distXY*invwt;
                    //double dist = distLAB*0+distXY*invwt;
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


