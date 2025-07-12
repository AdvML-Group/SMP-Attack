#include <stdio.h>
#include <math.h>

//#define char16_t uint16_T

#include "mex.h"
#include "EnfConn.h"

//-------------------------------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------------------------------

void SLICSP(double *CX,double *CY, double *CL, double *CA, double *CB, \
        int SeedsNum, double *L, double *A, double *B, int WIDTH,int HEIGHT, \
        double STEP, double M, double *labels) 
{
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
    int offset=STEP;
    
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

//--------------------------------------------------*------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------------------------------

//-------------------------------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------------------------------

void LabelConnectivity(double *labels,int width,int height,double *nlabels,int &numlabels,int K) 
{
//	const int dx8[8] = {-1, -1,  0,  1, 1, 1, 0, -1};
//	const int dy8[8] = { 0, -1, -1, -1, 0, 1, 1,  1};
	int dx4[4] = {-1,  0,  1,  0};
	int dy4[4] = { 0, -1,  0,  1};

	int sz = width*height;
    
	int SUPSZ = sz/K;
    
	for( int i = 0; i < sz; i++ ) nlabels[i] = -1;
    
	int label(0);
    
	double* xvec = ( double * ) calloc ( sz, sizeof ( double ) );
	double* yvec = ( double * ) calloc ( sz, sizeof ( double ) );
    
	int oindex(0);
    
	int adjlabel(0);//adjacent label
    
	for( int j = 0; j < height; j++ )
	{
		for( int k = 0; k < width; k++ )
		{
			if( 0 > nlabels[oindex] )
			{
				nlabels[oindex] = label;
				//--------------------
				// Start a new segment
				//--------------------
				xvec[0] = k;
				yvec[0] = j;
				//-------------------------------------------------------
				// Quickly find an adjacent label for use later if needed
				//-------------------------------------------------------
				{for( int n = 0; n < 4; n++ )
				{
					int x = xvec[0] + dx4[n];
					int y = yvec[0] + dy4[n];
					if( (x >= 0 && x < width) && (y >= 0 && y < height) )
					{
						int nindex = y*width + x;
						if(nlabels[nindex] >= 0) adjlabel = nlabels[nindex];
					}
				}}

				int count(1);
				for( int c = 0; c < count; c++ )
				{
					for( int n = 0; n < 4; n++ )
					{
						int x = xvec[c] + dx4[n];
						int y = yvec[c] + dy4[n];

						if( (x >= 0 && x < width) && (y >= 0 && y < height) )
						{
							int nindex = y*width + x;

							if( 0 > nlabels[nindex] && labels[oindex] == labels[nindex] )
							{
								xvec[count] = x;
								yvec[count] = y;
								nlabels[nindex] = label;
								count++;
							}
						}

					}
				}
				//-------------------------------------------------------
				// If segment size is less then a limit, assign an
				// adjacent label found before, and decrement label count.
				//-------------------------------------------------------
				if(count <= SUPSZ >> 2)
                //if(count <= 20)
				{
					for( int c = 0; c < count; c++ )
					{
						int ind = yvec[c]*width+xvec[c];
						nlabels[ind] = adjlabel;
					}
					label--;
				}
				label++;
			}
			oindex++;
		}
	}
    numlabels=label;
	free(xvec);
	free(yvec);
}

//-------------------------------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------------------------------
void DrawContoursAroundSuperpixel(double *ubuff,double *labels,int width,int height)
{
	const int dx8[8] = {-1, -1,  0,  1, 1, 1, 0, -1};
	const int dy8[8] = { 0, -1, -1, -1, 0, 1, 1,  1};
   
	//reset the image to black color for extracting superpixels
	for( int j = 0; j < height; j++ )
	{
		for( int k = 0; k < width; k++ )
		{
			int index=j*width+k;
			ubuff[index]=0;
		}
	}

	int sz = width*height;
    int* istaken = ( int * ) calloc ( sz, sizeof ( int ) ); // default value:0--->false
	double* contourx = ( double * ) calloc ( sz, sizeof ( double ) );
    double* contoury = ( double * ) calloc ( sz, sizeof ( double ) );
	int mainindex(0);int cind(0);
	for( int j = 0; j < height; j++ )
	{
		for( int k = 0; k < width; k++ )
		{
			int np(0);
			for( int i = 0; i < 8; i++ )
			{
				int x = k + dx8[i];
				int y = j + dy8[i];

				if( (x >= 0 && x < width) && (y >= 0 && y < height) )
				{
					int index = y*width + x;

					if( 0 == istaken[index] )//comment this to obtain internal contours
					{
						if( labels[mainindex] != labels[index] ) np++;
					}
				}
			}
			if( np > 1 )
			{
				contourx[cind] = k;
				contoury[cind] = j;
				istaken[mainindex] = 1;
				cind++;
			}
			mainindex++;
		}
	}

	int numboundpix = cind;
	for( int j = 0; j < numboundpix; j++ )
	{
		int ii = contoury[j]*width + contourx[j];
		ubuff[ii] = 1; 
		for( int n = 0; n < 8; n++ )
		{
			int x = contourx[j] + dx8[n];
			int y = contoury[j] + dy8[n];
			if( (x >= 0 && x < width) && (y >= 0 && y < height) )
			{
				int ind = y*width + x;
				if(istaken[ind]==0) ubuff[ind] = 0;
			}
		}
	}
    free(istaken);
    free(contourx);
    free(contoury);
}
//-------------------------------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------------------------------

//-------------------------------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------------------------------

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  /*seeds:(x,y) Lab_img:(L,A,B)*/
  double *in_CX, *in_CY, *in_CL, *in_CA, *in_CB, *in_L, *in_A, *in_B, *STEP, *M, *K;
  /*output: labels map*/
  double *out_labels,*out_enlabels,*out_contour;
  int iWidth, iHeight;
  int SeedsNum;
  int SuperpixelNum;
  
  /* Check for proper number of arguments. */
  if (nrhs != 11) {
    mexErrMsgTxt("Eight input required <X,Y,L,A,B,STEP,M,K>.");
  } else if (nlhs > 3) {
    mexErrMsgTxt("Too many output arguments");
  }
  
  /* The input must be a noncomplex scalar double.*/
  iWidth = mxGetM(prhs[6]);
  iHeight = mxGetN(prhs[6]);
  //printf("The size of image: row-%d, col-%d.\n",iWidth,iHeight);
  SeedsNum = mxGetM(prhs[0]);
  //printf("The number of seeds is %d.\n",SeedsNum);
  
  /* Create matrix for the return argument. */
  plhs[0] = mxCreateDoubleMatrix(iWidth,iHeight, mxREAL);
  plhs[1] = mxCreateDoubleMatrix(iWidth,iHeight, mxREAL);
  plhs[2] = mxCreateDoubleMatrix(iWidth,iHeight, mxREAL);
  
  /* Assign pointers to each input and output. */
  in_CX = mxGetPr(prhs[0]);
  in_CY = mxGetPr(prhs[1]);
  in_CL = mxGetPr(prhs[2]);
  in_CA = mxGetPr(prhs[3]);
  in_CB = mxGetPr(prhs[4]);
  in_L = mxGetPr(prhs[5]);
  in_A = mxGetPr(prhs[6]);
  in_B = mxGetPr(prhs[7]);
  STEP =mxGetPr(prhs[8]);
  M =mxGetPr(prhs[9]);
  K =mxGetPr(prhs[10]);
  out_labels = mxGetPr(plhs[0]);
  out_enlabels = mxGetPr(plhs[1]);
  out_contour = mxGetPr(plhs[2]);
  
  /* Perform SLICSuperpixel. */
  /* 
     待修改：
     1. 参考python版本的init_clusters、reset_pixels、get_gradient、move_clusters的相关逻辑，在attack_use_c.py代码中超像素中心点初始化部分进行对应实现。
     2. SLICSP删除labelConnectivity、EnforceConnectivity函数，和Python版本进行匹配。
     2.1 SLICSP核心代码搜索范围offset=step，与python版本不一致，需要修改为"2S"，即offset=2*step.
     3. 需要验证C++获取的数据与Python是完全一致的。
     4. 最后需要将超像素代码和攻击代码进行拆分，超像素代码应该独立为一个py文件，攻击attack代码调用该文件的函数。
  */
  SLICSP(in_CX,in_CY,in_CL,in_CA,in_CB,SeedsNum,in_L,in_A,in_B,iWidth,iHeight,*STEP,*M,out_labels);
  LabelConnectivity(out_labels,iWidth,iHeight,out_enlabels,SuperpixelNum,*K); // change "SUPSZ>>2" to 20
  EnforceConnectivity(in_L,in_A,in_B,*M,*STEP,out_enlabels,iWidth,iHeight);  // add 2-stage EC. 
  //printf("The number of superpixel is %d.\n",SuperpixelNum);
  DrawContoursAroundSuperpixel(out_contour,out_enlabels,iWidth,iHeight);
}
