#include<iostream>
#include<queue>
#include<vector>
#include<algorithm>
#include<float.h>
using namespace std;

//================================

class Superpixel   // Class for single superpixel in the label map
{
public:
	int Label;
	int Size;
	vector<int> Neighbor;
	Superpixel(int L=0,int S=0):Label(L),Size(S){}
	vector<int> xLoc;
	vector<int> yLoc;
	friend bool operator==(Superpixel& S,int L);
	friend bool operator==(int L,Superpixel& S);
};

bool operator==(Superpixel& S,int L)
{
	return S.Label==L;
}

bool operator==(int L,Superpixel& S)
{
	return S.Label==L;
}

//================================

void EnforceConnectivity(
		double* fL,
		double* fA,
		double* fB,
        double spM,
        double spSTEP,
		double* label,
		int nRows,
		int nCols
	)
{
    int threshold = int(spSTEP*spSTEP)>>2;   //  Th: 1/4 * SUPSZ
	unsigned char** mask=new unsigned char*[nRows];
	for(int i=0;i<nRows;i++)
	{
		mask[i]=new unsigned char[nCols];
		for(int j=0;j<nCols;j++)
			mask[i][j]=0;
	}

	vector<unsigned short>strayX;
	vector<unsigned short>strayY;
	vector<unsigned short>Size;
	queue<unsigned short> xLoc;
	queue<unsigned short> yLoc;
	vector<double>centerL;
	vector<double>centerA;
	vector<double>centerB;
	vector<double>centerX;
	vector<double>centerY;
    
	int sLabel=-1;
	int L;
	for(int i=0;i<nRows;i++)
		for(int j=0;j<nCols;j++)
		{
			if(mask[i][j]==0) // mask is the visited map
			{
				sLabel++;
				int Count=1;
                
				centerL.insert(centerL.end(),0);   // add one element into the vector
				centerA.insert(centerA.end(),0);
				centerB.insert(centerB.end(),0);
				centerX.insert(centerX.end(),0);
				centerY.insert(centerY.end(),0);
				strayX.insert(strayX.end(),i);
				strayY.insert(strayY.end(),j);
                
                // calculate the features of superpixel centers
				centerL[sLabel]+=fL[j*nRows+i];
				centerA[sLabel]+=fA[j*nRows+i];
				centerB[sLabel]+=fB[j*nRows+i];
				centerX[sLabel]+=i;
				centerY[sLabel]+=j;
                
				L=label[j*nRows+i];
				label[j*nRows+i]=sLabel;
				mask[i][j]=1;
				xLoc.push(i);yLoc.push(j);
                
				while(!xLoc.empty())  // find the pixels with same label
				{
					int x=xLoc.front();xLoc.pop();
					int y=yLoc.front();yLoc.pop();
					int minX=(x-1<=0)?0:x-1;
					int maxX=(x+1>=nRows-1)?nRows-1:x+1;
					int minY=(y-1<=0)?0:y-1;
					int maxY=(y+1>=nCols-1)?nCols-1:y+1;
					for(int m=minX;m<=maxX;m++)
						for(int n=minY;n<=maxY;n++)
						{
							if(mask[m][n]==0&&label[n*nRows+m]==L)
							{
								Count++;
								xLoc.push(m);
								yLoc.push(n);
								mask[m][n]=1;
								label[n*nRows+m]=sLabel;
								centerL[sLabel]+=fL[n*nRows+m];
								centerA[sLabel]+=fA[n*nRows+m];
								centerB[sLabel]+=fB[n*nRows+m];
								centerX[sLabel]+=m;
								centerY[sLabel]+=n;
							}
						}
				} // end of the "while" loop
                
				Size.insert(Size.end(),Count);
				centerL[sLabel]/=Size[sLabel];     // mean of <L,A,B,X,Y> in single superpixel
				centerA[sLabel]/=Size[sLabel];
				centerB[sLabel]/=Size[sLabel];
				centerX[sLabel]/=Size[sLabel];
				centerY[sLabel]/=Size[sLabel];
			}
		} // end of two "for" loops

	sLabel=sLabel+1;
	int Count=0;
	
	vector<int>::iterator Pointer;
	vector<Superpixel> Sarray;
	for(int i=0;i<sLabel;i++)  // Fetch the information of each superpixel
	{
		if(Size[i]<threshold)   
		{
			int x=strayX[i];int y=strayY[i];
			L=label[y*nRows+x];
			mask[x][y]=0;
			int indexMark=0;
			Superpixel S(L,Size[i]);
			S.xLoc.insert(S.xLoc.end(),x);
			S.yLoc.insert(S.yLoc.end(),y);
			while(indexMark<S.xLoc.size())
			{
				x=S.xLoc[indexMark];y=S.yLoc[indexMark];
				indexMark++;
				int minX=(x-1<=0)?0:x-1;
				int maxX=(x+1>=nRows-1)?nRows-1:x+1;
				int minY=(y-1<=0)?0:y-1;
				int maxY=(y+1>=nCols-1)?nCols-1:y+1;
				for(int m=minX;m<=maxX;m++)
					for(int n=minY;n<=maxY;n++)
					{
						if(mask[m][n]==1&&label[n*nRows+m]==L)
						{
							mask[m][n]=0;
							S.xLoc.insert(S.xLoc.end(),m);   // Location of each pixel in the Superpixel
							S.yLoc.insert(S.yLoc.end(),n);
						}
						else if(label[n*nRows+m]!=L)
						{
							int NewLabel=label[n*nRows+m];
							Pointer=find(S.Neighbor.begin(),S.Neighbor.end(),NewLabel);
							if(Pointer==S.Neighbor.end())
							{
								S.Neighbor.insert(S.Neighbor.begin(),NewLabel);  // Label of Neighbors of the Superpixel.
							}
						}
					}

			}
			Sarray.insert(Sarray.end(),S);
		}
	}  // end of "for" loop

	vector<Superpixel>::iterator S;
	vector<int>::iterator I;
	vector<int>::iterator I2;
	S=Sarray.begin();
	while(S!=Sarray.end())   // Merge the Superpixels with medium size to Neighbors
	{
		double MinDist=DBL_MAX;
		int Label1=(*S).Label;
		int Label2=-1;
		for(I=(*S).Neighbor.begin();I!=(*S).Neighbor.end();I++)   // Find the neighbor superpixel with the minimum distance.
		{
            double invwt = (spM*spM)/(spSTEP*spSTEP);
            double D = (centerL[Label1]-centerL[*I])*(centerL[Label1]-centerL[*I]) +
                    (centerA[Label1]-centerA[*I])*(centerA[Label1]-centerA[*I]) + 
                    (centerB[Label1]-centerB[*I])*(centerB[Label1]-centerB[*I]) + 
                    invwt*(centerX[Label1]-centerX[*I])*(centerX[Label1]-centerX[*I]) +
                    invwt*(centerY[Label1]-centerY[*I])*(centerY[Label1]-centerY[*I]);

			if(D<MinDist)
			{
				MinDist=D;
				Label2=(*I);
			}
		}
        
		double Size1=Size[Label1];
		double Size2=Size[Label2];
		double Size12=Size1+Size2;
        centerL[Label2]=(Size2*centerL[Label2]+Size1*centerL[Label1])/Size12;
        centerA[Label2]=(Size2*centerA[Label2]+Size1*centerA[Label1])/Size12;
        centerB[Label2]=(Size2*centerB[Label2]+Size1*centerB[Label1])/Size12;
        centerX[Label2]=(Size2*centerX[Label2]+Size1*centerX[Label1])/Size12;
        centerY[Label2]=(Size2*centerY[Label2]+Size1*centerY[Label1])/Size12;
        
		for(int i=0;i<(*S).xLoc.size();i++)  // Update labels
		{
			int x=(*S).xLoc[i];int y=(*S).yLoc[i];
			label[y*nRows+x]=Label2;
		}
        
		vector<Superpixel>::iterator Stmp;
		Stmp=find(Sarray.begin(),Sarray.end(),Label2);
		if(Stmp!=Sarray.end())  // Update the information of class-vector Superpixel.
		{
			Size[Label2]=Size[Label1]+Size[Label2];
			if(Size[Label2]>=threshold)
			{
				Sarray.erase(Stmp);
				Sarray.erase(S);
			}
			else
			{
				(*Stmp).xLoc.insert((*Stmp).xLoc.end(),(*S).xLoc.begin(),(*S).xLoc.end()); // merge Label1-SP to Label2-SP
				(*Stmp).yLoc.insert((*Stmp).yLoc.end(),(*S).yLoc.begin(),(*S).yLoc.end());
				(*Stmp).Neighbor.insert((*Stmp).Neighbor.end(),(*S).Neighbor.begin(),(*S).Neighbor.end());
				sort((*Stmp).Neighbor.begin(),(*Stmp).Neighbor.end());
				I=unique((*Stmp).Neighbor.begin(),(*Stmp).Neighbor.end());
				(*Stmp).Neighbor.erase(I,(*Stmp).Neighbor.end());
				I=find((*Stmp).Neighbor.begin(),(*Stmp).Neighbor.end(),Label1);
				(*Stmp).Neighbor.erase(I);
				I=find((*Stmp).Neighbor.begin(),(*Stmp).Neighbor.end(),Label2);
				(*Stmp).Neighbor.erase(I);
				Sarray.erase(S);
			}
		}
		else
		{
			Sarray.erase(S);
		}
        
		for(int i=0;i<Sarray.size();i++) // update neighbor information of superpixel, Label1 is changed to Label2
		{
			I=find(Sarray[i].Neighbor.begin(),Sarray[i].Neighbor.end(),Label1);
			I2=find(Sarray[i].Neighbor.begin(),Sarray[i].Neighbor.end(),Label2);
			if(I!=Sarray[i].Neighbor.end()&&I2!=Sarray[i].Neighbor.end())
				Sarray[i].Neighbor.erase(I);
			else if(I!=Sarray[i].Neighbor.end()&&I2==Sarray[i].Neighbor.end())
				(*I)=Label2;
		}
		S=Sarray.begin();
	} // end of "while" loop
    
	for(int i=0;i<nRows;i++)
		delete [] mask[i];
	delete [] mask;
	return;
}

