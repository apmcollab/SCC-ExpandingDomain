/*
 * ScreenedNpole3d.h
 *
 *  A class for creating and caching smooth moment matching functions.
 *
 *  The moment matching functions is constructed using
 *  a linear combination of a ScreenedSoln3d and
 *  it's derivatives.
 *
 *  Moments with multi-indices from 0 to maxOrder (= 0, 1, 2)
 *  are matched.
 *
 *
 *  Created on: July 3, 2015
 *      Author: anderson
 */
/*
#############################################################################
#
# Copyright 2015-16 Chris Anderson
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the Lesser GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# For a copy of the GNU General Public License see
# <http://www.gnu.org/licenses/>.
#
#############################################################################
*/

#ifdef _MSC_VER
#include "iso646.h"          // So "and" is equivalenced to &&
typedef unsigned int uint;   // Define uint to be unsigned int
#undef min
#undef max
#endif

#include "DoubleVectorNd/SCC_DoubleVector3d.h"
#include "GridFunctionNd/SCC_GridFunction3d.h"

#include "ScreenedSoln3d.h"

#include <cmath>
#include <cstdio>
#include <vector>
#include <algorithm>

#ifndef SCREENED_N_POLE_3D_
#define SCREENED_N_POLE_3D_

#ifdef _OPENMP
#include <omp.h>
#endif

#define  DEFAULT_DIFFERENTIABILITY_ 6
//
// Moment array specification
//
// (0) = constant
// (1) = x
// (2) = y
// (3) = z
// (4) = x^2
// (5) = xy
// (6) = xz
// (7) = y^2
// (8) = yz
// (9) = z^2

class ScreenedNpole3d
{
    public:

	ScreenedNpole3d()
	{
	this->differentiabilityOrder = DEFAULT_DIFFERENTIABILITY_;
	initialize();
	}


	// Creates an ScreenedNpole3d used to match moments with multi-indices from 0 to maxOrder

	ScreenedNpole3d(double xPos, double yPos, double zPos, double radius, int maxOrder, double laplaceCoeff, double screenCoeff)
	{
	this->differentiabilityOrder = DEFAULT_DIFFERENTIABILITY_;
	initialize(xPos,yPos,zPos,radius,maxOrder,laplaceCoeff,screenCoeff);
	}

    // Creates an ScreenedNpole3d used to to match moments with multi-indices from 0 to maxOrder
    // and with strengths given by the values in the str array of values

	ScreenedNpole3d(double xPos, double yPos, double zPos, double radius,std::vector<double>& str,
	int maxOrder, double laplaceCoeff, double screenCoeff)
	{
	this->differentiabilityOrder = DEFAULT_DIFFERENTIABILITY_;
	initialize(xPos,yPos,zPos,radius,str,maxOrder,laplaceCoeff,screenCoeff);
	}


	void setSourceDifferentiability(int diffOrder)
	{
		this->differentiabilityOrder = diffOrder;
		this->screenedSoln.setSourceDifferentiability(diffOrder);
	}

	int getSourceDifferentiablity()
	{
	return this->differentiabilityOrder;
	}

    void setCoefficients(double laplaceCoeff, double screenCoeff)
	{
	this->laplaceCoeff  =  laplaceCoeff;
	this->screenCoeff = screenCoeff;
	this->screenedSoln.setCoefficients(laplaceCoeff,screenCoeff);
	}

	void getCoefficients(double& laplaceCoeff, double& screenCoeff) const
	{
	laplaceCoeff  = this->laplaceCoeff;
	screenCoeff   = this->screenCoeff;
	}


	void initialize()
	{
	    screenedSoln.initialize();
		momentCount    = 0;
		radius         = 0;
		laplaceCoeff   = 1.0;
		screenCoeff    = 0.0;
		xPos           = 0.0;
		yPos           = 0.0;
	    zPos           = 0.0;
	}


	void initialize(double xPos, double yPos, double zPos, double radius, int maxOrder, double laplaceCoeff, double screenCoeff)
	{

	switch (maxOrder)
   	{
   	case 0  :  this->momentCount = 1; break;
   	case 1  :  this->momentCount = 4; break;
   	case 2  :  this->momentCount = 10; break;
   	default :  this->momentCount = 10; break;
   	}
	std::vector<double> str(momentCount, 0.0);
	initialize(xPos,yPos,zPos,radius,str,maxOrder,laplaceCoeff,screenCoeff);
	}

	void initialize(double xPos, double yPos, double zPos, double radius, std::vector<double>& str, int maxOrder,
	double laplaceCoeff, double screenCoeff)
	{
    switch (maxOrder)
   	{
   	case 0  :  this->momentCount = 1; break;
   	case 1  :  this->momentCount = 4; break;
   	case 2  :  this->momentCount = 10; break;
   	default :  this->momentCount = 10; break;
   	}

	this->xPos          = xPos;
	this->yPos          = yPos;
	this->zPos          = zPos;
	this->radius        = radius;
	this->laplaceCoeff  = laplaceCoeff;
	this->screenCoeff   = screenCoeff;
    this->differentiabilityOrder =  DEFAULT_DIFFERENTIABILITY_;

    // Capture any input strengths

    this->str.clear();
    this->str.resize(momentCount,0.0);
    for(long i = 0; i < std::min(momentCount,(long)(str.size())); i++) {this->str[i] = str[i];}

    // Now set moments so that specified number match input strengths

    setMoments(this->str);

    screenedSoln.initialize(xPos, yPos, zPos, radius, 1.0,laplaceCoeff,screenCoeff);
	screenedSoln.setSourceDifferentiability(differentiabilityOrder);
}

	virtual ~ScreenedNpole3d()
	{}



    double operator()(double x, double y,double z) const
    {
    return evaluatePotential(x,y,z);
    }

    // This routine evaluates the discrete potential
    // associated with an nPole with a specified collection
    // of strengths. The discrete components of the nPole
    // are only computed if there has been a grid structure
    // change.
    //

    void evaluatePotential(SCC::GridFunction3d& V) const
    {
    if(momentCount == 0) return;
    createPotential(V);
    return;
    }

    void evaluateSource(SCC::GridFunction3d& V) const
    {
    if(momentCount == 0) return;
    createSource(V);
    return;
    }

    double evaluatePotential(double x, double y, double z) const
    {

    int maxOrder;
	switch (momentCount)
   	{
   	case 1  :  maxOrder = 0; break;
   	case 4  :  maxOrder = 1; break;
   	case 10 :  maxOrder = 2; break;
   	default :  maxOrder = 2; break;
   	}

   	std::vector<double> dList(momentCount);

    screenedSoln.derivatives(x,y,z,dList,maxOrder);
    double potValue = 0.0;
    for(long i = 0; i < momentCount; i++)
    {
    	potValue += str[i]*dList[i];
    }
    return potValue;
    }

    double evaluateSource(double x, double y,double z) const
    {
    int maxOrder;
	switch (momentCount)
   	{
   	case 1  :  maxOrder = 0; break;
   	case 4  :  maxOrder = 1; break;
   	case 10 :  maxOrder = 2; break;
   	default :  maxOrder = 2; break;
   	}
   	std::vector<double> dList(momentCount);

    screenedSoln.sourceDerivatives(x,y,z,dList,maxOrder);

    double srcValue = 0.0;
    long   i;
    for(i = 0; i < momentCount; i++)
    {
    	srcValue += str[i]*dList[i];
    }
    return srcValue;
    }

    //  Returns a std::function that is bound to the source evaluation operator of *this
 

	std::function<double(double,double,double)> getSourceEvaluationPtr() const
	{
	std::function<double(double,double,double)> F = [this](double x,double y,double z) {return this->evaluateSource(x,y,z);};
	return F;
	}
 

	 //  Returns a std::function that is bound to the potential evaluation operator of *this
	std::function<double(double,double,double)> getPotentialEvaluationPtr() const
	{
	std::function<double(double,double,double)> F = [this](double x,double y,double z) {return this->evaluatePotential(x,y,z);};
	return F;
	}
 

    void setStrength(long index, double val)
    {
    	if(index < (long)str.size())
   		{
    	str[index] = val;
    	}
    }

    void setStrength(std::vector<double>& str)
    {
    for(long i = 0; i < (long)this-> str.size(); i++)
    {
    	this->str[i] = str[i];
    }
    }

	// Moment indexing:
	//
	// moments[0] =         1
	// moments[1] = (x-xCent)
	// moments[2] = (y-yCent)
	// moments[3] = (z-zCent)
	// moments[4] = (x-xCent)^2
	// moments[5] = (x-xCent)*(y-yCent)
	// moments[6] = (x-xCent)*(z-zCent)
	// moments[7] = (y-yCent)^2
	// moments[8] = (y-yCent)*(z-zCent)
	// moments[9] = (z-zCent)^2

    void createMomentMatchedNpole(SCC::GridFunction3d& V)
    {
   	std::vector <double > B(momentCount,0.0);
    getMoments3d(xPos,yPos,zPos, V, B);

    double sourceIntegral  = screenedSoln.evaluateSourceIntegral();
    double source2ndMoment = screenedSoln.evaluateSourceXr2Integral();

    B[0] *= 1.0/sourceIntegral;

    if(momentCount >= 4)
    {
    B[1]   *= -1.0/sourceIntegral;
    B[2]   *= -1.0/sourceIntegral;
    B[3]   *= -1.0/sourceIntegral;
    }
    if(momentCount >= 10)
    {
    B[4]    = (B[4] - source2ndMoment*B[0])/(2.0*sourceIntegral);
    B[5]   *= 1.0/sourceIntegral;
    B[6]   *= 1.0/sourceIntegral;
    B[7]    = (B[7] - source2ndMoment*B[0])/(2.0*sourceIntegral);
    B[8]   *= 1.0/sourceIntegral;
    B[9]    = (B[9] - source2ndMoment*B[0])/(2.0*sourceIntegral);
    }

    for(long i = 0; i < (long)str.size();    i++)  {str[i] = 0.0;}
    for(long i = 0; i < momentCount;   i++)  {str[i] = B[i];}
    }

    void setMoments(std::vector <double> B)
    {
    double sourceIntegral  = screenedSoln.evaluateSourceIntegral();
    double source2ndMoment = screenedSoln.evaluateSourceXr2Integral();

    B[0] *= 1.0/sourceIntegral;

    if(momentCount >= 4)
    {
    B[1]   *= -1.0/sourceIntegral;
    B[2]   *= -1.0/sourceIntegral;
    B[3]   *= -1.0/sourceIntegral;
    }
    if(momentCount >= 10)
    {
    B[4]    = (B[4] - source2ndMoment*B[0])/(2.0*sourceIntegral);
    B[5]   *= 1.0/sourceIntegral;
    B[6]   *= 1.0/sourceIntegral;
    B[7]    = (B[7] - source2ndMoment*B[0])/(2.0*sourceIntegral);
    B[8]   *= 1.0/sourceIntegral;
    B[9]    = (B[9] - source2ndMoment*B[0])/(2.0*sourceIntegral);
    }

    for(long i = 0; i < (long)str.size();    i++)  {str[i] = 0.0;}
    for(long i = 0; i < momentCount;   i++)  {str[i] = B[i];}
    }

	long       momentCount;
	double          xPos;
	double          yPos;
	double          zPos;
	double        radius;
	double  laplaceCoeff;
	double   screenCoeff;

	int differentiabilityOrder;

	std::vector<double>                str;
	ScreenedSoln3d        screenedSoln;

void createSource(SCC::GridFunction3d& V) const
{
    std::vector <double> sourceFun(momentCount,0.0);

    int maxOrder;
	switch (momentCount)
   	{
   	case 1  :  maxOrder = 0; break;
   	case 4  :  maxOrder = 1; break;
   	case 10 :  maxOrder = 2; break;
   	default :  maxOrder = 2; break;
   	}

    #ifdef _OPENMP
    int threadIndex;
    int threadCount = omp_get_max_threads();
    std::vector< ScreenedSoln3d  > screenedSolnArray(threadCount);
    std::vector< std::vector <double> >       sourceFunArray;

    sourceFunArray.resize(threadCount);

    for(threadIndex = 0; threadIndex < threadCount; threadIndex++)
    {
    screenedSolnArray[threadIndex].initialize(screenedSoln);
    sourceFunArray[threadIndex].resize(momentCount,0.0);
    }
    #endif

    double xMin  = V.getXmin();
    double yMin  = V.getYmin();
    double zMin  = V.getZmin();

    long xPanels = V.getXpanelCount();
    long yPanels = V.getYpanelCount();
    long zPanels = V.getZpanelCount();

    double  hx   = V.getHx();
    double  hy   = V.getHy();
    double  hz   = V.getHz();

    double x;  double y; double z;

#ifndef _OPENMP
    for(long i = 0; i <= xPanels; i++)
    {
    x = xMin + i*hx;
    for(long j = 0; j <= yPanels; j++)
    {
    y = yMin + j*hy;
    for(long k = 0; k <= zPanels; k++)
    {
    z = zMin + k*hz;
    screenedSoln.sourceDerivatives(x,y,z,sourceFun,maxOrder);
    V.Values(i,j,k) = sourceFun[0]*str[0];
    for(long index = 1; index < momentCount; index++)
    {
    V.Values(i,j,k) += sourceFun[index]*str[index];
    }
    }}}
    return;
#endif


	#ifdef _OPENMP
#pragma omp parallel for  \
private(x,y,z,threadIndex)\
schedule(static,1)
    for(long i = 0; i <= xPanels; i++)
    {
    threadIndex = omp_get_thread_num();
    x = xMin + i*hx;
    for(long j = 0; j <= yPanels; j++)
    {
    y = yMin + j*hy;
    for(long k = 0; k <= zPanels; k++)
    {
    z = zMin + k*hz;
    screenedSolnArray[threadIndex].sourceDerivatives(x,y,z,sourceFunArray[threadIndex],maxOrder);
    V.Values(i,j,k) = sourceFunArray[threadIndex][0]*str[0];
    for(long index = 1; index < momentCount; index++)
    {
    V.Values(i,j,k) += sourceFunArray[threadIndex][index]*str[index];
    }
    }}}
 #endif
}
void createPotential(SCC::GridFunction3d& V) const
{
    std::vector <double>    potentialFun(momentCount,0.0);

    int maxOrder;
	switch (momentCount)
   	{
   	case 1  :  maxOrder = 0; break;
   	case 4  :  maxOrder = 1; break;
   	case 10 :  maxOrder = 2; break;
   	default :  maxOrder = 2; break;
   	}

    #ifdef _OPENMP
    int threadIndex;
    int threadCount = omp_get_max_threads();
    std::vector < ScreenedSoln3d  >  screenedSolnArray(threadCount);
    std::vector< std::vector <double> >         potentialFunArray;

    potentialFunArray.resize(threadCount);
    for(threadIndex = 0; threadIndex < threadCount; threadIndex++)
    {
    screenedSolnArray[threadIndex].initialize(screenedSoln);
    potentialFunArray[threadIndex].resize(momentCount,0.0);
    }
    #endif

    double xMin  = V.getXmin();
    double yMin  = V.getYmin();
    double zMin  = V.getZmin();

    long xPanels = V.getXpanelCount();
    long yPanels = V.getYpanelCount();
    long zPanels = V.getZpanelCount();

    double  hx   = V.getHx();
    double  hy   = V.getHy();
    double  hz   = V.getHz();

    double x;  double y; double z;

#ifndef _OPENMP
    for(long i = 0; i <= xPanels; i++)
    {
    x = xMin + i*hx;
    for(long j = 0; j <= yPanels; j++)
    {
    y = yMin + j*hy;
    for(long k = 0; k <= zPanels; k++)
    {
    z = zMin + k*hz;
    screenedSoln.derivatives(x,y,z,potentialFun,maxOrder);
    V.Values(i,j,k) = potentialFun[0]*str[0];
    for(long index = 1; index < momentCount; index++)
    {
    V.Values(i,j,k) += potentialFun[index]*str[index];
    }
    }}}
    return;
#endif


	#ifdef _OPENMP
#pragma omp parallel for  \
private(x,y,z,threadIndex)\
schedule(static,1)
    for(long i = 0; i <= xPanels; i++)
    {
    threadIndex = omp_get_thread_num();
    x = xMin + i*hx;
    for(long j = 0; j <= yPanels; j++)
    {
    y = yMin + j*hy;
    for(long k = 0; k <= zPanels; k++)
    {
    z = zMin + k*hz;
    screenedSolnArray[threadIndex].derivatives(x,y,z,potentialFunArray[threadIndex],maxOrder);
    V.Values(i,j,k) = potentialFunArray[threadIndex][0]*str[0];
    for(long index = 1; index < momentCount; index++)
    {
    V.Values(i,j,k) += potentialFunArray[threadIndex][index]*str[index];
    }
    }}}
 #endif
}


//ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ

void createSource(SCC::GridFunction3d& V, int momentIndex)
{
    std::vector <double> sourceFun(momentCount,0.0);

    int maxOrder;
	switch (momentCount)
   	{
   	case 1  :  maxOrder = 0; break;
   	case 4  :  maxOrder = 1; break;
   	case 10 :  maxOrder = 2; break;
   	default :  maxOrder = 2; break;
   	}

    #ifdef _OPENMP
    int threadIndex;
    int threadCount = omp_get_max_threads();

    std::vector< ScreenedSoln3d  > screenedSolnArray(threadCount);
    std::vector< std::vector <double> > sourceFunArray;

    sourceFunArray.resize(threadCount);

    for(threadIndex = 0; threadIndex < threadCount; threadIndex++)
    {
    screenedSolnArray[threadIndex].initialize(screenedSoln);
    sourceFunArray[threadIndex].resize(momentCount,0.0);
    }
    #endif

    double xMin  = V.getXmin();
    double yMin  = V.getYmin();
    double zMin  = V.getZmin();

    long xPanels = V.getXpanelCount();
    long yPanels = V.getYpanelCount();
    long zPanels = V.getZpanelCount();

    double  hx   = V.getHx();
    double  hy   = V.getHy();
    double  hz   = V.getHz();

    double x;  double y; double z;

#ifndef _OPENMP
    for(long i = 0; i <= xPanels; i++)
    {
    x = xMin + i*hx;
    for(long j = 0; j <= yPanels; j++)
    {
    y = yMin + j*hy;
    for(long k = 0; k <= zPanels; k++)
    {
    z = zMin + k*hz;
    screenedSoln.sourceDerivatives(x,y,z,sourceFun,maxOrder);
    V.Values(i,j,k) = sourceFun[momentIndex];
    }}}
    return;
#endif


	#ifdef _OPENMP
#pragma omp parallel for  \
private(x,y,z,threadIndex)\
schedule(static,1)
    for(long i = 0; i <= xPanels; i++)
    {
    threadIndex = omp_get_thread_num();
    x = xMin + i*hx;
    for(long j = 0; j <= yPanels; j++)
    {
    y = yMin + j*hy;
    for(long k = 0; k <= zPanels; k++)
    {
    z = zMin + k*hz;
    screenedSolnArray[threadIndex].sourceDerivatives(x,y,z,sourceFunArray[threadIndex],maxOrder);
    V.Values(i,j,k) = sourceFunArray[threadIndex][momentIndex];
    }}}
 #endif
}
void createDiscreteSourceMoments(long xPanel, double xMin, double xMax, long yPanel, double yMin, double yMax, long zPanel, double zMin,double zMax)
{
	SCC::GridFunction3d src(xPanel,xMin,xMax,yPanel,yMin,yMax,zPanel,zMin,zMax);
	std::vector<double> moments;
	for(long i = 0; i < momentCount; i++)
	{
	createSource(src, i);
	getMoments3d(xPos,  yPos, zPos, src,moments);
	printf("\n\n");
	for(long k = 0; k < (long)moments.size(); k++)
	{
	printf("Source %ld - Moment %ld : %10.5e \n",i,k,moments[k]);
	}
	printf("\n\n");
	}
}

void  getMoments3d(double xCent, double yCent, double zCent, SCC::GridFunction3d& F,std::vector<double>& moments)
{
    moments.clear();
    moments.resize(momentCount,0.0);

    #ifdef _OPENMP
    std::vector< std::vector < double > >  momentArray;
    int threadIndex;
    int threadCount = omp_get_max_threads();
    momentArray.resize(threadCount,moments);
    #endif

    double xMin  = F.getXmin();
    double yMin  = F.getYmin();
    double zMin  = F.getZmin();

    long xPanels = F.getXpanelCount();
    long yPanels = F.getYpanelCount();
    long zPanels = F.getZpanelCount();

    double  hx   = F.getHx();
    double  hy   = F.getHy();
    double  hz   = F.getHz();

    double x;  double y;  double z; double hxhyhz;
    double xp; double yp; double zp;

    hxhyhz = hx*hy*hz;
    double fVal;

#ifndef _OPENMP
    for(long i = 0; i <= xPanels; i++)
    {
    x = xMin + i*hx;
    for(long j = 0; j <= yPanels; j++)
    {
    y = yMin + j*hy;
    for(long k = 0; k <= zPanels; k++)
    {
    z = zMin + k*hz;

    fVal =   F.Values(i,j,k);

    xp = (x-xCent);
    yp = (y-yCent);
    zp = (z-zCent);

    if(momentCount >= 1)
    {
    moments[0] += fVal*hxhyhz;
    }
    if(momentCount >= 4)
    {
    moments[1] += fVal*hxhyhz*xp;
    moments[2] += fVal*hxhyhz*yp;
    moments[3] += fVal*hxhyhz*zp;
    }
    if(momentCount >= 10)
    {
    moments[4] += fVal*hxhyhz*xp*xp;
    moments[5] += fVal*hxhyhz*xp*yp;
    moments[6] += fVal*hxhyhz*xp*zp;
    moments[7] += fVal*hxhyhz*yp*yp;
    moments[8] += fVal*hxhyhz*yp*zp;
    moments[9] += fVal*hxhyhz*zp*zp;
    }
    }}}

    return;
#endif


	#ifdef _OPENMP
#pragma omp parallel for  \
private(fVal,xp,yp,zp,x,y,z,threadIndex)\
schedule(static,1)
    for(long i = 0; i <= xPanels; i++)
    {
    threadIndex = omp_get_thread_num();
    x = xMin + i*hx;
    for(long j = 0; j <= yPanels; j++)
    {
    y = yMin + j*hy;
    for(long k = 0; k <= zPanels; k++)
    {
    z = zMin + k*hz;
    fVal =  F.Values(i,j,k);

    xp = (x-xCent);
    yp = (y-yCent);
    zp = (z-zCent);

    if(momentCount >= 1)
    {
    momentArray[threadIndex][0] += fVal*hxhyhz;
    }
    if(momentCount >= 4)
    {
    momentArray[threadIndex][1] +=  fVal*hxhyhz*xp;
    momentArray[threadIndex][2] +=  fVal*hxhyhz*yp;
    momentArray[threadIndex][3] +=  fVal*hxhyhz*zp;
    }
    if(momentCount >= 10)
    {
    momentArray[threadIndex][4] += fVal*hxhyhz*xp*xp;
    momentArray[threadIndex][5] += fVal*hxhyhz*xp*yp;
    momentArray[threadIndex][6] += fVal*hxhyhz*xp*zp;
    momentArray[threadIndex][7] += fVal*hxhyhz*yp*yp;
    momentArray[threadIndex][8] += fVal*hxhyhz*yp*zp;
    momentArray[threadIndex][9] += fVal*hxhyhz*zp*zp;
    }
    }}}

    // Accumulate moment contributions from each thread
    //
    for(threadIndex = 0; threadIndex < threadCount; threadIndex++)
    {
    for(long i     = 0; i     < momentCount; i++)
    {
    moments[i] += momentArray[threadIndex][i];
    }
    }
 #endif
}

};
#undef DEFAULT_DIFFERENTIABILITY_
#endif /* NPOLE2D_H_ */
