/*
 * PoissonNpole3d.h
 *
 *  A class for creating and caching smooth moment matching functions.
 *
 *  The moment matching functions is constructed using
 *  a linear combination of a SmoothPolyMollifier3d and
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

#include "MollifierNd/SmoothPolyPotential3d.h"
#include "MollifierNd/SmoothPolyMollifier3d.h"

#include <cmath>
#include <cstdio>
#include <vector>
#include <algorithm>


#ifndef POISSON_N_POLE_3D_
#define pOISSON_N_POLE_3D_

#ifdef _OPENMP
#include <omp.h>
#endif


#define  _DEFAULT_DIFFERENTIABILITY_ 6
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

class PoissonNpole3d
{
    public:

	PoissonNpole3d()
	{
	this->differentiabilityOrder = _DEFAULT_DIFFERENTIABILITY_;
	initialize();
	}


	// Creates an PoissonNpole3d used to match moments with multi-indices from 0 to maxOrder

	PoissonNpole3d(double xPos, double yPos, double zPos, double radius, int maxOrder, double laplaceCoeff = 1.0)
	{
	this->differentiabilityOrder = _DEFAULT_DIFFERENTIABILITY_;
	initialize(xPos,yPos,zPos,radius,maxOrder,laplaceCoeff);
	}

    // Creates an PoissonNpole3d used to to match moments with multi-indices from 0 to maxOrder
    // and with strengths given by the values in the str array of values
    /*
	PoissonNpole3d(double xPos, double yPos, double zPos, double radius,std::vector<double>& str, int maxOrder, double laplaceCoeff = 1.0)
	{
	this->differentiabilityOrder = _DEFAULT_DIFFERENTIABILITY_;
	initialize(xPos,yPos,zPos,radius,str,maxOrder,laplaceCoeff);
	}
    */

	void setDifferentiability(int diffOrder)
	{
		this->differentiabilityOrder = diffOrder;
		this->polyMollifier.setDifferentiability(diffOrder);
		this->polyPotential.setSourceDifferentiability(diffOrder);
	}

	int getDifferentiablity()
	{
	return this->differentiabilityOrder;
	}

    void setLaplaceCoefficient(double laplaceCoeff)
	{
	this->laplaceCoeff = laplaceCoeff;
	polyPotential.setLaplaceCoefficient(laplaceCoeff);
	}

	double getLaplaceCoefficient() const
	{
	return this->laplaceCoeff;
	}

	void initialize()
	{
	    polyMollifier.initialize();
	    polyPotential.initialize();
		momentCount    = 0;
		radius         = 0;
		laplaceCoeff   = 1.0;
		xPos           = 0.0;
		yPos           = 0.0;
	    zPos           = 0.0;
	}


	void initialize(double xPos, double yPos, double zPos, double radius, int maxOrder, double laplaceCoeff = 1.0)
	{
	switch (maxOrder)
   	{
   	case 0  :  this->momentCount = 1; break;
   	case 1  :  this->momentCount = 4; break;
   	case 2  :  this->momentCount = 10; break;
   	default :  this->momentCount = 10; break;
   	}

   	str.clear();
	str.resize(momentCount, 0.0);

	this->xPos          = xPos;
	this->yPos          = yPos;
	this->zPos          = zPos;
	this->radius        = radius;
	this->laplaceCoeff  = laplaceCoeff;
    this->differentiabilityOrder =  _DEFAULT_DIFFERENTIABILITY_;


    polyMollifier.initialize(xPos, yPos, zPos, radius, 1.0);
	polyMollifier.setDifferentiability(differentiabilityOrder);

    polyPotential.initialize(xPos, yPos, zPos, radius, 1.0, laplaceCoeff);
	polyPotential.setSourceDifferentiability(differentiabilityOrder);

	//initialize(xPos,yPos,zPos,radius,str,maxOrder,laplaceCoeff);
	}

    /*
	void initialize(double xPos, double yPos, double zPos, double radius,
	std::vector<double>& str, int maxOrder, double laplaceCoeff = 1.0)
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
    this->differentiabilityOrder =  _DEFAULT_DIFFERENTIABILITY_;

    // Capture any input strengths

    this->str.clear();
    this->str.resize(momentCount,0.0);
    for(long i = 0; i < std::min(momentCount,(long)(str.size())); i++) {this->str[i] = str[i];}

    // Now set moments so that specified number match input strengths
    setMoments(this->str);

    polyMollifier.initialize(xPos, yPos, zPos, radius, 1.0);
	polyMollifier.setDifferentiability(differentiabilityOrder);

    polyPotential.initialize(xPos, yPos, zPos, radius, 1.0, laplaceCoeff);
	polyPotential.setSourceDifferentiability(differentiabilityOrder);
    }
    */

	virtual ~PoissonNpole3d()
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

    polyPotential.derivatives(x,y,z,dList,maxOrder);
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

    polyMollifier.derivatives(x,y,z,dList,maxOrder);

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
    long i;
    for(i = 0; i < (long)this-> str.size(); i++)
    {
    	this->str[i] = str[i];
    }
    }

	// Moment indexing:
	//
	// moments[0] =         1            F
	// moments[1] = (x-xCent)            F
	// moments[2] = (y-yCent)            F
	// moments[3] = (z-zCent)            F
	// moments[4] = (x-xCent)^2          F
	// moments[5] = (x-xCent)*(y-yCent)  F
	// moments[6] = (x-xCent)*(z-zCent)  F
	// moments[7] = (y-yCent)^2          F
	// moments[8] = (y-yCent)*(z-zCent)  F
	// moments[9] = (z-zCent)^2          F

    void createMomentMatchedNpole(SCC::GridFunction3d& V)
    {
	std::vector <double > B(momentCount,0.0);
    getMoments3d(xPos,yPos,zPos, V, B);

    double offDval = (radius*radius)/(2.0*(this->differentiabilityOrder + 3.0) + 1);
    if(momentCount >= 4)
    {
    B[1]   *= -1.0;
    B[2]   *= -1.0;
    B[3]   *= -1.0;
    }
    if(momentCount >= 10)
    {
    B[4]    = (B[4] - offDval*B[0])/2.0;
    B[7]    = (B[7] - offDval*B[0])/2.0;
    B[9]    = (B[9] - offDval*B[0])/2.0;
    }

    for(long i = 0; i < (long)str.size();    i++)  {str[i] = 0.0;}
    for(long i = 0; i < momentCount;   i++)  {str[i] = B[i];}
    }

    void setMoments(std::vector <double> B)
    {
    double offDval = (radius*radius)/(2.0*(this->differentiabilityOrder + 3.0) + 1);
    if(momentCount >= 4)
    {
    B[1]   *= -1.0;
    B[2]   *= -1.0;
    B[3]   *= -1.0;
    }
    if(momentCount >= 10)
    {
    B[4]    = (B[4] - offDval*B[0])/2.0;
    B[7]    = (B[7] - offDval*B[0])/2.0;
    B[9]    = (B[9] - offDval*B[0])/2.0;
    }

    for(long i = 0; i < (long)str.size(); i++)  {str[i] = 0.0;}
    for(long i = 0; i < momentCount;   i++)  {str[i] = B[i];}
    }

	long       momentCount;
	double          xPos;
	double          yPos;
	double          zPos;
	double        radius;
	double  laplaceCoeff;
	int differentiabilityOrder;

	std::vector<double>                str;
	bool             discreteCacheFlag;

	SmoothPolyPotential3d   polyPotential;
    SmoothPolyMollifier3d   polyMollifier;

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
    std::vector< SmoothPolyMollifier3d  > polyMollifierArray(threadCount);
    std::vector< std::vector <double> >       sourceFunArray;

    sourceFunArray.resize(threadCount);

    for(threadIndex = 0; threadIndex < threadCount; threadIndex++)
    {
    polyMollifierArray[threadIndex].initialize(polyMollifier);
    sourceFunArray[threadIndex].resize(momentCount,0.0);
    }
    #endif


    long i; long j; long k;
    long index;

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
    for(i = 0; i <= xPanels; i++)
    {
    x = xMin + i*hx;
    for(j = 0; j <= yPanels; j++)
    {
    y = yMin + j*hy;
    for(k = 0; k <= zPanels; k++)
    {
    z = zMin + k*hz;
    polyMollifier.derivatives(x,y,z,sourceFun,maxOrder);
    V.Values(i,j,k) = sourceFun[0]*str[0];
    for(index = 1; index < momentCount; index++)
    {
    V.Values(i,j,k) += sourceFun[index]*str[index];
    }
    }}}
    return;
#endif


	#ifdef _OPENMP
#pragma omp parallel for  \
private(index,i,j,k,x,y,z,threadIndex)\
schedule(static,1)
    for(i = 0; i <= xPanels; i++)
    {
    threadIndex = omp_get_thread_num();
    x = xMin + i*hx;
    for(j = 0; j <= yPanels; j++)
    {
    y = yMin + j*hy;
    for(k = 0; k <= zPanels; k++)
    {
    z = zMin + k*hz;
    polyMollifierArray[threadIndex].derivatives(x,y,z,sourceFunArray[threadIndex],maxOrder);
    V.Values(i,j,k) = sourceFunArray[threadIndex][0]*str[0];
    for(index = 1; index < momentCount; index++)
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
    std::vector < SmoothPolyPotential3d  >  polyPotentialArray(threadCount);
    std::vector< std::vector <double> >         potentialFunArray;

    potentialFunArray.resize(threadCount);
    for(threadIndex = 0; threadIndex < threadCount; threadIndex++)
    {
    polyPotentialArray[threadIndex].initialize(polyPotential);
    potentialFunArray[threadIndex].resize(momentCount,0.0);
    }
    #endif


    long i; long j; long k;
    long index;

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
    for(i = 0; i <= xPanels; i++)
    {
    x = xMin + i*hx;
    for(j = 0; j <= yPanels; j++)
    {
    y = yMin + j*hy;
    for(k = 0; k <= zPanels; k++)
    {
    z = zMin + k*hz;
    polyPotential.derivatives(x,y,z,potentialFun,maxOrder);
    V.Values(i,j,k) = potentialFun[0]*str[0];
    for(index = 1; index < momentCount; index++)
    {
    V.Values(i,j,k) += potentialFun[index]*str[index];
    }
    }}}
    return;
#endif


	#ifdef _OPENMP
#pragma omp parallel for  \
private(index,i,j,k,x,y,z,threadIndex)\
schedule(static,1)
    for(i = 0; i <= xPanels; i++)
    {
    threadIndex = omp_get_thread_num();
    x = xMin + i*hx;
    for(j = 0; j <= yPanels; j++)
    {
    y = yMin + j*hy;
    for(k = 0; k <= zPanels; k++)
    {
    z = zMin + k*hz;
    polyPotentialArray[threadIndex].derivatives(x,y,z,potentialFunArray[threadIndex],maxOrder);
    V.Values(i,j,k) = potentialFunArray[threadIndex][0]*str[0];
    for(index = 1; index < momentCount; index++)
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

    std::vector< SmoothPolyMollifier3d  > polyMollifierArray(threadCount);
    std::vector< std::vector <double> >       sourceFunArray;

    sourceFunArray.resize(threadCount);

    for(threadIndex = 0; threadIndex < threadCount; threadIndex++)
    {
    polyMollifierArray[threadIndex].initialize(polyMollifier);
    sourceFunArray[threadIndex].resize(momentCount,0.0);
    }
    #endif


    long i; long j; long k;


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
    for(i = 0; i <= xPanels; i++)
    {
    x = xMin + i*hx;
    for(j = 0; j <= yPanels; j++)
    {
    y = yMin + j*hy;
    for(k = 0; k <= zPanels; k++)
    {
    z = zMin + k*hz;
    polyMollifier.derivatives(x,y,z,sourceFun,maxOrder);
    V.Values(i,j,k) = sourceFun[momentIndex];
    }}}
    return;
#endif


	#ifdef _OPENMP
#pragma omp parallel for  \
private(i,j,k,x,y,z,threadIndex)\
schedule(static,1)
    for(i = 0; i <= xPanels; i++)
    {
    threadIndex = omp_get_thread_num();
    x = xMin + i*hx;
    for(j = 0; j <= yPanels; j++)
    {
    y = yMin + j*hy;
    for(k = 0; k <= zPanels; k++)
    {
    z = zMin + k*hz;
    polyMollifierArray[threadIndex].derivatives(x,y,z,sourceFunArray[threadIndex],maxOrder);
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

    long i; long j; long k;

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
    for(i = 0; i <= xPanels; i++)
    {
    x = xMin + i*hx;
    for(j = 0; j <= yPanels; j++)
    {
    y = yMin + j*hy;
    for(k = 0; k <= zPanels; k++)
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
private(fVal,xp,yp,zp,i,j,k,x,y,z,threadIndex)\
schedule(static,1)
    for(i = 0; i <= xPanels; i++)
    {
    threadIndex = omp_get_thread_num();
    x = xMin + i*hx;
    for(j = 0; j <= yPanels; j++)
    {
    y = yMin + j*hy;
    for(k = 0; k <= zPanels; k++)
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
    for(i     = 0; i     < momentCount; i++)
    {
    moments[i] += momentArray[threadIndex][i];
    }
    }
 #endif
}

};
#undef _DEFAULT_DIFFERENTIABILITY_
#endif /* NPOLE2D_H_ */
