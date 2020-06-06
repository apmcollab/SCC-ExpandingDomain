/*
 * PoissonNpole2d.h
 *
 *  A class for creating and caching smooth moment matching functions.
 *
 *  The moment matching functions is constructed using
 *  a linear combination of a SmoothPolyMollifier2d and
 *  it's derivatives.
 *
 *
 *  Moments with multi-indices from 0 to maxOrder (= 0, 1, 2)
 *  are matched.
 *
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

#include "DoubleVectorNd/SCC_DoubleVector2d.h"
#include "GridFunctionNd/SCC_GridFunction2d.h"

#include "MollifierNd/SmoothPolyPotential2d.h"
#include "MollifierNd/SmoothPolyMollifier2d.h"

#include <cmath>
#include <cstdio>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef _PoissonNpole2d_
#define _PoissonNpole2d_


#define DEFAULT_DIFFERENTIABILITY_ 6

//
// moments[0] =         1
// moments[1] = (x-xCent)
// moments[2] = (y-yCent)
// moments[3] = (x-xCent)^2
// moments[4] = (x-xCent)*(y-yCent)
// moments[5] = (y-yCent)^2
//

class PoissonNpole2d
{
    public:

	PoissonNpole2d()
	{
	this->momentCount   = 0;
	this->xPos          = 0;
	this->yPos          = 0;
	this->radius        = 0;
	this->laplaceCoeff  = 1.0;
	this->differentiabilityOrder = DEFAULT_DIFFERENTIABILITY_;
	initialize();
	}

	// Creates a PoissonNpole2d used to match moments with multi-indices from 0 to maxOrder

	PoissonNpole2d(double xPos, double yPos, double radius, int maxOrder, double laplaceCoeff = 1.0)
	{
	switch (maxOrder)
   	{
   	case 0  :  this->momentCount = 1; break;
   	case 1  :  this->momentCount = 3; break;
   	case 2  :  this->momentCount = 6; break;
   	default :  this->momentCount = 6; break;
   	}

	this->differentiabilityOrder = DEFAULT_DIFFERENTIABILITY_;
	initialize(xPos,yPos,radius,maxOrder,laplaceCoeff);
	}

    // Creates a PoissonNpole2d instance used to to match moments with multi-indices from 0 to maxOrder
    // and with strengths given by the values in the str array of values

	PoissonNpole2d(double xPos, double yPos, double radius, std::vector<double>& str, double laplaceCoeff = 1.0)
	{
	this->momentCount = str.size();
	if(this->momentCount > 6) this->momentCount = 6;

	int maxOrder;
	switch (momentCount)
   	{
   	case 1  :  maxOrder = 0; break;
   	case 3  :  maxOrder = 1; break;
   	case 6  :  maxOrder = 2; break;
   	default :  maxOrder = 2; break;
   	}

	this->differentiabilityOrder =  DEFAULT_DIFFERENTIABILITY_;
	initialize(xPos,yPos,radius,str,maxOrder,laplaceCoeff);
	}

	void setDifferentiability(int diffOrder)
	{
		this->differentiabilityOrder = diffOrder;
		polyMollifier.setDifferentiability(diffOrder);
		polyPotential.setSourceDifferentiability(diffOrder);
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
		xPos           = 0.0;
		yPos           = 0.0;
		laplaceCoeff   = 1.0;
		differentiabilityOrder =  DEFAULT_DIFFERENTIABILITY_;
	}

	void initialize(double xPos, double yPos, double radius,int maxOrder, double laplaceCoeff = 1.0)
	{
    switch (maxOrder)
   	{
   	case 0  :  this->momentCount = 1; break;
   	case 1  :  this->momentCount = 3; break;
   	case 2  :  this->momentCount = 6; break;
   	default :  this->momentCount = 6; break;
   	}

	std::vector<double> str(momentCount,0.0);
	initialize(xPos,yPos,radius,str,maxOrder,laplaceCoeff);
	}

	void initialize(double xPos, double yPos, double radius, std::vector<double>& str,int maxOrder, double laplaceCoeff = 1.0)
	{
    switch (maxOrder)
   	{
   	case 0  :  this->momentCount = 1; break;
   	case 1  :  this->momentCount = 3; break;
   	case 2  :  this->momentCount = 6; break;
   	default :  this->momentCount = 6; break;
   	}

	this->xPos                   = xPos;
	this->yPos                   = yPos;
	this->radius                 = radius;
    this->differentiabilityOrder = DEFAULT_DIFFERENTIABILITY_;

    // Capture specified moments

    this->str.clear();
    this->str.resize(momentCount,0.0);
    for(long i = 0; i < std::min(momentCount,(long)(str.size())); i++)
    {
    this->str[i] = str[i];
    }
    // Now set moments so that specified number match input strengths

    setMoments(this->str);

	// Initialize

	polyMollifier.initialize(xPos, yPos, radius, 1.0);
    polyPotential.initialize(xPos, yPos, radius, 1.0,laplaceCoeff);
	setDifferentiability(differentiabilityOrder);
    }

	virtual ~PoissonNpole2d()
	{}


    double operator()(double x, double y) const
    {
    return evaluatePotential(x,y);
    }


    void evaluatePotential(SCC::GridFunction2d& V) const
    {
    if(momentCount == 0) return;
    createPotential(V);
    return;
    }

    void evaluateSource(SCC::GridFunction2d& V) const
    {
    if(momentCount == 0) return;
    createSource(V);
    return;
    }

    double evaluatePotential(double x, double y) const
    {

    int maxOrder;
	switch (momentCount)
   	{
   	case 1  :  maxOrder = 0; break;
   	case 3  :  maxOrder = 1; break;
   	case 6  :  maxOrder = 2; break;
   	default :  maxOrder = 2; break;
   	}

   	std::vector<double> dList(momentCount);

    polyPotential.derivatives(x,y,dList,maxOrder);

    double potValue = 0.0;

    for(long i = 0; i < momentCount; i++)
    {
    	potValue += str[i]*dList[i];
    }
    return potValue;
    }

    double evaluateSource(double x, double y) const
    {
    int maxOrder;
	switch (momentCount)
   	{
   	case 1  :  maxOrder = 0; break;
   	case 3  :  maxOrder = 1; break;
   	case 6  :  maxOrder = 2; break;
   	default :  maxOrder = 2; break;
   	}

   	std::vector<double> dList(momentCount);

    polyMollifier.derivatives(x,y,dList,maxOrder);

    double srcValue = 0.0;
    long   i;
    for(i = 0; i < momentCount; i++)
    {
    	srcValue += str[i]*dList[i];
    }
    return srcValue;
    }


//  Returns a std::function that is bound to the potential evaluation operator of *this

#if __cplusplus > 199711L
	std::function<double(double,double)> getPotentialEvaluationPtr() const
	{
	std::function<double(double,double)> F = [this](double x,double y) {return this->evaluatePotential(x,y);};
	return std::move(F);
	}
#endif

//  Returns a std::function that is bound to the source evaluation operator of *this

#if __cplusplus > 199711L
	std::function<double(double,double)> getSourceEvaluationPtr() const
	{
	std::function<double(double,double)> F = [this](double x,double y) {return this->evaluateSource(x,y);};
	return std::move(F);
	}
#endif


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

    //
    // moments[0] =         1            F
	// moments[1] = (x-xCent)            F
	// moments[2] = (y-yCent)            F
	// moments[3] = (x-xCent)^2          F
	// moments[4] = (x-xCent)*(y-yCent)  F
	// moments[5] = (y-yCent)^2          F
	//
	//

    void createMomentMatchedNpole(SCC::GridFunction2d& V)
    {
	std::vector <double > B(momentCount,0.0);

    getMoments2d(xPos,yPos, V, B);

    double offDval = (radius*radius)/(2.0*(this->differentiabilityOrder + 3.0));

    if(momentCount >= 3)
    {
    B[1]   *= -1.0;
    B[2]   *= -1.0;
    }
    if(momentCount >= 6)
    {
    B[3]    = (B[3] - offDval*B[0])/2.0;
    B[5]    = (B[5] - offDval*B[0])/2.0;
    }

    for(long i = 0; i < (long)str.size();    i++)  {str[i] = 0.0;}
    for(long i = 0; i < momentCount;   i++)  {str[i] = B[i];}
    }

    void setMoments(std::vector <double > B)
    {
    double offDval = (radius*radius)/(2.0*(this->differentiabilityOrder + 3.0));

    if(momentCount >= 3)
    {
    B[1]   *= -1.0;
    B[2]   *= -1.0;
    }
    if(momentCount >= 6)
    {
    B[3]    = (B[3] - offDval*B[0])/2.0;
    B[5]    = (B[5] - offDval*B[0])/2.0;
    }

    for(long i = 0; i < (long)str.size();    i++)  {str[i] = 0.0;}
    for(long i = 0; i < momentCount;   i++)  {str[i] = B[i];}
    }

	long      momentCount;
	double           xPos;
	double           yPos;
	double         radius;
	double   laplaceCoeff;
	int differentiabilityOrder;

	std::vector<double>   str;

	bool   discreteCacheFlag;

	SmoothPolyPotential2d               polyPotential;
	SmoothPolyMollifier2d               polyMollifier;
//
// This routine computes the values of the correction
// sources and potentials to be cached from
//
void createSource(SCC::GridFunction2d& V) const
{

    std::vector <double> sourceFun(momentCount,0.0);

    int maxOrder;
	switch (momentCount)
   	{
   	case 1  :  maxOrder = 0; break;
   	case 3  :  maxOrder = 1; break;
   	case 6  :  maxOrder = 2; break;
   	default :  maxOrder = 2; break;
   	}

    #ifdef _OPENMP

    int threadIndex;
    int threadCount = omp_get_max_threads();

    std::vector< SmoothPolyMollifier2d  > polyMollifierArray(threadCount);
    std::vector< std::vector <double> >                        sourceFunArray;

    sourceFunArray.resize(threadCount);

    for(threadIndex = 0; threadIndex < threadCount; threadIndex++)
    {
    polyMollifierArray[threadIndex].initialize(polyMollifier);
    sourceFunArray[threadIndex].resize(momentCount,0.0);
    }
    #endif


    long i; long j;
    long index;

    double xMin  = V.getXmin();
    double yMin  = V.getYmin();


    long xPanels = V.getXpanelCount();
    long yPanels = V.getYpanelCount();


    double  hx   = V.getHx();
    double  hy   = V.getHy();

    double x;  double y;

#ifndef _OPENMP
    for(i = 0; i <= xPanels; i++)
    {
    x = xMin + i*hx;
    for(j = 0; j <= yPanels; j++)
    {
    y = yMin + j*hy;
    polyMollifier.derivatives(x,y,sourceFun,maxOrder);
    V.Values(i,j) = sourceFun[0]*str[0];
    for(index = 1; index < momentCount; index++)
    {
    V.Values(i,j) += sourceFun[index]*str[index];
    }
    }}
    return;
#endif


	#ifdef _OPENMP
#pragma omp parallel for  \
private(index,i,j,x,y,threadIndex)\
schedule(static,1)
    for(i = 0; i <= xPanels; i++)
    {
    threadIndex = omp_get_thread_num();
    x = xMin + i*hx;
    for(j = 0; j <= yPanels; j++)
    {
    y = yMin + j*hy;
    polyMollifierArray[threadIndex].derivatives(x,y,sourceFunArray[threadIndex],maxOrder);
    V.Values(i,j) = sourceFunArray[threadIndex][0]*str[0];
    for(index = 1; index < momentCount; index++)
    {
    V.Values(i,j) += sourceFunArray[threadIndex][index]*str[index];
    }
    }}
 #endif
}
void createPotential(SCC::GridFunction2d& V) const
{
    std::vector <double>    potentialFun(momentCount,0.0);

 	int maxOrder;
	switch (momentCount)
   	{
   	case 1  :  maxOrder = 0; break;
   	case 3  :  maxOrder = 1; break;
   	case 6  :  maxOrder = 2; break;
   	default :  maxOrder = 2; break;
   	}

    #ifdef _OPENMP
    int threadIndex;
    int threadCount = omp_get_max_threads();

    std::vector< SmoothPolyPotential2d  >  polyPotentialArray(threadCount);
    std::vector< std::vector <double> > potentialFunArray;

    potentialFunArray.resize(threadCount);
    for(threadIndex = 0; threadIndex < threadCount; threadIndex++)
    {
    polyPotentialArray[threadIndex].initialize(polyPotential);
    potentialFunArray[threadIndex].resize(momentCount,0.0);
    }
    #endif


    long i; long j;
    long index;

    double xMin  = V.getXmin();
    double yMin  = V.getYmin();


    long xPanels = V.getXpanelCount();
    long yPanels = V.getYpanelCount();


    double  hx   = V.getHx();
    double  hy   = V.getHy();

    double x;  double y;

#ifndef _OPENMP
    for(i = 0; i <= xPanels; i++)
    {
    x = xMin + i*hx;
    for(j = 0; j <= yPanels; j++)
    {
    y = yMin + j*hy;
    polyPotential.derivatives(x,y,potentialFun,maxOrder);
    V.Values(i,j) = potentialFun[0]*str[0];
    for(index = 1; index < momentCount; index++)
    {
    V.Values(i,j) += potentialFun[index]*str[index];
    }
    }}
    return;
#endif


	#ifdef _OPENMP
#pragma omp parallel for  \
private(index,i,j,x,y,threadIndex)\
schedule(static,1)
    for(i = 0; i <= xPanels; i++)
    {
    threadIndex = omp_get_thread_num();
    x = xMin + i*hx;
    for(j = 0; j <= yPanels; j++)
    {
    y = yMin + j*hy;
    polyPotentialArray[threadIndex].derivatives(x,y,potentialFunArray[threadIndex],maxOrder);
    V.Values(i,j) = potentialFunArray[threadIndex][0]*str[0];
    for(index = 1; index < momentCount; index++)
    {
    V.Values(i,j) += potentialFunArray[threadIndex][index]*str[index];
    }
    }}
 #endif
}

void  getMoments2d(double xCent, double yCent, SCC::GridFunction2d& F,std::vector<double>& moments)
{
    moments.clear();
    moments.resize(momentCount,0.0);

    #ifdef _OPENMP
    std::vector< std::vector < double > >  momentArray;
    int threadIndex;
    int threadCount = omp_get_max_threads();
    momentArray.resize(threadCount,moments);
    #endif

    long i; long j;

    double xMin  = F.getXmin();
    double yMin  = F.getYmin();

    long xPanels = F.getXpanelCount();
    long yPanels = F.getYpanelCount();

    double  hx   = F.getHx();
    double  hy   = F.getHy();

    double x;  double y;  double hxhy;
    double xp; double yp;

    hxhy = hx*hy;
    double fVal;

#ifndef _OPENMP
    for(i = 0; i <= xPanels; i++)
    {
    x = xMin + i*hx;
    for(j = 0; j <= yPanels; j++)
    {
    y = yMin + j*hy;
    fVal =   F.Values(i,j);

    xp = (x-xCent);
    yp = (y-yCent);

    if(momentCount >= 1)
    {
    moments[0] += fVal*hxhy;
    }
    if(momentCount >= 3)
    {
    moments[1] += fVal*hxhy*xp;
    moments[2] += fVal*hxhy*yp;
    }
    if(momentCount >= 6)
    {
    moments[3] += fVal*hxhy*xp*xp;
    moments[4] += fVal*hxhy*xp*yp;
    moments[5] += fVal*hxhy*yp*yp;
    }
    }}

    return;
#endif


#ifdef _OPENMP
#pragma omp parallel for  \
private(fVal,xp,yp,i,j,x,y,threadIndex)\
schedule(static,1)
    for(i = 0; i <= xPanels; i++)
    {
    threadIndex = omp_get_thread_num();
    x = xMin + i*hx;
    for(j = 0; j <= yPanels; j++)
    {
    y = yMin + j*hy;
    fVal =  F.Values(i,j);
    xp = (x-xCent);
    yp = (y-yCent);

    if(momentCount >= 1)
    {
    momentArray[threadIndex][0] += fVal*hxhy;
    }
    if(momentCount >= 3)
    {
    momentArray[threadIndex][1] +=  fVal*hxhy*xp;
    momentArray[threadIndex][2] +=  fVal*hxhy*yp;
    }
    if(momentCount >= 6)
    {
    momentArray[threadIndex][3] += fVal*hxhy*xp*xp;
    momentArray[threadIndex][4] += fVal*hxhy*xp*yp;
    momentArray[threadIndex][5] += fVal*hxhy*yp*yp;
    }
    }}

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
#undef DEFAULT_DIFFERENTIABILITY_
#endif /* NPOLE2D_H_ */
