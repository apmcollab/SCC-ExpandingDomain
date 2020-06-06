/*
 * ScreenedSoln3d.h
 *
 *  Created on: Jul 10, 2015
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
#include <vector>
#include <cmath>
#include <iostream>


#include "MollifierNd/SmoothPolyStep.h"
#include "MollifierNd/SmoothPolyMollifier1d.h"

#ifndef SCREENED_SOLN_3D_
#define SCREENED_SOLN_3D_

#define DEFAULT_DIFFERENTIABILITY_ 6

//  Fundamental solution of
//
//   [laplaceCoeff*DELTA + screenCoeff]
//
//  -1/(4*pi*laplaceCoeff) * std::exp(-std::sqrt(|screenCoeff/laplaceCoeff|)*r)/r
//
//   screenCoeff*laplaceCoeff < 0
//
class ScreenedSoln3d
{
    public:

	ScreenedSoln3d() : s(),sPrime()
	{
    initialize();
	}

	ScreenedSoln3d(const ScreenedSoln3d& H)
	{
	initialize(H);
	}

	ScreenedSoln3d( double xPos,  double yPos, double zPos, double radius,double strength, double laplaceCoeff, double screenCoeff)
	{
	initialize(xPos,yPos,zPos, radius,strength, laplaceCoeff, screenCoeff);
	}

	void initialize(const ScreenedSoln3d& H)
	{
	xPos                   = H.xPos;
	yPos                   = H.yPos;
	zPos                   = H.zPos;
	radius                 = H.radius;
	strength               = H.strength;
	screenCoeff            = H.screenCoeff;
	laplaceCoeff           = H.laplaceCoeff;
	sqrtAOB                = H.sqrtAOB;
	differentiabilityOrder = H.differentiabilityOrder;
	s.initialize(H.s);
	sPrime.initialize(H.sPrime);
	}


	void initialize()
	{
	xPos             = 0.0;
	yPos             = 0.0;
	zPos             = 0.0;
	radius           = 0.0;
	strength         = 0.0;
	screenCoeff      = 0.0;
	laplaceCoeff     = 0.0;
	sqrtAOB          = 0.0;
	differentiabilityOrder = DEFAULT_DIFFERENTIABILITY_;
	}


	//  Step function used
	//                                  |       ----
	//                           |
	//                     |
	//                |
	//           |
	// --------
	// 0   .1*radius                             radius
	// |======|======================================|
	//
	void initialize(double xPos,
	double yPos, double zPos, double radius, double strength,double laplaceCoeff, double screenCoeff)
	{

    //
	//  A fatal error if the signs of screenCoeff and laplaceCoeff are the same.
	//
	if(std::abs(screenCoeff) > 1.0e-10)
	{
	if(laplaceCoeff*screenCoeff > 0)
	{
    printf("XXXX Error :  ScreenedSoln3d  XXXX \n");
    printf("Coefficients of the Helmholtz operator are of the same sign. \n\n");
    printf("XXXX Execution Halted XXXX\n");
    exit(0);
    }
    }

    this->xPos                   = xPos;
	this->yPos                   = yPos;
	this->zPos                   = zPos;
	this->radius                 = radius;
	this->strength               = strength;
	this->screenCoeff            = screenCoeff;
	this->laplaceCoeff           = laplaceCoeff;
	this->sqrtAOB                = std::sqrt(std::abs(screenCoeff/laplaceCoeff));
	this->differentiabilityOrder = DEFAULT_DIFFERENTIABILITY_;

	s.initialize(0.55*radius,0.9*radius,1.0);
	sPrime.initialize(s.getDerivative());

	setSourceDifferentiability(differentiabilityOrder);
	}

    void setCoefficients(double laplaceCoeff, double screenCoeff)
	{
	this->laplaceCoeff  = laplaceCoeff;
	this->screenCoeff   = screenCoeff;
	this->sqrtAOB       = std::sqrt(std::abs(screenCoeff/laplaceCoeff));
	}

	void getCoefficients(double& laplaceCoeff, double& screenCoeff) const
	{
	laplaceCoeff  = this->laplaceCoeff;
	screenCoeff   = this->screenCoeff;
	}

	void setSourceDifferentiability(int diffOrder)
	{
    this->differentiabilityOrder = diffOrder;
    this->s.setDifferentiability(this->differentiabilityOrder+2);
    this->sPrime.setDifferentiability(s.getDifferentiablity()-1);
	}

	int getSourceDifferentiablity()
	{
	return this->differentiabilityOrder;
	}


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

    void derivatives(double x, double y, double z, std::vector<double>& derivativeList, int maxOrder) const
    {
    double r2 = std::abs((x-xPos)*(x-xPos) + (y-yPos)*(y-yPos) + (z-zPos)*(z-zPos));
    double r  = std::sqrt(r2);

    switch (maxOrder)
   	{
   	case 0   :  derivativeList.resize(1); break;
   	case 1   :  derivativeList.resize(4);  break;
   	case 2   :  derivativeList.resize(10); break;
   	default  :  derivativeList.resize(10); break;
   	}

    for(int i= 0; i < (int)derivativeList.size(); i++) {derivativeList[i] = 0.0;}

    if(r < 0.1*radius)
    {return;}

    double expVal;
    double sVal;

	sVal              = s(r);
	expVal            = std::exp(-sqrtAOB*r);
	derivativeList[0] = -(strength/laplaceCoeff)*(.79577471545947667884e-01)*sVal*(expVal/r);

	if(maxOrder == 0) return;

	std::vector<double>      sDerivatives(2);
    sPrime.derivatives(r,sDerivatives,1);

    double dsVal     = sDerivatives[0];
    double d2sVal    = sDerivatives[1];

	double dRad      =  -(strength/laplaceCoeff)*(.79577471545947667884e-01)*
	                      (dsVal*(expVal/r) - sVal*((expVal/r)*sqrtAOB + (expVal/(r*r))));

	derivativeList[1] = dRad*((x-xPos)/r);
	derivativeList[2] = dRad*((y-yPos)/r);
    derivativeList[3] = dRad*((z-zPos)/r);

    if(maxOrder == 1) return;

    double d2Rad =
    -(strength/laplaceCoeff)*(.79577471545947667884e-01)*
    (std::exp(-sqrtAOB*r)/(r*r*r))*(d2sVal*r*r-2.0*dsVal*r*(sqrtAOB*r+1.0)+s(r)*(sqrtAOB*sqrtAOB*r*r+2.0*sqrtAOB*r+2.0));

	// (4) = x^2
	// (5) = xy
	// (6) = xz
	// (7) = y^2
	// (8) = yz
	// (9) = z^2

    double r3 = r2*r;

	derivativeList[4] = (d2Rad/r2)*((x-xPos)*(x-xPos)) + (dRad/r3)*((y-yPos)*(y-yPos) + (z-zPos)*(z-zPos));
	derivativeList[5] = ((d2Rad/r2) - (dRad/r3))*((x-xPos)*(y-yPos)) ;
	derivativeList[6] = ((d2Rad/r2) - (dRad/r3))*((x-xPos)*(z-zPos));
	derivativeList[7] = (d2Rad/r2)*((y-yPos)*(y-yPos)) + (dRad/r3)*((x-xPos)*(x-xPos) + (z-zPos)*(z-zPos));
	derivativeList[8] = ((d2Rad/r2) - (dRad/r3))*((y-yPos)*(z-zPos));
	derivativeList[9] = (d2Rad/r2)*((z-zPos)*(z-zPos)) + (dRad/r3)*((x-xPos)*(x-xPos) + (y-yPos)*(y-yPos));

    return;
    }


	double evaluatePotential(double x, double y, double z) const
	{
	// double OneOver4pi = .79577471545947667884e-01;
	double r = std::sqrt(std::abs((x-xPos)*(x-xPos) + (y-yPos)*(y-yPos) + (z-zPos)*(z-zPos)));
	if(r < 0.1*radius) return 0.0;
	return -(strength/laplaceCoeff)*(.79577471545947667884e-01)*s(r)*(std::exp(-sqrtAOB*r)/r);
	}

    void evaluatePotentialDerivatives(double x, double y, double z, double& dPx, double& dPy, double& dPz) const
	{
	int maxOrder = 1;
    std::vector<double>    derivativeList(4);
	derivatives(x, y, z, derivativeList, maxOrder);
	dPx = derivativeList[1];
	dPy = derivativeList[2];
	dPz = derivativeList[3];
	}


    double evaluateSource(double x, double y, double z) const
	{
	double r = std::sqrt(std::abs((x-xPos)*(x-xPos) + (y-yPos)*(y-yPos) + (z-zPos)*(z-zPos)));
	return evaluateRadialSource(r);
	}

	void evaluateSourceDerivatives(double x, double y, double z, double& dSx, double& dSy, double& dSz) const
	{
	std::vector<double> derivativeList(4);
	sourceDerivatives(x, y, z, derivativeList, 1);
	dSx = derivativeList[1];
	dSy = derivativeList[2];
	dSz = derivativeList[3];
	}

    void sourceDerivatives(double x, double y, double z, std::vector<double>& derivativeList, int maxOrder=2) const
    {

    double r2 = std::abs((x-xPos)*(x-xPos) + (y-yPos)*(y-yPos) + (z-zPos)*(z-zPos));
    double r  = std::sqrt(r2);

    switch (maxOrder)
   	{
   	case 0   :  derivativeList.resize(1);  break;
   	case 1   :  derivativeList.resize(4);  break;
   	case 2   :  derivativeList.resize(10); break;
   	default  :  derivativeList.resize(10); break;
   	}

    for(int i= 0; i < (int)derivativeList.size(); i++) {derivativeList[i] = 0.0;}

    if((r < 0.1*radius)||(r > radius))
    {
    return;
   	}

   	std::vector<double>      sDerivatives(4);
    sPrime.derivatives(r,sDerivatives,3);

    double dsVal     = sDerivatives[0];
    double d2sVal    = sDerivatives[1];
    double d3sVal    = sDerivatives[2];
    double d4sVal    = sDerivatives[3];

    double expVal    = std::exp(-sqrtAOB*r);

    derivativeList[0] = (-.79577471545947667884e-01*strength)*(expVal/r)*(d2sVal - 2.0*dsVal*sqrtAOB);

    if(maxOrder == 0) return;

    double dRad
    = (-.79577471545947667884e-01*strength)*(
      (d3sVal - 2.0*d2sVal*sqrtAOB)*(expVal/r) -(d2sVal - 2.0*dsVal*sqrtAOB)*((expVal/r)*sqrtAOB+ (expVal/(r*r))));

    derivativeList[1] = dRad*((x-xPos)/r);
	derivativeList[2] = dRad*((y-yPos)/r);
	derivativeList[3] = dRad*((z-zPos)/r);

    if(maxOrder == 1) return;


	double d2Rad     =  (-.79577471545947667884e-01*strength)*(expVal/(r*r*r))*
	                     (-2.0*dsVal*r*r*sqrtAOB*sqrtAOB*sqrtAOB + 5.0*sqrtAOB*sqrtAOB*r*r*d2sVal
                         -4.0*r*r*d3sVal*sqrtAOB -4.0*dsVal*r*sqrtAOB*sqrtAOB +6.0*r*d2sVal*sqrtAOB
                         +r*r*d4sVal-2.0*r*d3sVal -4.0*dsVal*sqrtAOB+2.0*d2sVal);
    // (4) = x^2
	// (5) = xy
	// (6) = xz
	// (7) = y^2
	// (8) = yz
	// (9) = z^2

    double r3 = r2*r;

	derivativeList[4] = (d2Rad/r2)*((x-xPos)*(x-xPos)) + (dRad/r3)*((y-yPos)*(y-yPos) + (z-zPos)*(z-zPos));
	derivativeList[5] = ((d2Rad/r2) - (dRad/r3))*((x-xPos)*(y-yPos)) ;
	derivativeList[6] = ((d2Rad/r2) - (dRad/r3))*((x-xPos)*(z-zPos));
	derivativeList[7] = (d2Rad/r2)*((y-yPos)*(y-yPos)) + (dRad/r3)*((x-xPos)*(x-xPos) + (z-zPos)*(z-zPos));
	derivativeList[8] = ((d2Rad/r2) - (dRad/r3))*((y-yPos)*(z-zPos));
	derivativeList[9] = (d2Rad/r2)*((z-zPos)*(z-zPos)) + (dRad/r3)*((x-xPos)*(x-xPos) + (y-yPos)*(y-yPos));

    return;
    }

	double evaluateRadialDerivative(double r) const
	{
	if(r < 0.1*radius) return 0.0;
	if(r >=    radius) return 0.0;

   	std::vector<double>      sDerivatives(3);
    sPrime.derivatives(r,sDerivatives,2);

    double dsVal     = sDerivatives[0];
    double d2sVal    = sDerivatives[1];
    double d3sVal    = sDerivatives[2];

    double expVal    = std::exp(-sqrtAOB*r);

    double dRadVal = (d3sVal - 2.0*d2sVal*sqrtAOB)*(expVal/r)
                     -(d2sVal - 2.0*dsVal*sqrtAOB)*((expVal/r)*sqrtAOB+ (expVal/(r*r)));
    dRadVal *= (-.79577471545947667884e-01*strength);

    return dRadVal;
	}


	double evaluateRadialSource(double r) const
	{
	if(r < 0.1*radius) return 0.0;
	if(r >=    radius) return 0.0;

	double dsVal       = sPrime(r);
	double sDoublePrimeVal = sPrime.derivative(r);

	return (-.79577471545947667884e-01*strength)*(std::exp(-sqrtAOB*r)/r)*(sDoublePrimeVal - 2.0*dsVal*sqrtAOB);
	}


	double evaluateSourceIntegral() const
	{
	double fourPi = 12.566370614359172954;
	double sum   = 0.0;
	long rPanels = 100;
	double r;
	double dr  = radius/rPanels;
	for(long i = 0; i < rPanels+1; i++)
	{
		r = i*dr;
		sum += evaluateRadialSource(r)*r*r*dr;
	}
	return sum*fourPi;
	}

    double evaluateSourceXr2Integral() const
	{
	double fourPioverThree = 4.1887902047863909846;
	double sum   = 0.0;
	long rPanels = 100;
	double r;
	double dr  = radius/rPanels;
	for(long i = 0; i < rPanels+1; i++)
	{
		r = i*dr;
		sum += evaluateRadialSource(r)*r*r*r*r*dr;
	}
	return sum*fourPioverThree;
	}

    //  Returns a std::function that is bound to the source evaluation operator of *this

#if __cplusplus > 199711L

	std::function<double(double,double,double)> getSourceEvaluationPtr() const
	{
	std::function<double(double,double,double)> F = [this](double x,double y,double z)
	{return this->evaluateSource(x,y,z);};
	return std::move(F);
	}

#endif

	 //  Returns a std::function that is bound to the potential evaluation operator of *this

#if __cplusplus > 199711L

	std::function<double(double,double,double)> getEvaluationPtr() const
	{
	std::function<double(double,double,double)> F = [this](double x,double y,double z)
	{return this->evaluatePotential(x,y,z);};
	return std::move(F);
	}

#endif

	double xPos;
	double yPos;
	double zPos;
	double screenCoeff;
	double laplaceCoeff;
	double radius;
	double strength;
	double sqrtAOB;

	SmoothPolyStep s;
	SmoothPolyMollifier1d sPrime;

	double differentiabilityOrder;
};


#undef DEFAULT_DIFFERENTIABILITY_
#endif /* _ScreenedSoln3d_ */
