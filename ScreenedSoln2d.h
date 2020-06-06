/*
 * ScreenedSoln2d.h
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

#ifndef SCREENED_SOLN_2D_
#define SCREENED_SOLN_2D_

#define DEFAULT_DIFFERENTIABILITY_ 6

#include "k0_c.h"
#include "k1_c.h"

//   For reference: the fundamental solution of
//
//   [laplaceCoeff*DELTA + screenCoeff]
//
//   is
//
//  -1/(2*pi*laplaceCoeff) * BesselK0(std::sqrt(|screenCoeff/laplaceCoeff|)*r)
//
//   screenCoeff*laplaceCoeff < 0
//
class ScreenedSoln2d
{
    public:

	ScreenedSoln2d() : s(),sPrime()
	{
    initialize();
	}

	ScreenedSoln2d(const ScreenedSoln2d& H)
	{
	initialize(H);
	}

	ScreenedSoln2d(double xPos,  double yPos, double radius,double strength,double laplaceCoeff, double screenCoeff)
	{
	initialize(xPos,yPos, radius,strength,laplaceCoeff, screenCoeff);
	}

	void initialize(const ScreenedSoln2d& H)
	{
	xPos                   = H.xPos;
	yPos                   = H.yPos;
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
	radius           = 0.0;
	strength         = 0.0;
	screenCoeff      = 0.0;
	laplaceCoeff     = 0.0;
	sqrtAOB          = 0.0;
	differentiabilityOrder = DEFAULT_DIFFERENTIABILITY_;
	}


	//  Step function used
	//                                  |   |    ----
	//                           |
	//                     |
	//                |
	//           |
	// --------
	// 0   .1*radius                              radius
	// |======|======================================|
	//
	void initialize(double xPos,  double yPos,double radius, double strength,double laplaceCoeff, double screenCoeff)
	{

    //
	//  A fatal error if the signs of screenCoeff and laplaceCoeff are the same.
	//
	if(laplaceCoeff*screenCoeff > 0)
	{
    printf("XXXX Error :  ScreenedSoln2d  XXXX \n");
    printf("Coefficients of the ScreenedPoisson operator are of the same sign. \n\n");
    printf("XXXX Execution Halted XXXX\n");
    exit(0);
    }

    this->xPos                   = xPos;
	this->yPos                   = yPos;
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
	this->laplaceCoeff   = laplaceCoeff;
	this->screenCoeff    = screenCoeff;
	this->sqrtAOB        = std::sqrt(std::abs(screenCoeff/laplaceCoeff));
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

    void derivatives(double x, double y, std::vector<double>& derivativeList, int maxOrder) const
    {
    double r2 = (x-xPos)*(x-xPos) + (y-yPos)*(y-yPos);
    double r  = std::sqrt(std::abs(r2));

    switch (maxOrder)
   	{
   	case 0   :  derivativeList.resize(1); break;
   	case 1   :  derivativeList.resize(3); break;
   	case 2   :  derivativeList.resize(6); break;
   	default  :  derivativeList.resize(6); break;
   	}

   	for(int i= 0; i < (int)derivativeList.size(); i++) {derivativeList[i] = 0.0;}

    if(r < 0.1*radius) {return;}


    double besselK0val;
	double besselK1val;
    double sVal;

	besselK0val       = besselK0.besk0(sqrtAOB*r);

	sVal              = s(r);
	derivativeList[0] = -(0.15915494309189533577*strength/laplaceCoeff)*sVal*besselK0val;

	if(maxOrder == 0) return;

    besselK1val     = besselK1.besk1(sqrtAOB*r);

	std::vector<double>      sDerivatives(2);
    sPrime.derivatives(r,sDerivatives,1);

    double dsVal     = sDerivatives[0];
    double d2sVal    = sDerivatives[1];

	double dRad   =  -(0.15915494309189533577*(strength/laplaceCoeff))*(dsVal*besselK0val-sVal*besselK1val*sqrtAOB);


	derivativeList[1] = dRad*((x-xPos)/r);
	derivativeList[2] = dRad*((y-yPos)/r);

	if(maxOrder == 1) return;


    double d2Rad =
    -(0.15915494309189533577*(strength/laplaceCoeff))*
    (sVal*besselK0val*r*sqrtAOB*sqrtAOB - 2.0*dsVal*besselK1val*sqrtAOB*r + d2sVal*besselK0val*r + sVal*besselK1val*sqrtAOB)/ r;

    double r3 = r2*r;

	derivativeList[3] = (d2Rad/r2)*((x-xPos)*(x-xPos)) + (dRad/r3)*((y-yPos)*(y-yPos));
	derivativeList[4] = ((d2Rad/r2) - (dRad/r3))*((x-xPos)*(y-yPos)) ;
	derivativeList[5] = (d2Rad/r2)*((y-yPos)*(y-yPos)) + (dRad/r3)*((x-xPos)*(x-xPos));

    return;
    }


	double evaluatePotential(double x, double y) const
	{

	//double OneOver2pi = 0.15915494309189533577;

	double r = std::sqrt(std::abs((x-xPos)*(x-xPos) + (y-yPos)*(y-yPos)));
	if(r < 0.1*radius) return 0.0;
	return -(0.15915494309189533577*strength/laplaceCoeff)*s(r)*besselK0.besk0(sqrtAOB*r);
	}

   	 //  Returns a std::function that is bound to the potential evaluation operator of *this

#if __cplusplus > 199711L

	std::function<double(double,double)> getEvaluationPtr() const
	{
	std::function<double(double,double)> F = [this](double x,double y) {return this->evaluatePotential(x,y);};
	return std::move(F);
	}

#endif


    void evaluatePotentialDerivatives(double x, double y, double& dPx, double& dPy) const
	{
	int maxOrder = 1;
    std::vector<double>    derivativeList(3);
	derivatives(x, y, derivativeList, maxOrder);
	dPx = derivativeList[1];
	dPy = derivativeList[2];
	}


    double evaluateSource(double x, double y) const
	{
    std::vector<double>          derivativeList(1);
	sourceDerivatives(x, y, derivativeList, 0);
	return derivativeList[0];
	}

    //  Returns a std::function that is bound to the source evaluation operator of *this

#if __cplusplus > 199711L

	std::function<double(double,double)> getSourceEvaluationPtr() const
	{
	std::function<double(double,double)> F = [this](double x,double y) {return this->evaluateSource(x,y);};
	return std::move(F);
	}

#endif



	void evaluateSourceDerivatives(double x, double y, double& dSx, double& dSy) const
	{
	std::vector<double> derivativeList(3);
	sourceDerivatives(x, y, derivativeList, 1);
	dSx = derivativeList[1];
	dSy = derivativeList[2];
	}

    void sourceDerivatives(double x, double y, std::vector<double>& derivativeList, int maxOrder) const
    {
    double r2 = (x-xPos)*(x-xPos) + (y-yPos)*(y-yPos);
    double r  = std::sqrt(std::abs(r2));

    switch (maxOrder)
   	{
   	case 0   :  derivativeList.resize(1); break;
   	case 1   :  derivativeList.resize(3); break;
   	case 2   :  derivativeList.resize(6); break;
   	default  :  derivativeList.resize(6); break;
   	}

   	for(int i= 0; i < (int)derivativeList.size(); i++) {derivativeList[i] = 0.0;}

    if((r < 0.1*radius)||(r >= radius)) {return;}

   	std::vector<double>      sDerivatives(4);
    sPrime.derivatives(r,sDerivatives,3);

    double dsVal     = sDerivatives[0];
    double d2sVal    = sDerivatives[1];
    double d3sVal    = sDerivatives[2];
    double d4sVal    = sDerivatives[3];

	double besselK0val  = besselK0.besk0(sqrtAOB*r);
	double besselK1val  = besselK1.besk1(sqrtAOB*r);

    derivativeList[0] = -0.15915494309189533577*(strength)*(besselK0val*(dsVal/r + d2sVal) - 2.0*sqrtAOB*dsVal*besselK1val);
    if(maxOrder == 0) return;


    double dRad = (-0.15915494309189533577*(strength))*
                   (-besselK1val*sqrtAOB*(dsVal/r + d2sVal)
                   + besselK0val*(d2sVal/r - dsVal/(r2) + d3sVal)
                   - 2.0*sqrtAOB*besselK1val*d2sVal
                   + 2.0*sqrtAOB*dsVal*(besselK0val*sqrtAOB + besselK1val/r));


    derivativeList[1] = dRad*((x-xPos)/r);
	derivativeList[2] = dRad*((y-yPos)/r);

    if(maxOrder == 1) return;

    double r3 = r2*r;

    double d2Rad =(-0.15915494309189533577*(strength))*
    (-2.0*dsVal*besselK1val*sqrtAOB*sqrtAOB*sqrtAOB*r3
    + 5.0*d2sVal*besselK0val*r3*sqrtAOB*sqrtAOB
    - 4.0*sqrtAOB*besselK1val*d3sVal*r3
    - dsVal*besselK0val*r2*sqrtAOB*sqrtAOB
    + 3.0*sqrtAOB*besselK1val*d2sVal*r2
    + besselK0val*d4sVal*r3
    - besselK1val*dsVal*r*sqrtAOB
    + besselK0val*d3sVal*r2
    - 2.0*d2sVal*besselK0val*r +2.0*dsVal*besselK0val)/(r3);


	derivativeList[3] = (d2Rad/r2)*((x-xPos)*(x-xPos)) + (dRad/r3)*((y-yPos)*(y-yPos));
	derivativeList[4] = ((d2Rad/r2) - (dRad/r3))*((x-xPos)*(y-yPos)) ;
	derivativeList[5] = (d2Rad/r2)*((y-yPos)*(y-yPos)) + (dRad/r3)*((x-xPos)*(x-xPos));
    return;
    }


	double evaluateRadialSource(double r) const
	{
	if(r < 0.1*radius) return 0.0;
	if(r >=    radius) return 0.0;

	double besselK0val     = besselK0.besk0(sqrtAOB*r);
	double besselK1val     = besselK1.besk1(sqrtAOB*r);

	std::vector<double>      sDerivatives(2);
    sPrime.derivatives(r,sDerivatives,1);

    double dsVal       = sDerivatives[0];
    double d2sVal      = sDerivatives[1];
	return -0.15915494309189533577*(strength)*(besselK0val*(dsVal/r + d2sVal) - 2.0*sqrtAOB*dsVal*besselK1val);
	}


    double evaluateRadialSourceDerivative(double r) const
	{
	if(r < 0.1*radius) return 0.0;
	if(r >     radius) return 0.0;

	double besselK0val     = besselK0.besk0(sqrtAOB*r);
	double besselK1val     = besselK1.besk1(sqrtAOB*r);

   	std::vector<double>      sDerivatives(3);
    sPrime.derivatives(r,sDerivatives,2);

    double dsVal     = sDerivatives[0];
    double d2sVal    = sDerivatives[1];
    double d3sVal    = sDerivatives[2];

    // derivative of (besselK0val*(dsVal/r + d2sVal) - 2.0*sqrtAOB*dsVal*besselK1val)

    double dVal = -besselK1val*sqrtAOB*(dsVal/r + d2sVal)
                  + besselK0val*(d2sVal/r - dsVal/(r*r) + d3sVal)
                  - 2.0*sqrtAOB*besselK1val*d2sVal
                  + 2.0*sqrtAOB*dsVal*(besselK0val*sqrtAOB + besselK1val/r);

	return  -0.15915494309189533577*(strength)*dVal;
	}

	double evaluateSourceIntegral()
	{
	double twoPi = 6.2831853071795864769;
	double sum   = 0.0;
	long rPanels = 200;
	double r;
	double dr  = radius/rPanels;
	for(long i = 0; i < rPanels+1; i++)
	{
		r = i*dr;
		sum += evaluateRadialSource(r)*r*dr;
	}
	return sum*twoPi;
	}

    double evaluateSourceXr2Integral()
	{
	double pi = 3.1415926535897932385;
	double sum   = 0.0;
	long rPanels = 200;
	double r;
	double dr  = radius/rPanels;
	for(long i = 0; i < rPanels+1; i++)
	{
		r = i*dr;
		sum += evaluateRadialSource(r)*r*r*r*dr;
	}
	return sum*pi;
	}



	double xPos;
	double yPos;
	double screenCoeff;
	double laplaceCoeff;
	double radius;
	double strength;
	double sqrtAOB;

	SmoothPolyStep s;
	SmoothPolyMollifier1d sPrime;

	double differentiabilityOrder;

	BesselK0 besselK0;
	BesselK1 besselK1;
};


#undef DEFAULT_DIFFERENTIABILITY_
#endif /* _ScreenedSoln2d_ */



