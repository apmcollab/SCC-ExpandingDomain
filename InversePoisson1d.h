
//###########################################################
//                InversePoisson1d
//###########################################################
//
// This class uses discrete sin transforms and Green's function
// corrections to evaluate the inverse of the constant coefficient
// Poisson and screened Poisson operator in 1 dimension given by
//
// [ laplaceCoeff (d^2/dx^2) + screenCoeff ]  u
//
// The product of laplaceCoeff and screenCoeff must be less than or equal to 0.
//
// The unit test code is Poisson1dDev.cpp
//
// Chris Anderson, Dec. 18, 2013
// (C) UCLA
//
// Note: When using this routine in separate threads the initialization
// step must be done in single-threaded mode so that the internal
// fftw transform plans get created in single-threaded mode. If this
// isn't done properly, then one will typically get and error message
// indicating "double free or memory corruption".
//
//
//
/*
#############################################################################
#
# Copyright 2013-2015 Chris Anderson
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

#include "DoubleVectorNd/SCC_DoubleVector1d.h"
#include "FFTW3_InterfaceNd/SCC_fftw3_sin1d.h"
#include "FFTW3_InterfaceNd/SCC_FFT_Nvalues.h"

#include <cmath>
#include <cstdio>
#include <iostream>
#include <cstdlib>


#ifndef INVERSE_POISSON_1D_
#define INVERSE_POISSON_1D_

class InversePoisson1d
{
    public:

	InversePoisson1d()
	{
	initialize();
	}

    InversePoisson1d(double laplaceCoeff, long xPanel, double xMin, double xMax)
	{
	initialize(laplaceCoeff,0.0,xPanel, xMin, xMax);
	}


	InversePoisson1d(double laplaceCoeff, double screenCoeff, long xPanel, double xMin, double xMax)
	{
	initialize(laplaceCoeff, screenCoeff,xPanel, xMin, xMax);
	}

    InversePoisson1d (const InversePoisson1d& H)
    {
    initialize(H);
    }

	void initialize()
	{

    this->nx    = 0;
	this->xMin  = 0.0;
	this->xMax  = 0.0;
    this->laplaceCoeff  = 0.0;
	this->screenCoeff   = 1.0;

    this->extNx      = 0;
	this->extXoffset = 0;
    this->xMinExt    = 0.0;
    this->xMaxExt    = 0.0;

	SFT.initialize();
	Vtmp.initialize();
	VsinTemp.initialize();
	}

    void initialize(double laplaceCoeff,  long xPanel, double xMin, double xMax)
	{
	initialize(laplaceCoeff,0.0,xPanel, xMin, xMax);
	}

	void initialize(double laplaceCoeff, double screenCoeff, long xPanel, double xMin, double xMax)
	{
	//
	//  A fatal error if the signs of screenCoeff and laplaceCoeff are the same.
	//
	if(laplaceCoeff*screenCoeff > 0)
	{
    printf("XXXX Error :  InversePoisson1d  XXXX \n");
    printf("Coefficients of the Helmholtz operator are of the same sign. \n\n");
    printf("XXXX Execution Halted XXXX\n");
    exit(0);
    }

    this->nx    = xPanel;
    this->xMin  = xMin;
    this->xMax  = xMax;
	this->laplaceCoeff  = laplaceCoeff;
	this->screenCoeff   = screenCoeff;

	//  Determine extended domain and panel counts

    double hx = (xMax-xMin)/(double)(nx);

    // Resize domain if needed to obtain a panel count that leads to efficient FFT's

    SCC::FFT_Nvalues fftNvalues;

    extNx      = fftNvalues.getFFT_N(nx);
    if(extNx > nx)
    {
    extXoffset = (extNx-nx)/2;
    xMinExt    = xMin - extXoffset*hx;
    xMaxExt    = xMinExt + extNx*hx;
    }
    else
    {
    extXoffset = 0;
    xMinExt    = xMin;
    xMaxExt    = xMax;
    }

	//  Initialize the discrete Fourier transform class.
    /*
	printf(" nx    :  %ld  \n",nx);
    printf(" extNx :  %ld  \n",extNx);
    printf("[ Xmin, Xmax ]       = [ %10.5f , %10.5f ] \n",xMin,xMax);
    printf("[ extXmin, extXmax ] = [ %10.5f , %10.5f ] \n",xMinExt,xMaxExt);
    */

    SFT.initialize(extNx);
    Vtmp.initialize(extNx,xMinExt,xMaxExt);
    VsinTemp.initialize(extNx-1);
	}

	void initialize(const InversePoisson1d& H)
	{
	if(this->nx == 0) initialize();
	this->nx    = H.nx;
	this->xMin  = H.xMin;
	this->xMax  = H.xMax;
	this->laplaceCoeff  = H.laplaceCoeff;
	this->screenCoeff = H.screenCoeff;

	this->extNx      = H.extNx;
	this->extXoffset = H.extXoffset;
    this->xMinExt    = H.xMinExt;
    this->xMaxExt    = H.xMaxExt;

	SFT.initialize(extNx);
	Vtmp.initialize(extNx,xMinExt,xMaxExt);
	VsinTemp.initialize(extNx-1);
	}

	void setCoefficients(double laplaceCoeff, double screenCoeff)
	{
	this->laplaceCoeff  = laplaceCoeff;
	this->screenCoeff = screenCoeff;
	}

//
// This routine evaluates the infinite domain solution of the constant coefficient Helmholtz equation
//
// The evaluation domain is that specified by the grid structure
// information in the input argument V.
//
// Input  : GridFunction1d V  -- Specifies the right hand side. The endpoint values are ignored.
//
// Output : GridFunction1d V   -- The solution of the iHelmholtz operator with "infinite" boundary conditions
//
	void applyInverseOp(SCC::DoubleVector1d& V)
	{
	double dV_A;
	double dV_B;

	// Create solution with homogeneous boundary data and capture the
	// the derivatives of the solution at the end points.

	homogeneousBCHelmOpInv1d(V, dV_A, dV_B);

	// Add in correction to cancel discontinuities at endpoints

	addInfInfHelmOpInvCorrection1d(dV_A, dV_B);

	// Insert solution values created in Vtmp

    for(long i = 0;  i <= nx; i++)  {V(i) = Vtmp(i + extXoffset);}
	}

//
// This routine evaluates the infinite domain solution of the constant coefficient Helmholtz equation
//
// The evaluation domain is that specified by the grid structure
// information in the input argument V.
//
// Input  :  GridFunction1d V  -- Specifies the right hand side. The endpoint values are ignored.
//           sigma_A, sigma_B  -- Specifies the strength of a specified flux jump in the solution
//                                at the endpoints.
//
//   laplaceCoeff*[dV/dx]_xMin = sigma_A
//   laplaceCoeff*[dV/dx]_xMax = sigma_B
//
/// where [dV/dx]_xStar = the jump in dV/dx at x=xStar.
//
//
//
// Output : GridFunction1d V   -- The solution of the Helmholtz operator with "infinite" boundary conditions
//                                and the required jump in the derivatives at x = xMin and x = xMax
//
//
	void applyInverseOp(SCC::DoubleVector1d& V,double sigma_A, double sigma_B)
	{
	double dV_A;
	double dV_B;

	// Create solution with homogeneous boundary data and capture the
	// the derivatives of the solution at the end points.

	homogeneousBCHelmOpInv1d(V, dV_A, dV_B);

	// Add in correction to cancel discontinuities at endpoints and
	// add in a component with a specified jump.

	addInfInfHelmOpInvCorrection1d(dV_A, dV_B);

	// Insert solution values

    for(long i = 0;  i <= nx; i++)  {V(i) = Vtmp(i + extXoffset);}

    //
    // Add in boundary source contribution
    //
    addBoundarySourceSolution(sigma_A, sigma_B,V);
	}
//
// This routine inverts the Helmholtz operator using discrete sin transforms with homogeneous boundary conditions.
//
// Input  :  SCC::DoubleVector1d  V  -- Specifies the right hand side. The endpoint values are ignored.
//
// Output : SCC::DoubleVector1d   V   -- The solution of the Helmholtz operator with homogeneous Diriclet values.
//          dV_A, dV_B          -- The one sided derivatives of the solution at the endpoints using
//                                 interior values of V.
//
// All grid and domain size information is extracted from the input function V
//
//
	void homogeneousBCHelmOpInv1d(SCC::DoubleVector1d& V,  double& dV_A, double& dV_B)
	{
//
//  Number of panels  = nx
//
    if(V.getIndex1Size()-1 != nx)
	{
    printf("XXXX Error :  InversePoisson1d  XXXX \n");
    printf("Input GridFunction1d right hand side panel \n");
    printf("count does not match solver panel count.\n\n");
    printf("Solver panel count : %ld  \n",nx);
    printf("RHS    panel count : %ld  \n\n",V.getIndex1Size()-1);
    printf("XXXX Execution Halted XXXX\n");
    exit(0);
    }

    double extSizeX = (xMaxExt-xMinExt);
    double pi = .3141592653589793e+01;

    for(long i =    0;  i < extXoffset; i++)     {Vtmp(i) = 0.0;}
    for(long i =    0;  i <= nx; i++)            {Vtmp(i + extXoffset) = V(i);}
    for(long i = nx+1;  i <= extNx; i++)         {Vtmp(i) = 0.0;}

//  Apply forward transform

	SFT.fftw1d_sin_forward(Vtmp,VsinTemp);

//  Multiply by inverse Fourier factors
//  associated with the second difference operator.

    long k1;
    double opTransformFactor;

	for(k1 = 1; k1 <=extNx-1; k1++)
	{
	  opTransformFactor = -laplaceCoeff*(((pi*k1)*(pi*k1))/(extSizeX*extSizeX)) + screenCoeff;
	  VsinTemp(k1-1)   *= 1.0/opTransformFactor;
	}

    // Before transforming back, evaluate the derivatives at the left and
	// right endpoints by summing the sin series for the derivative.
	//
	// a = kth coefficient => a*std::sqrt(2/L)*sin(k*pi*x_i/L) => derivative = a*std::sqrt(2/L)*cos(k*pi*x_i/L)*(k*pi/L);

    dV_A = 0.0;
	dV_B = 0.0;

	double cosSign = -1.0;

	for(long k1 = 1; k1 <= extNx-1; k1++)
	{
	dV_A    += VsinTemp(k1-1)*((double)k1*pi/extSizeX);
	dV_B    += VsinTemp(k1-1)*((double)k1*pi/extSizeX)*cosSign;
	cosSign *= -1.0;
	}

    dV_A *= std::sqrt(2.0/extSizeX);
    dV_B *= std::sqrt(2.0/extSizeX);

//  Transform back.

	SFT.fftw1d_sin_inverse(VsinTemp,Vtmp);

    /*
    // The following code used an earlier version of fftw_sin1d that
    // used input and output data vectors of size xPanels + 1
    // 11/12/2015
    //
    // Copy data to larger array

    for(long i = 0;  i < extXoffset; i++)   {Vtmp(i) = 0.0;}
    for(long i = 0;  i <= nx; i++)          {Vtmp(i + extXoffset) = V(i);}
    for(long i = nx+1; i <= extNx; i++)     {Vtmp(i) = 0.0;}
//
//  Apply forward transform
//
	SFT.fftw1d_sin_forward(Vtmp);

//
//  Multiply the interior values of V by inverse Fourier factors
//  associated with the second difference operator.
//
    long k1;
    double opTransformFactor;

	for(k1 = 1; k1 <=extNx-1; k1++)
	{
	  opTransformFactor = -laplaceCoeff*(((pi*k1)*(pi*k1))/(extSizeX*extSizeX)) + screenCoeff;
	  Vtmp(k1)            *= 1.0/opTransformFactor;
	}
	// Before transforming back, evaluate the derivatives at the left and
	// right endpoints by summing the sin series for the derivative.
	//
	// a = kth coefficient => a*std::sqrt(2)*sin(k*pi*x_i/L) => derivative a*std::sqrt(2)*cos(k*pi*x_i/L)*(k*pi/L);

    dV_A = 0.0;
	dV_B = 0.0;

	double cosSign = -1.0;

	for(long k1 = 1; k1 <= extNx-1; k1++)
	{
	dV_A    += Vtmp(k1)*((double)k1*pi/extSizeX);
	dV_B    += Vtmp(k1)*((double)k1*pi/extSizeX)*cosSign;
	cosSign *= -1.0;
	}

	dV_A *= std::sqrt(2.0);
	dV_B *= std::sqrt(2.0);
//
//  Transform back. Since the boundary values are invariant under the
//  transformation fftw1d_sin_forward implemented by the interface class,
//  there is no need to std::explicitly work with the boundary values.
//
	SFT.fftw1d_sin_inverse(Vtmp);

//  Set end conditions == 0

	Vtmp(0)  = 0.0;
	Vtmp(extNx) = 0.0;
*/
}

// This routine adds the correction to vTmp a solution of the Laplace equation associated with
// inf-inf boundary conditions.

   void addInfInfHelmOpInvCorrection1d(double dV_A, double dV_B)
   {
    double hx = (xMaxExt-xMinExt)/(double)(extNx);
    double str;
    double correction;
    double laplaceCoeffAlphaFactor;
    double correctionFactor;
    double x;

    if(std::abs(laplaceCoeff)  < 1.0e-99) return; // No correction if laplaceCoeff is zero


	if(std::abs(screenCoeff) < 1.0e-12)  // Add str*( 1/(2*laplaceCoeff) |x-xStar| ) for xStar = xMin and xStar = xMax
	                            // to cancel derivative jumps at the endpoints
	                            // Jump = str*(1/laplaceCoeff).
	{
	for(long i = 0; i <= extNx; i++)
	{
	x = i*hx;
	Vtmp(i)   += -dV_A*(x/2.0);  // laplaceCoeff's cancel
	x = (extNx-i)*hx;
	Vtmp(i)   += +dV_B*(x/2.0);  // laplaceCoeff's cancel
	}

	}
	else  // Add str*std::exp(- std::sqrt(|screenCoeff/laplaceCoeff|) |x-xStar| ) for xStar = xMin and xStar = xMax
	      // to cancel derivative jumps at the endpoints
	      // Jump  = -2*str*std::sqrt(std::abs(screenCoeff/laplaceCoeff))
	{
	laplaceCoeffAlphaFactor  = std::sqrt(std::abs(screenCoeff/laplaceCoeff));
	str              = (-dV_A)/(-2.0*laplaceCoeffAlphaFactor);
	correction       = str;
	Vtmp(0)         += correction;
	correctionFactor = std::exp(-laplaceCoeffAlphaFactor*hx);
	for(long i = 1; i <= extNx; i++)
	{
	correction *= correctionFactor;
    Vtmp(i)     += correction;
	}

	str              = (dV_B)/(-2.0*laplaceCoeffAlphaFactor);
	correction       = str;
	Vtmp(extNx)     += correction;
	correctionFactor = std::exp(-laplaceCoeffAlphaFactor*hx);
	for(long i = extNx-1; i >= 0; i--)
	{
	correction      *= correctionFactor;
    Vtmp(i)         += correction;
	}

	}
	}
//
//  This routine adds to the input function values a solution with jumps in the derivatives
//  at the endpoints of the originally specified interval (not the internal std::expanded interval)
//
//
//   laplaceCoeff*[du/dx]_xMin = sigma_A
//   laplaceCoeff*[du/dx]_xMax = sigma_B
//
/// where [du/dx]_xStar = the jump in du/dx at x=xStar.
//
//
	void addBoundarySourceSolution(double sigma_A, double sigma_B,SCC::DoubleVector1d& V)
    {
    double hx = (xMax-xMin)/(double)(nx);
    double str;
    double correction;
    double laplaceCoeffAlphaFactor;
    double correctionFactor;
    double x;

    if(std::abs(laplaceCoeff)  < 1.0e-99) return; // No correction if laplaceCoeff is zero


	if(std::abs(screenCoeff) < 1.0e-12)  // Add str*( 1/(2*laplaceCoeff) |x-xStar| ) for xStar = xMin and xStar = xMax
	                            // to cancel derivative jumps at the endpoints
	                            // Jump = str*(1/laplaceCoeff).
	{
		for(long i = 0; i <= nx; i++)
		{
		x = i*hx;
		V(i)   += (sigma_A/laplaceCoeff)*(x/2.0);
		x = (nx-i)*hx;
		V(i)   += (sigma_B/laplaceCoeff)*(x/2.0);
		}

	}
	else  // Add str*std::exp(- std::sqrt(|screenCoeff/laplaceCoeff|) |x-xStar| ) for xStar = xMin and xStar = xMax
	      // to cancel derivative jumps at the endpoints
	      // Jump  = -2*str*std::sqrt(std::abs(screenCoeff/laplaceCoeff))
	{
	laplaceCoeffAlphaFactor  = std::sqrt(std::abs(screenCoeff/laplaceCoeff));
	str  = (sigma_A/laplaceCoeff)/(-2.0*laplaceCoeffAlphaFactor);

	correction       = str;
	V(0)         += correction;
	correctionFactor = std::exp(-laplaceCoeffAlphaFactor*hx);
	for(long i = 1; i <= nx; i++)
	{
	correction *= correctionFactor;
    V(i)     += correction;
	}

	str  = (sigma_B/laplaceCoeff)/(-2.0*laplaceCoeffAlphaFactor);
	correction       = str;
	V(nx)     += correction;
	correctionFactor = std::exp(-laplaceCoeffAlphaFactor*hx);
	for(long i = nx-1; i >= 0; i--)
	{
	correction   *= correctionFactor;
    V(i)         += correction;
	}

	}
	}

	double     screenCoeff;
	double    laplaceCoeff;
	long        nx;
	double    xMin;
	double    xMax;

	long        extNx;
	double    xMinExt;
	double    xMaxExt;
	long    extXoffset;

	SCC::fftw3_sin1d         SFT;
	SCC::GridFunction1d     Vtmp;
	SCC::DoubleVector1d VsinTemp;
};

#endif

