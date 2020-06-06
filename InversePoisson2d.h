
//###########################################################
//                InversePoisson2d
//###########################################################
//
// This class uses the Fourier X Z procedure to compute
// the evaluate the infinite domain solution to
//
// [ laplaceCoeff (d^2/dx^2) + laplaceCoeff (d^2/dy^2)  +   screenCoeff ]  u
//
// in the region [xMin,xMax] x [yMin,yMax]
//
// The product of laplaceCoeff and screenCoeff must be less than or equal to 0.
//
// The unit test code is Poisson2dDev.cpp
//
// Chris Anderson, July 12, 2015
// (C) UCLA
//
// Note: When using this routine in separate threads the initialization
// step must be done in single-threaded mode so that the internal
// fftw transform plans get created in single-threaded mode. If this
// isn't done properly, then one will typically get and error message
// indicating "double free or memory corruption".
//
//
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
/*
 ToDo: Determine appropriate screenCoeff cut-off for Helmholtz solution
*/
#include "DoubleVectorNd/SCC_DoubleVector1d.h"
#include "DoubleVectorNd/SCC_DoubleVector2d.h"

#include "GridFunctionNd/SCC_GridFunction2d.h"

#include "FFTW3_InterfaceNd/SCC_fftw3_1d.h"
#include "FFTW3_InterfaceNd/SCC_fftw3_sin1d.h"
#include "FFTW3_InterfaceNd/SCC_FFT_Nvalues.h"

#include "ScreenedNpole2d.h"
#include "PoissonNpole2d.h"
#include "InversePoisson1d.h"

#include <cmath>
#include <cstdio>
#include <iostream>
#include <cstdlib>



#ifndef INVERSE_POISSON_2D_
#define INVERSE_POISSON_2D_

#define DEFAULT_EXTENSION_FACTOR_     2.0
#define DEFAULT_MAX_NPOLE_ORDER_      2
#define DEFAULT_DIFFERENTIABILITY_    8
#define DEFAULT_SCREEN_BOUND_         10.0


class InversePoisson2d
{
    public:

	InversePoisson2d()
	{
	initialize();
	}

	InversePoisson2d(double laplaceCoeff, double screenCoeff, long xPanel, double xMin, double xMax,
	long yPanel, double yMin, double yMax, double extensionFactor =-1.0)
	{
    this->maxNpoleOrder    = DEFAULT_MAX_NPOLE_ORDER_;
	this->nPoleDiffOrder   = DEFAULT_DIFFERENTIABILITY_;

	initialize(laplaceCoeff, screenCoeff,xPanel, xMin, xMax,yPanel,yMin,yMax,extensionFactor);
	}

	InversePoisson2d(double laplaceCoeff, long xPanel, double xMin, double xMax,
	long yPanel, double yMin, double yMax, double extensionFactor =-1.0)
	{
    this->maxNpoleOrder    = DEFAULT_MAX_NPOLE_ORDER_;
	this->nPoleDiffOrder   = DEFAULT_DIFFERENTIABILITY_;

	initialize(laplaceCoeff, 0.0,xPanel, xMin, xMax,yPanel,yMin,yMax,extensionFactor);
	}

    InversePoisson2d (const InversePoisson2d& H)
    {
    initialize(H);
    }

    void setMaxNpoleOrder(int maxNpoleOrder)
    {
    this->maxNpoleOrder = maxNpoleOrder;

    	if(std::abs(screenCoeff) < 1.0e-10)
		{
			nPole.initialize(xCent,yCent,rBar,maxNpoleOrder,laplaceCoeff);
		}
		else if(std::abs(screenCoeff) < screenCoeffBound)
		{
		    SnPole.initialize(xCent,yCent,rBar,maxNpoleOrder,laplaceCoeff,screenCoeff);
		}
    }

    void setNpoleDiffOrder(long nPoleDiffOrder)
    {this->nPoleDiffOrder  = nPoleDiffOrder;}

    void setScreenCoeffBound(double screenCoeffBound)
    {
    this->screenCoeffBound = screenCoeffBound;
    }

	void initialize()
	{
    this->nx    = 0;   this->ny   = 0;
    this->xMin  = 0.0; this->yMin = 0.0;
    this->xMax  = 0.0; this->yMax = 0.0;

    this->laplaceCoeff  = 1.0;
	this->screenCoeff   = 0.0;

	this->screenCoeffBound = DEFAULT_SCREEN_BOUND_;
	this->extensionFactor  = DEFAULT_EXTENSION_FACTOR_;

	this->xCent = 0.0; this-> yCent = 0.0; this->rBar = 0.0;

	this->maxNpoleOrder  = DEFAULT_MAX_NPOLE_ORDER_;
	this->nPoleDiffOrder = DEFAULT_DIFFERENTIABILITY_;
	DFT.initialize();
	}

	void initialize(double laplaceCoeff, long xPanel, double xMin, double xMax,
	long yPanel, double yMin, double yMax, double extFactor = -1.0)
	{
	initialize(laplaceCoeff, 0.0,xPanel, xMin, xMax,yPanel,yMin,yMax,extFactor);
	}

	void initialize(double laplaceCoeff, double screenCoeff, long xPanel, double xMin, double xMax,
	long yPanel, double yMin, double yMax, double extFactor = -1.0)
	{
	//
	//  A fatal error if the signs of screenCoeff and laplaceCoeff are the same.
	//
	if(std::abs(screenCoeff) > 1.0e-10)
	{
	if(laplaceCoeff*screenCoeff > 0)
	{
    printf("XXXX Error :  InversePoisson2d  XXXX \n");
    printf("Coefficients of the Helmholtz operator are of the same sign. \n\n");
    printf("XXXX Execution Halted XXXX\n");
    exit(0);
    }}

    this->nx    = xPanel;   this->ny   = yPanel;
    this->xMin  =  xMin;    this->yMin = yMin;
    this->xMax  =  xMax;    this->yMax = yMax;

	this->laplaceCoeff     =  laplaceCoeff;
	this->screenCoeff      =  screenCoeff;
	this->screenCoeffBound = DEFAULT_SCREEN_BOUND_;


	if(extFactor  < 0)  {extensionFactor = DEFAULT_EXTENSION_FACTOR_;}
	else                {extensionFactor = extFactor;}


	//  Determine extended domain and panel counts

    double hx = (xMax-xMin)/(double)(nx);
    double hy = (yMax-yMin)/(double)(ny);

    extSizeX = (xMax-xMin)*extensionFactor;
    extNx    = (long)(extSizeX/hx);

    // Add a fix if one is using extensionFactor 1 so that the transform is done over a marginally larger domain

    if(extNx == nx) extNx += 2;

    // Resize domain if needed to obtain a panel count that leads to efficient
    // FFT's

    SCC::FFT_Nvalues fftNvalues;
    extNx      = fftNvalues.getFFT_N(extNx);
    extXoffset = (extNx-nx)/2;


    /*
    printf("[ nx, ny ] : [ %ld , %ld ] \n",nx,ny);
    printf("[ Xmin, Xmax ] = [ %10.5f , %10.5f ] \n",xMin,xMax);
    printf("[ Ymin, Ymax ] = [ %10.5f , %10.5f ] \n",yMin,yMax);
    printf("[ extXmin, extXmax ] = [ %10.5f , %10.5f ] \n",xMinExt,xMaxExt);
    */

	//  Initialize Fourier transform data


	DFT.initialize(extNx);
    inReal1Dx.initialize(extNx);
    inImag1Dx.initialize(extNx);

	outReal1Dx.initialize(extNx);
    outImag1Dx.initialize(extNx);

    realData1Dx1D.initialize((long)(extNx/2) + 1 ,ny+1);
    imagData1Dx1D.initialize((long)(extNx/2) + 1 ,ny+1);


    // Initialize N-pole data

    nPoleSource.initialize(nx,xMin,xMax,ny,yMin,yMax);
    nPolePotential.initialize(nx,xMin,xMax,ny,yMin,yMax);

    //
    // Determine the coordinates of the grid point
    // nearest to the center of the domain at which
    // to center the nPole distribution.
    //
    //
    xCent      = xMin + (xMax-xMin)/2.0;
    yCent      = yMin + (yMax-yMin)/2.0;

    long   xCentIndex = (long)(((xCent-xMin) + hx/2.0)/hx);
    long   yCentIndex = (long)(((yCent-yMin) + hy/2.0)/hy);

    xCent = xMin + xCentIndex*hx;
    yCent = yMin + yCentIndex*hy;

    //
    // Determine the maximal radius for the central monopole
    // This is chosen to be the radius from the (xCent,yCent)
    // so that the outer boundary is at least two mesh panels
    // away from any boundary
    //
    // Use "set view equal xy" to view distribution in gnuplot if
    // the dimensions are unequal


    rBar = std::abs(xMax-xCent) - 2.0*hx;
    rBar  = (rBar > (std::abs(xMax-xCent) - 2.0*hx)) ? (std::abs(xMax-xCent) - 2.0*hx) : rBar;
    rBar  = (rBar > (std::abs(yMin-yCent) - 2.0*hy)) ? (std::abs(yMin-yCent) - 2.0*hy) : rBar;
    rBar  = (rBar > (std::abs(yMax-yCent) - 2.0*hy)) ? (std::abs(yMax-yCent) - 2.0*hy) : rBar;


    if(std::abs(screenCoeff) < 1.0e-10)
    {
	nPole.initialize(xCent,yCent,rBar,maxNpoleOrder,laplaceCoeff);
	nPole.setDifferentiability(nPoleDiffOrder);
	}
	else if(std::abs(screenCoeff) < screenCoeffBound)
	{
    SnPole.initialize(xCent,yCent,rBar,maxNpoleOrder,laplaceCoeff,screenCoeff);
	SnPole.setSourceDifferentiability(nPoleDiffOrder);
	}

	//
	// Initialize 1D operator and data. This operator must be initialized in
	// single threaded mode so that the internal fftw routine is initialized
	// properly.
	//

	invPoisson1d.initialize(1.0,0.0,ny,yMin,yMax);
	vTransform.initialize(ny,yMin,yMax);
	}

	void initialize(const InversePoisson2d& H)
	{
	if(this->nx == 0) {initialize(); return;}
	initialize(H.laplaceCoeff, H.screenCoeff, H.nx, H.xMin, H.xMax, H.ny, H.yMin, H.yMax, H.extensionFactor);

	this->screenCoeffBound = H.screenCoeffBound;
	}

	void setCoefficients(double laplaceCoeff, double screenCoeff)
	{
	this->laplaceCoeff  = laplaceCoeff;
	this->screenCoeff   = screenCoeff;

    if(std::abs(screenCoeff) < 1.0e-10)
    {
	nPole.initialize(xCent,yCent,rBar,maxNpoleOrder,laplaceCoeff);
	nPole.setDifferentiability(nPoleDiffOrder);
	}
	else if(std::abs(screenCoeff) < screenCoeffBound)
	{
    SnPole.initialize(xCent,yCent,rBar,maxNpoleOrder,laplaceCoeff,screenCoeff);
	SnPole.setSourceDifferentiability(nPoleDiffOrder);
	}
	}


    void applyInverseOp(SCC::GridFunction2d& V)
    {
	double hx = (xMax-xMin)/(double)(nx);

	// Create matching coefficients

    if(std::abs(screenCoeff) < 1.0e-10)
    {
	nPole.createMomentMatchedNpole(V);
    nPole.evaluateSource(nPoleSource);
    nPole.evaluatePotential(nPolePotential);
	V -= nPoleSource;
    }
    else if(std::abs(screenCoeff) < screenCoeffBound)
    {
    SnPole.createMomentMatchedNpole(V);
    SnPole.evaluateSource(nPoleSource);
    SnPole.evaluatePotential(nPolePotential);
	V -= nPoleSource;
    }

    double dx = 1.0/double(extNx);

    long k1;
    long k1Index;
    double opTransformFactor;
    double pi = .3141592653589793e+01;

    long kConjIndex;

	for(long j = 0; j <= ny; j++)
	{
	inImag1Dx.setToValue(0.0);
	inReal1Dx.setToValue(0.0);

	for(long i = 0; i <= nx; i++)
	{
	inReal1Dx(i + extXoffset) = V.Values(i,j);
	}

	DFT.fftw1d_forward(inReal1Dx, inImag1Dx, outReal1Dx, outImag1Dx);

	for(long i = 0; i < (long)(extNx/2) + 1; i++)
	{
	realData1Dx1D(i,j) = outReal1Dx(i);
    imagData1Dx1D(i,j) = outImag1Dx(i);
	}
	}


    for(k1 = -(extNx/2); k1 <= 0; k1++)
    {
    k1Index           =  k1 + (extNx/2);
    opTransformFactor = -laplaceCoeff*(((2.0*pi*k1*dx)*(2.0*pi*k1*dx))/(hx*hx)) + screenCoeff;
    invPoisson1d.setCoefficients(laplaceCoeff,opTransformFactor);

    // Real component

    for(long j = 0; j <= ny; j++)
    {
    vTransform.Values(j)     = realData1Dx1D(k1Index,j);
    }

    invPoisson1d.applyInverseOp(vTransform);

    for(long j = 0; j <= ny; j++)
    {
    realData1Dx1D(k1Index,j) = vTransform.Values(j);
    }

    // Imag component

    for(long j = 0; j <= ny; j++)
    {
    vTransform.Values(j)     = imagData1Dx1D(k1Index,j);
    }

    invPoisson1d.applyInverseOp(vTransform);

    for(long j = 0; j <= ny; j++)
    {
    imagData1Dx1D(k1Index,j) = vTransform.Values(j);
    }
    }

    // Inverse transform

	for(long j = 0; j <= ny; j++)
	{
    for(k1 = -(extNx/2); k1 <= 0; k1++)
    {
    k1Index           =  k1 + (extNx/2);
 	inReal1Dx(k1Index) = realData1Dx1D(k1Index,j);
    inImag1Dx(k1Index) = imagData1Dx1D(k1Index,j);
    }

    for(k1 = 1; k1 <= (extNx-1)/2; k1++)
    {
    k1Index            =  k1 + (extNx/2);
    kConjIndex         = -k1 + (extNx/2);
 	inReal1Dx(k1Index) =  realData1Dx1D(kConjIndex,j);
    inImag1Dx(k1Index) = -imagData1Dx1D(kConjIndex,j);
    }

    DFT.fftw1d_inverse(inReal1Dx, inImag1Dx, outReal1Dx, outImag1Dx);

    for(long i = 0; i <= nx; i++)
	{
	V.Values(i,j) = outReal1Dx(i + extXoffset);
	}

	}
    // Add in contribution from the nPoles or SnPoles

    if(std::abs(screenCoeff) < 1.0e-10)
    {
    	V += nPolePotential;
    }
    else if(std::abs(screenCoeff) < screenCoeffBound)
    {
		V += nPolePotential;
    }

    }


    double extensionFactor;
	long        extXoffset;
    double       extSizeX;
    long          extNx;


	double     screenCoeff;
	double    laplaceCoeff;

	long        nx;     long ny;
	double    xMin; double yMin;
	double    xMax; double yMax;


    SCC::GridFunction1d  vTransform;
    InversePoisson1d     invPoisson1d;

	// Transform data

	SCC::fftw3_1d            DFT;

	SCC::DoubleVector1d inReal1Dx;
    SCC::DoubleVector1d inImag1Dx;

	SCC::DoubleVector1d outReal1Dx;
    SCC::DoubleVector1d outImag1Dx;

	SCC::DoubleVector2d realData1Dx1D;
    SCC::DoubleVector2d imagData1Dx1D;


    // N-pole and Screened N-pole

    PoissonNpole2d                 nPole;
    ScreenedNpole2d               SnPole;

   	SCC::GridFunction2d    nPoleSource;
   	SCC::GridFunction2d nPolePotential;
   	int                   nPoleDiffOrder;
   	int                    maxNpoleOrder;

   	double                 screenCoeffBound;

   	double xCent; double yCent; double rBar;

};

#undef DEFAULT_EXTENSION_FACTOR_
#undef DEFAULT_MAX_NPOLE_ORDER_
#undef DEFAULT_DIFFERENTIABILITY_
#undef DEFAULT_SCREEN_BOUND_



#endif

