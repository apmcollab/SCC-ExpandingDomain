/*
 * InversePoissonThin3d.h
 *
 *  Created on: July 11, 2016
 *      Author: anderson
 *
 * This is a class that provides a restricted interfaces to InversePoisson3d and
 * transforms the input data structure so that the expanding domain procedure
 * is applied to a domain in which the first coordinate is associated with the
 * thinnest width. This transformation can improve the accuracy of the
 * procedure substantially.
 *
 * To do:  Either add the data transformation code to InversePoisson3d
 * or create a more complete interface.
 *
 */

/*
#############################################################################
#
# Copyright 2016 Chris Anderson
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
#include "GridFunctionNd/SCC_GridFunction3d.h"
#include "InversePoisson3d.h"

#ifndef _InversePoissonThin3d_
#define _InversePoissonThin3d_

#define DEFAULT_INF_EXTENSION_FACTOR 2.0

class InversePoissonThin3d
{
 public:

	InversePoissonThin3d()
	{initialize();}

    InversePoissonThin3d(double laplaceCoeff, long xPanel, double xMin, double xMax,
	long yPanel, double yMin, double yMax, long zPanel, double zMin, double zMax,
	double extensionFactor = -1.0)
	{
	initialize(laplaceCoeff,xPanel,xMin,xMax,yPanel,yMin,yMax,zPanel,zMin,zMax,extensionFactor);
	}

    InversePoissonThin3d (const InversePoissonThin3d& H)
    {initialize(H);}

	void initialize()
	{
		this->inversePoisson.initialize();
		this->extFactor = -1.0;
		this->laplaceCoeff = 1.0;

		this->nx    = 0;   this->ny   = 0;   this->nz   = 0;
		this->xMin  = 0.0; this->yMin = 0.0; this->zMin = 0.0;
		this->xMax  = 0.0; this->yMax = 0.0; this->zMax = 0.0;
	}

	void initialize(const InversePoissonThin3d& H)
    {
	if(this->nx == 0) initialize();
	initialize(H.laplaceCoeff,  H.nx, H.xMin, H.xMax, H.ny, H.yMin, H.yMax, H.nz, H.zMin, H.zMax, H.extFactor);
    }

    void initialize(double laplaceCoeff, long xPanel, double xMin, double xMax,
	long yPanel, double yMin, double yMax, long zPanel, double zMin, double zMax, double extFactor = -1.0)
	{
    if(extFactor < 0) this->extFactor = DEFAULT_INF_EXTENSION_FACTOR;
    else              this->extFactor = extFactor;

    this->laplaceCoeff                = laplaceCoeff;

    this->nx    = xPanel;   this->ny   = yPanel; this->nz   = zPanel;
    this->xMin  =  xMin;    this->yMin = yMin;   this->zMin = zMin;
    this->xMax  =  xMax;    this->yMax = yMax;   this->zMax = zMax;

    //
    // Determine coordinate in which the computational domain is the thinnest
    //

    double xSize = abs(xMax-xMin);
    double ySize = abs(yMax-yMin);
    double zSize = abs(zMax-zMin);

    if      ((xSize <= ySize)&&(xSize <= zSize)){thinCoordIndex = 1;}
    else if ((ySize <= xSize)&&(ySize <= zSize)){thinCoordIndex = 2;}
    else                                        {thinCoordIndex = 3;}

    // Initialize required temporaries


    // if thinCoordIndex = 1: No need for intermediate temporary
    // if thinCoordIndex = 2; Rotate about z axis (x,y,z) ---> (y,-x, z)
    // if thinCoordIndex = 3; Rotate about y axis (x,y,z) ---> (z, y,-x)

    switch(thinCoordIndex)
	{
    case 1:
    {
      inversePoisson.initialize(laplaceCoeff, xPanel, xMin,xMax, yPanel,yMin,yMax,zPanel,zMin,zMax,this->extFactor);
    } break;

    case 2:
    {
      Vtmp.initialize(yPanel,yMin,yMax,xPanel,-xMax,-xMin,zPanel,zMin,zMax);
      inversePoisson.initialize(laplaceCoeff, yPanel,yMin,yMax,xPanel,-xMax,-xMin,zPanel,zMin,zMax,this->extFactor);
    } break;

    case 3:
    {
      Vtmp.initialize(zPanel,zMin,zMax,yPanel,yMin,yMax,xPanel,-xMax,-xMin);
      inversePoisson.initialize(laplaceCoeff, zPanel,zMin,zMax,yPanel,yMin,yMax,xPanel,-xMax,-xMin,this->extFactor);
    } break;
	}
	}

    void setMomentCancellationMax(int momentOrder)
    {
    	inversePoisson.setMaxNpoleOrder(momentOrder);
    }
    void applyInverseOp(SCC::GridFunction3d& V)
    {

	// Copy data to local V if needed

	switch(thinCoordIndex)
	{
    case 1:
    {
	inversePoisson.applyInverseOp(V);
    } break;

    case 2: //Vtmp.initialize(yPanel,yMin,yMax,xPanel,-xMax,-xMin,zPanel,zMin,zMax);
    {
    for(long i = 0; i <= ny; i++)
    {
    for(long j = 0; j <= nx; j++)
    {
    for(long k = 0; k <= nz; k++)
    {
    Vtmp(i,j,k) = V(j,i,k);
    }}}

	inversePoisson.applyInverseOp(Vtmp);

	for(long i = 0; i <= ny; i++)
    {
    for(long j = 0; j <= nx; j++)
    {
    for(long k = 0; k <= nz; k++)
    {
    V(j,i,k) =  Vtmp(i,j,k);
    }}}

    } break;

    case 3: // Vtmp.initialize(zPanel,zMin,zMax,yPanel,yMin,yMax,xPanel,-xMax,-xMin);
    {

    for(long i = 0; i <= nz; i++)
    {
    for(long j = 0; j <= ny; j++)
    {
    for(long k = 0; k <= nx; k++)
    {
    Vtmp(i,j,k) = V(k,j,i);
    }}}

	inversePoisson.applyInverseOp(Vtmp);

    for(long i = 0; i <= nz; i++)
    {
    for(long j = 0; j <= ny; j++)
    {
    for(long k = 0; k <= nx; k++)
    {
    V(k,j,i) = Vtmp(i,j,k);
    }}}

    } break;

    }

    }

    InversePoisson3d inversePoisson;

    SCC::GridFunction3d      Vtmp;

    double        extFactor;
	double      laplaceCoeff;
	int       thinCoordIndex;

	long        nx;     long ny;   long  nz;
	double    xMin; double yMin; double zMin;
	double    xMax; double yMax; double zMax;
};

#undef DEFAULT_INF_EXTENSION_FACTOR
#endif
