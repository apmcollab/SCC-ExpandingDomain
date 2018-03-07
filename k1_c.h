#include <cmath>
#include <cfloat>
using namespace std;

#ifndef _BesselK1_
#define _BesselK1_

/* k1.f -- translated by f2c (version 20100827).
  from http://www.netlib.org/specfun
 
Modifications:

removed static 
doublereal <- double
integer    <- long 
inserted machine constants from cfloat (DBL_EPSILON,DBL_MIN, DBL_MAX)

Chris Anderson (C) UCLA July 8, 2015

This class has no dependencies other than C++ header files 
*/
/*
#############################################################################
#
# Modification of the f2c translation of k1.f by 
# W. J. Cody and Laura Stoltz 
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

class BesselK1
{

public :

/* -------------------------------------------------------------------- */
/* This function program computes approximate values for the */
/* modified Bessel function of the second kind of order one */
/* for arguments  XLEAST .LE. ARG .LE. XMAX. */
/* Authors: W. J. Cody and Laura Stoltz */
/* Latest modification: January 28, 1988 */

double besk1(double x) const
{

    double ret_val;
    long jint;
    double result;

    jint = 1;
    calck1_(&x, &result, &jint);
    ret_val = result;
    return ret_val;
}  

/*   This function program computes approximate values for the */
/*   modified Bessel function of the second kind of order one */
/*   multiplied by the exponential function, for arguments */
/*   XLEAST .LE. ARG .LE. XMAX. */
/*   Authors: W. J. Cody and Laura Stoltz */
/*   Latest modification: January 28, 1988 */


double besek1(double x) const
{
    double ret_val;
    long jint;
    double result;

    jint = 2;
    calck1_(&x, &result, &jint);
    ret_val = result;
    return ret_val;

}

private: 


int calck1_(double *arg, double *result, long *jint) const
{
    /* Initialized data */

     double one = 1.;
     double g[3] = { -305.07151578787595807,43117.653211351080007,
	    -2706232.2985570842656 };
     double pp[11] = { .064257745859173138767,7.558458463117603081,
	    131.82609918569941308,810.94256146537402173,2312.374220916887155,
	    3454.0675585544584407,2859.0657697910288226,1331.948643318322199,
	    341.2295348680131291,44.137176114230414036,2.2196792496874548962 }
	    ;
     double qq[9] = { 36.001069306861518855,330.31020088765390854,
	    1208.2692316002348638,2118.100048717194381,1944.8440788918006154,
	    969.29165726802648634,259.51223655579051357,34.552228452758912848,
	    1.7710478032601086579 };
     double zero = 0.;
     double xleast = DBL_MIN;     // 2.23e-308;
     double xsmall = DBL_EPSILON; // 1.11e-16;
     double xinf   = DBL_MAX;     // 1.79e308;
     double xmax = 705.343;
     double p[5] = { .4812707045687844231,99.991373567429309922,
	    7188.5382604084798576,177333.2403514701563,719389.20065420586101 }
	    ;
     double q[3] = { -281.43915754538725829,37264.298672067697862,
	    -2214937.4878243304548 };
     double f[5] = { -.2279559082695500239,-53.103913335180275253,
	    -4505.1623763436087023,-147580.69205414222471,
	    -1353116.1492785421328 };

    /* Builtin functions */
    //double log(double), exp(double), sqrt(double);

    /* Local variables */
     long i__;
     double x, xx, sumf, sumg, sump, sumq;

    x = *arg;
    if (x < xleast) {
/* -------------------------------------------------------------------- */
/*  Error return for  ARG  .LT. XLEAST */
/* -------------------------------------------------------------------- */
	*result = xinf;
    } else if (x <= one) {
/* -------------------------------------------------------------------- */
/*  XLEAST .LE.  ARG  .LE. 1.0 */
/* -------------------------------------------------------------------- */
	if (x < xsmall) {
/* -------------------------------------------------------------------- */
/*  Return for small ARG */
/* -------------------------------------------------------------------- */
	    *result = one / x;
	} else {
	    xx = x * x;
	    sump = ((((p[0] * xx + p[1]) * xx + p[2]) * xx + p[3]) * xx + p[4]
		    ) * xx + q[2];
	    sumq = ((xx + q[0]) * xx + q[1]) * xx + q[2];
	    sumf = (((f[0] * xx + f[1]) * xx + f[2]) * xx + f[3]) * xx + f[4];
	    sumg = ((xx + g[0]) * xx + g[1]) * xx + g[2];
	    *result = (xx * log(x) * sumf / sumg + sump / sumq) / x;
	    if (*jint == 2) {
		*result *= exp(x);
	    }
	}
    } else if (*jint == 1 && x > xmax) {
/* -------------------------------------------------------------------- */
/*  Error return for  ARG  .GT. XMAX */
/* -------------------------------------------------------------------- */
	*result = zero;
    } else {
/* -------------------------------------------------------------------- */
/*  1.0 .LT.  ARG */
/* -------------------------------------------------------------------- */
	xx = one / x;
	sump = pp[0];
	for (i__ = 2; i__ <= 11; ++i__) {
	    sump = sump * xx + pp[i__ - 1];
/* L120: */
	}
	sumq = xx;
	for (i__ = 1; i__ <= 8; ++i__) {
	    sumq = (sumq + qq[i__ - 1]) * xx;
/* L140: */
	}
	sumq += qq[8];
	*result = sump / sumq / sqrt(x);
	if (*jint == 1) {
	    *result *= exp(-x);
	}
    }
    return 0;
/* ---------- Last line of CALCK1 ---------- */
} /* calck1_ */


};

#endif

