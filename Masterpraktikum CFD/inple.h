//
// Created by 高英强 on 2018/4/21.
//

#include <stdio.h>
#include "helper.h"
#ifndef CFD_EX1_INPLE_H
#define CFD_EX1_INPLE_H

#endif //CFD_EX1_INPLE_H

void init_uvp(double UI, double VI, double PI, int imax, int jmax, double **U, double **V, double **P);

void calculate_dt(double Re, double tau, double dt, double dx, double dy, int imax, int jmax, double U, double V);

void boundaryvalues(int imax, int jmax, double U, double V);

void calculate_fg(double Re, double GX, double GY, double alpha, double dt, double dx, double dy, int imax, int jmax, double U, double V, double F, double G);

void calculate_rs(double dt, double dx, double dy, int imax, int jmax, double F, double G, double RS);

void calculate_uv(double dt, double dx, double dy, int imax, int jmax, double U, double V, double F, double G, double P);