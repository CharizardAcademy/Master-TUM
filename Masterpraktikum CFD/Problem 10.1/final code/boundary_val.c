//
// Created by 高英强 on 2018/4/24.
//


#include <stdio.h>
#include "helper.h"
#include "init.h"
#include "math.h"

void boundaryvalues(int imax, int jmax, double **U, double **V){
    // right boundary and on the four wall
    for(int i=1;i<=imax;i++) {
        U[i][0] = -U[i][1];
        V[i][0] = 0;
        U[i][jmax+1] = 2.0 - U[i][jmax];
        V[i][jmax] = 0;
    }
    for(int j=1;j<=jmax;j++){
        U[0][j] = 0;
        V[0][j] = -V[1][j];
        U[imax][j] = 0;
        V[imax+1][j] = -V[imax][j];
    }

}