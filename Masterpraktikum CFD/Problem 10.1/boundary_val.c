//
// Created by 高英强 on 2018/4/24.
//


#include <stdio.h>
#include "helper.h"
#include "init.h"
#include "math.h"

void boundaryvalues(int imax, int jmax, double **U, double **V){
    // right boundary
    for(int i=1;i<=imax;i++) {
        V[i][0] = 0;
        V[i][jmax] = 0;
    }
    for(int j=1;j<=jmax;j++){
        U[0][j] = 0;
        U[imax][j] = 0;
    }
    // on the four wall
    for(int i=1;i<=imax;i++){
        U[i][0] = -U[i][1];
        U[i][jmax+1] = -U[i][jmax];
    }
    for(int j=1;j<=jmax;j++){
        V[0][j] = -V[1][j];
        V[imax+1][j] = -V[imax][j];
    }
}