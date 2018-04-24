//
// Created by 高英强 on 2018/4/22.
//

#include <stdio.h>
#include "helper.h"
#include "init.h"
#include "math.h"

void calculate_dt(double Re, double tau, double *dt, double dx, double dy, int imax, int jmax, double **U, double **V){
    //judge whether tau is positive
    if(tau>0){

        //find umax and vmax
        double umax = 0;
        double vmax = 0;

        for(int i=0;i<=imax;i++){
            for(int j=0;j<=jmax;j++){
                if(fabs(U[i][j])>umax){
                    umax = U[i][j];
                }
                if(fabs(V[i][j])>vmax){
                    vmax = V[i][j];
                }
            }
        }

        //compute dt
        double temp = 0;
        temp = fmin((Re/2.0)*(1.0/((1.0/(dx*dx))+ (1.0/(dy*dy)))), (dx/umax));
        double temp2 = 0;
        temp2 = fmin(temp, (dy/vmax));
        *dt = temp2 * tau;
    }
}

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

void calculate_fg(double Re, double GX, double GY, double alpha, double dt, double dx, double dy, int imax, int jmax, double **U, double **V, double **F, double **G){
    for(int i=1;i<=imax-1;i++){
        for(int j=1;j<=jmax;j++){
            //compute Fij
            double discreU_term1 = (U[i+1][j] - 2.0 * U[i][j] + U[i-1][j])/(dx*dx);
            double discreU_term2 = (U[i][j+1] - 2.0 * U[i][j] + U[i][j-1])/(dy*dy);
            double discreU_term3_part1 = (1.0/dx) * (pow((U[i][j] + U[i+1][j])/2.0, 2.0) - pow((U[i-1][j] + U[i][j])/2.0, 2.0));
            double discreU_term3_part2 = alpha * (1/dx) * ((fabs(U[i][j]+U[i+1][j])*(U[i][j]-U[i+1][j]))/4 - (fabs(U[i-1][j]+U[i][j])*(U[i-1][j]-U[i][j]))/4);
            double discreU_term3 = discreU_term3_part1 + discreU_term3_part2;
            double discreU_term4_part1 = (1.0/dy) * (((V[i][j]+V[i+1][j])*(U[i][j]+U[i][j+1]))/4.0 - ((V[i][j-1]+V[i+1][j-1])*(U[i][j-1]+U[i][j]))/4.0);
            double discreU_term4_part2 = alpha * (1.0/dy) * ((fabs(V[i][j]+V[i+1][j])*(U[i][j]-U[i][j+1]))/4.0 - (fabs(V[i][j-1]+V[i+1][j-1])*(U[i][j-1]-U[i][j]))/4.0);
            double discreU_term4 = discreU_term4_part1 + discreU_term4_part2;

            F[i][j] = U[i][j] + dt * ((1.0/Re) * (discreU_term1 + discreU_term2) - discreU_term3 - discreU_term4 + GX);

            //compute Gij
            double discreV_term1 = (V[i+1][j] - 2.0*V[i][j] + V[i-1][j])/(dx*dx);
            double discreV_term2 = (V[i][j+1] - 2.0*V[i][j] + V[i][j-1])/(dy*dy);
            double discreV_term3_part1 = (1.0/dx) * (((U[i][j]+U[i][j+1])*(V[i][j]+V[i+1][j]))/4.0 - ((U[i-1][j]+U[i-1][j+1])*(V[i-1][j]+V[i][j]))/4.0);
            double discreV_term3_part2 = alpha * (1.0/dx) * ((fabs(U[i][j]+U[i][j+1])*(V[i][j]-V[i+1][j]))/4.0 - (fabs(U[i-1][j]+U[i-1][j+1])*(V[i-1][j]-V[i][j]))/4.0);
            double discreV_term3 = discreV_term3_part1 + discreV_term3_part2;
            double discreV_term4_part1 = (1.0/dy) * (pow((V[i][j]+V[i][j+1])/2.0, 2.0) - pow((V[i][j-1]+V[i][j])/2.0, 2.0));
            double discreV_term4_part2 = alpha * (1.0/dy) * ((fabs(V[i][j]+V[i][j+1])*(V[i][j]-V[i][j+1]))/4.0 - (fabs(V[i][j-1]+V[i][j])*(V[i][j-1]-V[i][j]))/4.0);
            double discreV_term4 = discreV_term4_part1 + discreV_term4_part2;
            G[i][j] = V[i][j] + dt * ((1.0/Re) * (discreV_term1 + discreV_term2) - discreV_term3 - discreV_term4 + GY);
        }
    }

    //apply boundary condition to Fij and Gij
    for(int i=1;i<=imax;i++){
        G[i][0] = V[i][0];
        G[i][jmax] = V[i][jmax];
    }
    for(int j=1;j<=jmax+1;j++){
        F[0][j] = U[0][j];
        F[imax][j] = U[imax][j];
    }
}

//the index for looping rs is 1...imax and 1...jmax?
void calculate_rs(double dt, double dx, double dy, int imax, int jmax, double **F, double **G, double **RS){
    for(int i=1;i<=imax;i++){
        for(int j=1;j<=jmax;j++){
            RS[i][j] = (1.0/dt) * (((F[i][j] - F[i-1][j])/dx) + ((G[i][j] - G[i][j-1])/dy));
        }
    }
}

void calculate_uv(double dt, double dx, double dy, int imax, int jmax, double **U, double **V, double **F, double **G, double **P){
    //compute u
    for(int i=1;i<=imax-1;i++){
        for(int j=1;j<=jmax;j++){
            U[i][j] = F[i][j] - (dt/dx) * (P[i+1][j] - P[i][j]);
        }
    }
    //compute v
    for(int i=1;i<imax+1;i++){
        for(int j=1;j<jmax;j++){
            V[i][j] = G[i][j] - (dt/dy) * (P[i][j+1] - P[i][j]);
        }
    }
}





