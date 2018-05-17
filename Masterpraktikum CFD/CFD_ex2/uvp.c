#include "uvp.h"
#include "helper.h"
#include <math.h>


/**
 * Determines the value of U and G according to the formula
 *
 * @f$ F_{i,j} := u_{i,j} + \delta t \left( \frac{1}{Re} \left( \left[
    \frac{\partial^2 u}{\partial x^2} \right]_{i,j} + \left[
    \frac{\partial^2 u}{\partial y^2} \right]_{i,j} \right) - \left[
    \frac{\partial (u^2)}{\partial x} \right]_{i,j} - \left[
    \frac{\partial (uv)}{\partial y} \right]_{i,j} + g_x \right) @f$
 *
 * @f$ i=1,\ldots,imax-1, \quad j=1,\ldots,jmax @f$
 *
 * @f$ G_{i,j} := v_{i,j} + \delta t \left( \frac{1}{Re} \left(
   \left[ \frac{\partial^2 v}{\partial x^2}\right]_{i,j} + \left[ \frac{\partial^2 v}{\partial
                   y^2} \right]_{i,j} \right) - \left[ \frac{\partial
                   (uv)}{\partial x} \right]_{i,j} - \left[
                 \frac{\partial (v^2)}{\partial y} \right]_{i,j} + g_y
               \right) @f$
 *
 * @f$ i=1,\ldots,imax, \quad j=1,\ldots,jmax-1 @f$
 *
 */
void calculate_fg(
  double Re,
  double GX,
  double GY,
  double alpha,
  double dt,
  double dx,
  double dy,
  int imax,
  int jmax,
  double **U,
  double **V,
  double **F,
  double **G,
  int **Flag,
  int TypeQuestion,
  double beta,
  double **Temp
)
{
    for (int i=1;i<imax;i++){
        for (int j=1;j<=jmax;j++){
            //Only take place on edges between 2 fluid cells
            if ((Flag[i][j]>255)&&(Flag[i+1][j]>255)){
                F[i][j]=U[i][j]+dt*(
                    ((U[i+1][j]-2.0*U[i][j]+U[i-1][j])/(dx*dx)+(U[i][j+1]-2.0*U[i][j]+U[i][j-1])/(dy*dy))/Re
                    -(((U[i][j]+U[i+1][j])/2.0)*((U[i][j]+U[i+1][j])/2.0)-((U[i-1][j]+U[i][j])/2.0)*((U[i-1][j]+U[i][j])/2.0)
                        +alpha*((fabs(U[i][j]+U[i+1][j])/2.0)*((U[i][j]-U[i+1][j])/2.0)-(fabs(U[i-1][j]+U[i][j])/2.0)*((U[i-1][j]-U[i][j])/2.0))
                    )/dx
                    -(((V[i][j]+V[i+1][j])/2.0)*((U[i][j]+U[i][j+1])/2.0)-((V[i][j-1]+V[i+1][j-1])/2.0)*((U[i][j-1]+U[i][j])/2.0)
                        +alpha*((fabs(V[i][j]+V[i+1][j])/2.0)*((U[i][j]-U[i][j+1])/2.0)-(fabs(V[i][j-1]+V[i+1][j-1])/2.0)*((U[i][j-1]-U[i][j])/2.0))
                    )/dy
                    +GX
                );
                if (TypeQuestion==2){
                    F[i][j]=F[i][j]-beta*dt*(Temp[i][j]+Temp[i+1][j])*GX/2.0;
                }
            }
            //Obstacles case
            else{
                if ((Flag[i][j]<=255)&&((Flag[i][j]&2)==2)){//W
                    F[i-1][j]=U[i-1][j];
                }
                if ((Flag[i][j]<=255)&&((Flag[i][j]&1)==1)){//E
                    F[i][j]=U[i][j];
                }
            }
        }
    }
    for (int i=1;i<=imax;i++){
        for (int j=1;j<jmax;j++){
            //Only take place on edges between 2 fluid cells
            if ((Flag[i][j]>255)&&(Flag[i][j+1]>255)){
                G[i][j]=V[i][j]+dt*(
                    ((V[i+1][j]-2.0*V[i][j]+V[i-1][j])/(dx*dx)+(V[i][j+1]-2.0*V[i][j]+V[i][j-1])/(dy*dy))/Re
                    -(((U[i][j]+U[i][j+1])/2.0)*((V[i][j]+V[i+1][j])/2.0)-((U[i-1][j]+U[i-1][j+1])/2.0)*((V[i-1][j]+V[i][j])/2.0)
                        +alpha*((fabs(U[i][j]+U[i][j+1])/2.0)*((V[i][j]-V[i+1][j])/2.0)-(fabs(U[i-1][j]+U[i-1][j+1])/2.0)*((V[i-1][j]-V[i][j])/2.0))
                    )/dx
                    -(((V[i][j]+V[i][j+1])/2.0)*((V[i][j]+V[i][j+1])/2.0)-((V[i][j-1]+V[i][j])/2.0)*((V[i][j-1]+V[i][j])/2.0)
                        +alpha*((fabs(V[i][j]+V[i][j+1])/2.0)*((V[i][j]-V[i][j+1])/2.0)-(fabs(V[i][j-1]+V[i][j])/2.0)*((V[i][j-1]-V[i][j])/2.0))
                    )/dy
                    +GY
                );
                if (TypeQuestion==2){
                    G[i][j]=G[i][j]-beta*dt*(Temp[i][j]+Temp[i][j+1])*GY/2.0;
                }
            }
            //Obstacle case
            else{
                if ((Flag[i][j]<=255)&&((Flag[i][j]&8)==8)){//N
                    G[i][j]=V[i][j];
                }
                if ((Flag[i][j]<=255)&&((Flag[i][j]&4)==4)){//S
                    G[i][j-1]=V[i][j-1];
                }
            }    
        }
    }
    //External Boundary Conditions
    for (int j=1;j<=jmax; j++){
        F[0][j]=U[0][j];
        F[imax][j]=U[imax][j];
    } 
    for (int i=1;i<=imax; i++){
        G[i][0]=V[i][0];
        G[i][jmax]=V[i][jmax];
    }
}


/**
 * This operation computes the right hand side of the pressure poisson equation.
 * The right hand side is computed according to the formula
 *
 * @f$ rs = \frac{1}{\delta t} \left( \frac{F^{(n)}_{i,j}-F^{(n)}_{i-1,j}}{\delta x} + \frac{G^{(n)}_{i,j}-G^{(n)}_{i,j-1}}{\delta y} \right)  @f$
 *
 */
void calculate_rs(
  double dt,
  double dx,
  double dy,
  int imax,
  int jmax,
  double **F,
  double **G,
  double **RS,
  int **Flag
)
{
    for (int i=1;i<=imax;i++){
        for (int j=1;j<=jmax;j++){
            //Only on fluid
            if (Flag[i][j]>255){
                RS[i][j]=((F[i][j]-F[i-1][j])/dx+(G[i][j]-G[i][j-1])/dy)/dt;
            }
        }
    }
}


/**
 * Determines the maximal time step size. The time step size is restricted
 * accordin to the CFL theorem. So the final time step size formula is given
 * by
 *
 * @f$ {\delta t} := \tau \, \min\left( \frac{Re}{2}\left(\frac{1}{{\delta x}^2} + \frac{1}{{\delta y}^2}\right)^{-1},  \frac{{\delta x}}{|u_{max}|},\frac{{\delta y}}{|v_{max}|} \right) @f$
 *
 */
void calculate_dt(
  double Re,
  double tau,
  double *dt,
  double dx,
  double dy,
  int imax,
  int jmax,
  double **U,
  double **V,
  int **Flag,
  double Pr
  )
{
    double umax=0.00001;
    double vmax=0.00001;
    for(int i = 1; i <= imax; i++) {
        for(int j = 1; j<=jmax; j++) {
            //Only on fluid
            if (Flag[i][j]>255){
                if (fabs(U[i][j])>umax){
                    umax=fabs(U[i][j]);
                }
                if (fabs(V[i][j])>vmax){
                    vmax=fabs(V[i][j]);
                }
            }
        }
    }
    *dt=tau*(fmin(Re*fmin(Pr,1.0)/(2.0*(1.0/(dx*dx)+1.0/(dy*dy))),fmin(dx/umax,dy/vmax)));
}


/**
 * Calculates the new velocity values according to the formula
 *
 * @f$ u_{i,j}^{(n+1)}  =  F_{i,j}^{(n)} - \frac{\delta t}{\delta x} (p_{i+1,j}^{(n+1)} - p_{i,j}^{(n+1)}) @f$
 * @f$ v_{i,j}^{(n+1)}  =  G_{i,j}^{(n)} - \frac{\delta t}{\delta y} (p_{i,j+1}^{(n+1)} - p_{i,j}^{(n+1)}) @f$
 *
 * As always the index range is
 *
 * @f$ i=1,\ldots,imax-1, \quad j=1,\ldots,jmax @f$
 * @f$ i=1,\ldots,imax, \quad j=1,\ldots,jmax-1 @f$
 *
 * @image html calculate_uv.jpg
 */
void calculate_uv(
  double dt,
  double dx,
  double dy,
  int imax,
  int jmax,
  double **U,
  double **V,
  double **F,
  double **G,
  double **P,
  int **Flag
)
{
    for(int i=1;i<imax;i++){
        for(int j=1;j<=jmax;j++){
            //Only take place on edges between 2 fluid cells
            if ((Flag[i][j]>255)&&(Flag[i+1][j]>255)){
                U[i][j]=F[i][j]-dt*(P[i+1][j]-P[i][j])/dx;
            }
        }
    }
    for(int i=1;i<=imax;i++){
        for(int j=1;j<jmax;j++){
            //Only take place on edges between 2 fluid cells
            if ((Flag[i][j]>255)&&(Flag[i][j+1]>255)){
                V[i][j]=G[i][j]-dt*(P[i][j+1]-P[i][j])/dy;
            }
        }
    }
}

//compute the temperature

void calculate_temp(
  double Re,
  double Pr,
  double dt,
  double dx,
  double dy,
  int imax,
  int jmax,
  double **U,
  double **V,
  double **Temp,
  double alpha,
  int **Flag,
  double Ti
){
    double **aux=matrix(1,imax,1,jmax);
    init_matrix(aux,1,imax,1,jmax,0.0);
    for (int i=1;i<=imax;i++){
        for (int j=1;j<=jmax;j++){
            if (Flag[i][j]>255){
                aux[i][j]=Temp[i][j]+dt*(
                    ((Temp[i+1][j]-2.0*Temp[i][j]+Temp[i-1][j])/(dx*dx)+(Temp[i][j+1]-2.0*Temp[i][j]+Temp[i][j-1])/(dy*dy))/(Re*Pr)
                    -(U[i][j]*((Temp[i][j]+Temp[i+1][j])/2.0)-U[i-1][j]*((Temp[i-1][j]+Temp[i][j])/2.0)
                        +alpha*(fabs(U[i][j])*((Temp[i][j]-Temp[i+1][j])/2.0)-fabs(U[i-1][j])*((Temp[i-1][j]-Temp[i][j])/2.0))
                    )/dx
                    -(V[i][j]*((Temp[i][j]+Temp[i][j+1])/2.0)-V[i][j-1]*((Temp[i][j-1]+Temp[i][j])/2.0)
                        +alpha*(fabs(V[i][j])*((Temp[i][j]-Temp[i][j+1])/2.0)-fabs(V[i][j-1])*((Temp[i][j-1]-Temp[i][j])/2.0))
                    )/dy);
            }
        }
    }
    for (int i=1;i<=imax;i++){
        for (int j=1;j<=jmax;j++){
            if (Flag[i][j]<=255){
                if (Flag[i][j]%16==10){//NW
                    aux[i][j]=(aux[i-1][j]+aux[i][j+1])/2.0;
                }
                else if (Flag[i][j]%16==9){//NE
                    aux[i][j]=(aux[i+1][j]+aux[i][j+1])/2.0;
                }
                else if (Flag[i][j]%16==6){//SW
                    aux[i][j]=(aux[i-1][j]+aux[i][j-1])/2.0;
                }
                else if (Flag[i][j]%16==5){//SE
                    aux[i][j]=(aux[i+1][j]+aux[i][j-1])/2.0;
                }
                else if (Flag[i][j]%16==8){//N
                    aux[i][j]=aux[i][j+1];
                }
                else if (Flag[i][j]%16==4){//S
                    aux[i][j]=aux[i][j-1];
                }
                else if (Flag[i][j]%16==2){//W
                    aux[i][j]=aux[i-1][j];
                }
                else if (Flag[i][j]%16==1){//E
                    aux[i][j]=aux[i+1][j];
                }
            }
        }
    }
    
    for (int i=1;i<=imax;i++){
        for (int j=1;j<=jmax;j++){
            Temp[i][j]=aux[i][j];
        }
    }
    free_matrix(aux,1,imax,1,jmax);
}
