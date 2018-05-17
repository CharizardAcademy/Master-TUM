#include "sor.h"
#include <math.h>

void sor(
  double omg,
  double dx,
  double dy,
  int    imax,
  int    jmax,
  double **P,
  double **RS,
  double *res,
  int **Flag,
  double NumberFluidCell
) {
  int i,j;
  double rloc;
  double coeff = omg/(2.0*(1.0/(dx*dx)+1.0/(dy*dy)));

  
  //Internal Obstacles
  for(i = 1; i <= imax; i++) {
    for(j = 1; j<=jmax; j++) {
      if (Flag[i][j]<=255){
        if (Flag[i][j]%16==10){//NW
          P[i][j]=(P[i-1][j]+P[i][j+1])/2.0;
        }
        else if (Flag[i][j]%16==9){//NE
          P[i][j]=(P[i+1][j]+P[i][j+1])/2.0;
        }
        else if (Flag[i][j]%16==6){//SW
          P[i][j]=(P[i-1][j]+P[i][j-1])/2.0;
        }
        else if (Flag[i][j]%16==5){//SE
          P[i][j]=(P[i+1][j]+P[i][j-1])/2.0;
        }
        else if (Flag[i][j]%16==8){//N
          P[i][j]=P[i][j+1];
        }
        else if (Flag[i][j]%16==4){//S
          P[i][j]=P[i][j-1];
        }
        else if (Flag[i][j]%16==2){//W
          P[i][j]=P[i-1][j];
        }
        else if (Flag[i][j]%16==1){//E
          P[i][j]=P[i+1][j];
        }
      }
    }
  }

  /* SOR iteration */
  for(i = 1; i <= imax; i++) {
    for(j = 1; j<=jmax; j++) {
      //Fluid case
      if (Flag[i][j]>255){
        P[i][j] = (1.0-omg)*P[i][j] + coeff*(( P[i+1][j]+P[i-1][j])/(dx*dx) + ( P[i][j+1]+P[i][j-1])/(dy*dy) - RS[i][j]);
      }
    }
  }

  /* compute the residual */
  rloc = 0;
  for(i = 1; i <= imax; i++) {
    for(j = 1; j <= jmax; j++) {
      //Fluid case
      if (Flag[i][j]>255){
        rloc += ( (P[i+1][j]-2.0*P[i][j]+P[i-1][j])/(dx*dx) + ( P[i][j+1]-2.0*P[i][j]+P[i][j-1])/(dy*dy) - RS[i][j])*
              ( (P[i+1][j]-2.0*P[i][j]+P[i-1][j])/(dx*dx) + ( P[i][j+1]-2.0*P[i][j]+P[i][j-1])/(dy*dy) - RS[i][j]);
      }
    }
  }
  rloc = rloc/NumberFluidCell;
  rloc = sqrt(rloc);
  /* set residual */
  *res = rloc;


  /* set boundary values */
  for(i = 1; i <= imax; i++) {
    P[i][0] = P[i][1];
    P[i][jmax+1] = P[i][jmax];
  }
  for(j = 1; j <= jmax; j++) {
    P[0][j] = P[1][j];
    P[imax+1][j] = P[imax][j];
  }


}

