#include "boundary_val.h"
#include "helper.h"
#include <math.h>

//set the boundary conditions
void boundaryvalues(
  int imax,
  int jmax,
  double **U,
  double **V,
  int **Flag,
  int NameQuestion,
  double Th,
  double Tc,
  double **Temp,
  double INUI,
  double INVI)
{
    //Internal Obstacles
    for (int j=1;j<=jmax;j++){
        for (int i=1;i<=imax;i++){
            //Only look at boundary cells
            if ((Flag[i][j]<=255)&&((Flag[i][j]%16)>0)){//Not fluid and not surrounded by obstacles
                if (Flag[i][j]%16==10){//NW=1010
                    U[i-1][j]=0;
                    U[i][j]=-U[i][j+1];
                    V[i][j]=0;
                    V[i][j-1]=-V[i-1][j-1];
                }
                else if (Flag[i][j]%16==9){//NE==1001
                    U[i-1][j]=-U[i-1][j+1];
                    U[i][j]=0;
                    V[i][j]=0;
                    V[i][j-1]=-V[i+1][j-1];
                }
                else if (Flag[i][j]%16==6){//SW=0110
                    U[i-1][j]=0;
                    U[i][j]=-U[i][j-1];
                    V[i][j]=-V[i-1][j];
                    V[i][j-1]=0;
                }
                else if (Flag[i][j]%16==5){//SE=0101
                    U[i-1][j]=-U[i-1][j-1];
                    U[i][j]=0;
                    V[i][j]=-V[i+1][j];
                    V[i][j-1]=0;
                }
                else if (Flag[i][j]%16==8){//N=1000
                    U[i-1][j]=-U[i-1][j+1];
                    U[i][j]=-U[i][j+1];
                    V[i][j]=0;
                }
                else if (Flag[i][j]%16==4){//S=0100
                    U[i-1][j]=-U[i-1][j-1];
                    U[i][j]=-U[i][j-1];
                    V[i][j-1]=0;
                }
                else if (Flag[i][j]%16==2){//W=0010
                    U[i-1][j]=0;
                    V[i][j]=-V[i-1][j];
                    V[i][j-1]=-V[i-1][j-1];
                }
                else if (Flag[i][j]%16==1){//E=0001
                    U[i][j]=0;
                    V[i][j]=-V[i+1][j];
                    V[i][j-1]=-V[i+1][j-1];
                }
            }
            else if ((Flag[i][j]<=255)&&((Flag[i][j]%16)==0)){//Inner obstacle surrounded by 4 obstacles
                U[i][j]=0;
                V[i][j]=0;
            }
        }
    }
    //External Boundary Condition
    for (int j=1;j<=jmax; j++){

        if((Flag[0][j]&128)==128){//No-slip=1000 0000
            U[0][j]=0;
            V[0][j]=-V[1][j];
        }
        else if((Flag[0][j]&64)==64){//Free-slip=0100 0000
            U[0][j]=0;
            V[0][j]=V[1][j];
        }
        else if((Flag[0][j]&32)==32){//Outflow=0010 0000
            U[0][j]=U[1][j];
            V[0][j]=V[1][j];
        }
        else if((Flag[0][j]&16)==16){//Inflow=0001 0000
            spec_boundary_vals(0,j,U,V,INUI, INVI);
        }

        if((Flag[imax][j]&128)==128){//No-slip
            U[imax][j]=0;
        }
        else if((Flag[imax][j]&64)==64){//Free-slip
            U[imax][j]=0;
        }
        else if((Flag[imax][j]&32)==32){//Outflow
            U[imax][j]=U[imax-1][j];
        }
        else if((Flag[imax][j]&16)==16){//Inflow
            spec_boundary_vals(imax,j,U,V,INUI, INVI);
        }

        if((Flag[imax+1][j]&128)==128){//No-slip
            V[imax+1][j]=-V[imax][j];
        }
        else if((Flag[imax+1][j]&64)==64){//Free-slip
            V[imax+1][j]=V[imax][j];
        }
        else if((Flag[imax+1][j]&32)==32){//Outflow
            V[imax+1][j]=V[imax][j];
        }
        else if((Flag[imax+1][j]&16)==16){//Inflow
            spec_boundary_vals(imax+1,j,U,V,INUI, INVI);
        }
    } 
    for (int i=1;i<=imax; i++){

        if((Flag[i][0]&128)==128){//No-slip
            U[i][0]=-U[i][1];
            V[i][0]=0;
        }
        else if((Flag[i][0]&64)==64){//Free-slip
            U[i][0]=U[i][1];
            V[i][0]=0;
        }
        else if((Flag[i][0]&32)==32){//Outflow
            U[i][0]=U[i][1];
            V[i][0]=V[i][1];
        }
        else if((Flag[i][0]&16)==16){//inflow
            spec_boundary_vals(i,0,U,V,INUI, INVI);
        }

        if((Flag[i][jmax+1]&128)==128){//No-slip
            U[i][jmax+1]=-U[i][jmax];
        }
        else if((Flag[i][jmax+1]&64)==64){//Free-slip
            U[i][jmax+1]=U[i][jmax];
        }
        else if((Flag[i][jmax+1]&32)==32){//Outflow
            U[i][jmax+1]=U[i][jmax];
        }
        else if((Flag[i][jmax+1]&16)==16){//Inflow
            spec_boundary_vals(i,jmax+1,U,V,INUI, INVI);
        }

        if((Flag[i][jmax]&128)==128){//No-slip
            V[i][jmax]=0;
        }   
        else if((Flag[i][jmax]&64)==64){//Free-slip
            V[i][jmax]=0;
        }
        else if((Flag[i][jmax]&32)==32){//Outflow
            V[i][jmax]=V[i][jmax-1];
        }
        else if((Flag[i][jmax]&16)==16){//Inflow
            spec_boundary_vals(i,jmax+1,U,V,INUI, INVI);
        }
        
    }
    //Boundary condition of Temperature
    if (NameQuestion==3){//natural convection
        for (int i=1;i<=imax;i++){
            Temp[i][0]=Temp[i][1];
            Temp[i][jmax+1]=Temp[i][jmax];
        }
        for (int j=1;j<=jmax;j++){
            Temp[0][j]=2.0*Th-Temp[1][j];
            Temp[imax+1][j]=2.0*Tc-Temp[imax][j];
        }

    }
    else if (NameQuestion==4){//heat trap
        for (int i=1;i<=imax;i++){
            Temp[i][0]=Temp[i][1];
            Temp[i][jmax+1]=Temp[i][jmax];
        }
        for (int j=1;j<=jmax;j++){
            Temp[0][j]=2.0*Th-Temp[1][j];
            Temp[imax+1][j]=2.0*Tc-Temp[imax][j];
        }

    }
    else if (NameQuestion==5){//rayleigh benard convection
        for (int i=1;i<=imax;i++){
            Temp[i][0]=2.0*Th-Temp[i][1];
            Temp[i][jmax+1]=2.0*Tc-Temp[i][jmax];
        }
        for (int j=1;j<=jmax;j++){
            Temp[0][j]=Temp[1][j];
            Temp[imax+1][j]=Temp[imax][j];
        }

    }
    else if (NameQuestion==7){//self defined problem with left th right tc
        for (int i=1;i<=imax;i++){
            Temp[i][0]=Temp[i][1];
            Temp[i][jmax+1]=Temp[i][jmax];
        }
        for (int j=1;j<=jmax;j++){
            Temp[0][j]=2.0*Th-Temp[1][j];
            Temp[imax+1][j]=2.0*Tc-Temp[imax][j];
        }

    }
    else if (NameQuestion==8){//self defined problem with down th upper tc
        for (int i=1;i<=imax;i++){
            Temp[i][0]=2.0*Th-Temp[i][1];
            Temp[i][jmax+1]=2.0*Tc-Temp[i][jmax];
        }
        for (int j=1;j<=jmax;j++){
            Temp[0][j]=Temp[1][j];
            Temp[imax+1][j]=Temp[imax][j];
        }

    }
}

//set boundary for inflow
void spec_boundary_vals(
  int i,
  int j,
  double **U,
  double **V,
  double INUI,
  double INVI
){
    U[i][j]=INUI;
    V[i][j]=INVI;
}
