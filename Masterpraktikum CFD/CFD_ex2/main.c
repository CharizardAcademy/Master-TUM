#include "helper.h"
#include "visual.h"
#include "init.h"
#include "uvp.h"
#include "boundary_val.h"
#include "sor.h"
#include <stdio.h>


/**
 * The main operation reads the configuration file, initializes the scenario and
 * contains the main loop. So here are the individual steps of the algorithm:
 *
 * - read the program configuration file using read_parameters()
 * - set up the matrices (arrays) needed using the matrix() command
 * - create the initial setup init_uvp(), init_flag(), output_uvp()
 * - perform the main loop
 * - trailer: destroy memory allocated and do some statistics
 *
 * The layout of the grid is decribed by the first figure below, the enumeration
 * of the whole grid is given by the second figure. All the unknowns corresond
 * to a two dimensional degree of freedom layout, so they are not stored in
 * arrays, but in a matrix.
 *
 * @image html grid.jpg
 *
 * @image html whole-grid.jpg
 *
 * Within the main loop the following big steps are done (for some of the 
 * operations a definition is defined already within uvp.h):
 *
 * - calculate_dt() Determine the maximal time step size.
 * - boundaryvalues() Set the boundary values for the next time step.
 * - calculate_fg() Determine the values of F and G (diffusion and confection).
 *   This is the right hand side of the pressure equation and used later on for
 *   the time step transition.
 * - calculate_rs()
 * - Iterate the pressure poisson equation until the residual becomes smaller
 *   than eps or the maximal number of iterations is performed. Within the
 *   iteration loop the operation sor() is used.
 * - calculate_uv() Calculate the velocity at the next time step.
 */
int main(int argn, char** args){//.sim karman/step/natural1/natural2/trap/rbc
  
  //The initial variables read from the .dat files, together with their pointers

  double valRe=0;
  double valUI=0;
  double valVI=0;
  double valPI=0;
  double valGX=0;
  double valGY=0;
  double valt_end=0;
  double valxlength=0;
  double valylength=0;
  double valdt=0;
  double valdx=0;
  double valdy=0;
  int  valimax=0;
  int  valjmax=0;
  double valalpha=0;
  double valomg=0;
  double valtau=0;
  int  valitermax=0;
  double valeps=0;
	double valdt_value=0;

  double *Re=&valRe;
  double *UI=&valUI;
  double *VI=&valVI;
  double *PI=&valPI;
  double *GX=&valGX;
  double *GY=&valGY;
  double *t_end=&valt_end;
  double *xlength=&valxlength;
  double *ylength=&valylength;
  double *dt=&valdt;
  double *dx=&valdx;
  double *dy=&valdy;
  int  *imax=&valimax;
  int  *jmax=&valjmax;
  double *alpha=&valalpha;
  double *omg=&valomg;
  double *tau=&valtau;
  int  *itermax=&valitermax;
  double *eps=&valeps;
	double *dt_value=&valdt_value;

  double valINUI=0.0;
  double valINVI=0.0;
  double *INUI=&valINUI;
  double *INVI=&valINVI;
  //The equivalent default value for a non Heat problem so that they can use the same equations
  double valTi=0.0;
  double valBeta=0.0;
  double valPr=1.0;
  double *Ti=&valTi;
  double *beta=&valBeta;
  double *Pr=&valPr;

  //To generate filename. problem=args[1].dat, geometry=args[1].pgm, outputfilename=output/args[1]
  
  /*char Filename1[(int)(sizeof(args[1])/sizeof(args[1][0]))];
  char Filename2[(int)(sizeof(args[1])/sizeof(args[1][0]))];
  char Filename3[(int)(sizeof(args[1])/sizeof(args[1][0]))];
  char *foldername="output/";
  //char *aftername1=".dat";
  //char *aftername2=".pgm";
  char Outputfilename[(int)(sizeof(args[1])/sizeof(args[1][0]))+7];
  //char problem[(int)(sizeof(args[1])/sizeof(args[1][0]))+4];
  //char geometry[(int)(sizeof(args[1])/sizeof(args[1][0]))+4];
  strcpy(Filename1,args[1]);
  strcpy(Filename2,args[1]);
  strcpy(Filename3,args[1]);
  char* problem=strcat(Filename1,".dat");
  char* geometry=strcat(Filename2,".pgm");
  for (int i=0;i<7;i++){
    Outputfilename[i]=foldername[i];
  }
  for (int i=0;i<(int)(sizeof(args[1])/sizeof(args[1][0]));i++){
    Outputfilename[i+7]=Filename3[i];
    //problem[i]=Filename1[i];
    //geometry[i]=Filename2[i];
  }
  //for (int i=0;i<4;i++){
    //problem[(int)(sizeof(args[1])/sizeof(args[1][0]))+i]=aftername1[i];
    //geometry[(int)(sizeof(args[1])/sizeof(args[1][0]))+i]=aftername2[i];
  //}
  printf("%s\n",problem);
  printf("%s\n",geometry);
  printf("%s\n",Outputfilename);*/

  //Set flags for the question: is this a fluid or heat problem? is it karman or step or natural convection or heat trap or rbc?

  char* problem;//address of the .dat file
  char* geometry;//address of the .pgy file
  char* Outputfilename;//where to save output

  double Th=1.0;
  double Tc=0.0;
  int TypeQuestion=0;//1=fluid, 2=heat
  int NameQuestion=0;//1=karman, 2=step, 3=natural convection, 4=trap, 5=rbc, 
                     //6=general fluid, 7=general heat with left hot right cold wall, 8=general heat with down hot up cold wall
  if (strcmp(args[1],"karman")==0){
    problem="karman.dat";
    geometry="karman.pgm";
    Outputfilename="output/karman";
    NameQuestion=1;
    TypeQuestion=1;//fluid
  }
  else if (strcmp(args[1],"step")==0){
    problem="step.dat";
    geometry="step.pgm";
    Outputfilename="output/step";
    NameQuestion=2;
    TypeQuestion=1;
  }
  else if (strcmp(args[1],"natural1")==0){
    problem="natural1.dat";
    geometry="natural1.pgm";
    Outputfilename="output/natural1";
    NameQuestion=3;
    TypeQuestion=2;//heat
    Th=1.0;
    Tc=0.0;
  }
  else if (strcmp(args[1],"natural2")==0){
    problem="natural2.dat";
    geometry="natural2.pgm";
    Outputfilename="output/natural2";
    NameQuestion=3;
    TypeQuestion=2;
    Th=1.0;
    Tc=0.0;
  }
  else if (strcmp(args[1],"trap")==0){
    problem="trap.dat";
    geometry="trap.pgm";
    Outputfilename="output/trap";
    NameQuestion=4;
    TypeQuestion=2;
    Th=0.5;
    Tc=-0.5;
  }
  else if (strcmp(args[1],"trapinverse")==0){//trap questions but inverse the hot and cold wall
    problem="trap.dat";
    geometry="trap.pgm";
    Outputfilename="output/trapinverse";
    NameQuestion=4;
    TypeQuestion=2;
    Th=-0.5;
    Tc=0.5;
  }
  else if (strcmp(args[1],"rbc")==0){
    problem="rbc.dat";
    geometry="rbc.pgm";
    Outputfilename="output/rbc";
    NameQuestion=5;
    TypeQuestion=2;
    Th=294.78;
    Tc=291.20;
  }
  else{//self defined cases if not among the 5 questions. You need to use ./sim problem geometry Outputfilename (fluid/heat) (downup/leftright) [Th] [Tc]
    problem=args[1];
    geometry=args[2];
    Outputfilename=args[3];
    if (strcmp(args[4],"fluid")==0){
      NameQuestion=6;
      TypeQuestion=1;
    }
    else{
      TypeQuestion=2;
      Th=atof(args[6]);
      Tc=atof(args[7]);
      if (strcmp(args[5],"leftright")==0){//left Th rihgt Tc
        NameQuestion=7;
      }
      else{//down Th up Tc
        NameQuestion=8;
      }
    }

  }

  //read parameters from args[1].dat
  
  read_parameters(problem, Re,UI,VI,PI,GX,GY,t_end,xlength,ylength,dt,dx,dy,imax,jmax,alpha,omg,tau,itermax,eps,dt_value,TypeQuestion, Ti, Pr, beta, INUI, INVI);
    
  //reserve memory for the global matrices

  double **U = matrix ( 0 , *imax , 0 , *jmax+1 );
  double **V = matrix ( 0 , *imax+1 , 0 , *jmax );
  double **P = matrix ( 0 , *imax+1 , 0 , *jmax+1 );
  double **RS = matrix ( 1 , *imax , 1 , *jmax );
  double **F = matrix ( 0 , *imax , 1 , *jmax );
  double **G = matrix ( 1 , *imax , 0 , *jmax );
  double **Temp = matrix (0, *imax+1, 0, *jmax+1);

  init_matrix(F, 0, *imax , 1 , *jmax, 0.0);
  init_matrix(G, 1, *imax , 0 , *jmax, 0.0);
  init_matrix(RS, 1, *imax , 1, *jmax, 0.0);

  int **Flag=imatrix(0,*imax+1,0,*jmax+1);
  init_imatrix(Flag, 0,*imax+1,0,*jmax+1, 0);
  double **Geo=matrix(0,*imax+1,0,*jmax+1);
  init_matrix(Geo, 0,*imax+1,0,*jmax+1, 0.0);

  //set global variables  to compute timestamps, residual of sor, and restTime variable judges when to print out a vtk file

  double valt=0;
  int valn=0;
  int valit=0;
  double valres= valeps+1;
  double restTime=0;

  double *t=&valt;
  int *n=&valn;
  int *it=&valit;
  double *res= &valres;
  int solbeh=0;

  //initialize U,V,P, and T

  init_uvp(*UI,*VI,*PI,*imax,*jmax,U,V,P, TypeQuestion, *Ti, Temp);

  //initialize flags together with its geometry structure (Geo, which can be print out later)

  init_flag(problem,geometry, *imax, *jmax, Flag, Geo);

  //Compute the number of fluid cells in the problem, can be used in the sor iteration as a global variable

  double NumberFluidCell=0.0;
  for (int i=1;i<=*imax;i++){
    for (int j=1;j<=*jmax;j++){
      if (Flag[i][j]>255){
        NumberFluidCell+=1.0;
      }
    }
  }

  //write a vtk file at the beginning

  boundaryvalues(*imax, *jmax, U, V, Flag, NameQuestion, Th, Tc, Temp, *INUI, *INVI);
  if (TypeQuestion==1){
    write_vtkFile(Outputfilename, *n, *xlength, *ylength, *imax, *jmax, *dx, *dy, U, V, P, Geo, Temp);
  }
  else if (TypeQuestion==2){
    write_vtkFile(Outputfilename, *n, *xlength, *ylength, *imax, *jmax, *dx, *dy, U, V, P, Geo, Temp);
  }

  //main loop

  while (*t<*t_end){

    //compute dt

    if (*tau<=0){
      *dt=*dt_value;//0.005;
    }
    else{
      calculate_dt(*Re, *tau, dt, *dx, *dy, *imax, *jmax, U, V, Flag,*Pr);
    }

    //compute boundary for U,V,T

    boundaryvalues(*imax, *jmax, U, V, Flag, NameQuestion, Th, Tc, Temp, *INUI, *INVI);
    
    //compute T^(n+1) if this is a heat question

    if (TypeQuestion==2){
      calculate_temp(*Re, *Pr, *dt, *dx, *dy, *imax, *jmax, U, V, Temp, *alpha, Flag, *Ti);
    }

    //compute F, G

    calculate_fg(*Re, *GX, *GY, *alpha, *dt, *dx, *dy, *imax, *jmax, U, V, F, G, Flag, TypeQuestion,*beta, Temp);

    //compute right hand side

    calculate_rs(*dt, *dx, *dy, *imax, *jmax, F, G, RS, Flag);

    //reinitialize the counter of iteration for the sor solver

    *it=0;
    *res=*eps+1;

    //compute sor, either converge or not converge but arrive at itermax

    while ((*it<*itermax) && (*res>*eps)){
      sor(*omg, *dx, *dy, *imax, *jmax, P, RS, res, Flag,NumberFluidCell);
      *it=*it+1;
      solbeh=solbeh+1;
    }
    if (*it==*itermax){
      printf("WARNING! The SOR have not converged in this iteration!\n");
    }

    //compute new U, V

    calculate_uv(*dt, *dx, *dy, *imax, *jmax, U, V, F, G, P, Flag);
    
    
    *t+=*dt;
    restTime+=*dt; 
    *n=*n+1;

    //write a vtk file after each dt_value

    if (restTime>=*dt_value){
      boundaryvalues(*imax, *jmax, U, V, Flag, NameQuestion, Th, Tc, Temp, *INUI, *INVI);
      if (TypeQuestion==1){
        write_vtkFile(Outputfilename, *n, *xlength, *ylength, *imax, *jmax, *dx, *dy, U, V, P, Geo, Temp);
      }
      else if (TypeQuestion==2){
        write_vtkFile(Outputfilename, *n, *xlength, *ylength, *imax, *jmax, *dx, *dy, U, V, P, Geo, Temp);
      }
      restTime=restTime-*dt_value;
    }

    //print out n, dt, t, #sor, Residual in the terminal

    printf("n=");
    printf("%i",*n);
    printf(", dt=");
    printf("%f",*dt);
    printf(", t=");
    printf("%f ,\n   ",*t);
    printf("#SOR=");
    printf("%i",*it);
    printf(", Res=");
    printf("%f\n",*res);
  }

  //write out the final file

  boundaryvalues(*imax, *jmax, U, V, Flag, NameQuestion, Th, Tc, Temp, *INUI, *INVI);
  if (TypeQuestion==1){
    write_vtkFile(Outputfilename, *n, *xlength, *ylength, *imax, *jmax, *dx, *dy, U, V, P, Geo, Temp);
  }
  else if (TypeQuestion==2){
    write_vtkFile(Outputfilename, *n, *xlength, *ylength, *imax, *jmax, *dx, *dy, U, V, P, Geo, Temp);
  }
  
  //print how many iterations in sor in all 

  printf("Total #SOR=");
  printf("%i \n", solbeh);

  //free memory space

  free_matrix (U, 0 , *imax , 0 , *jmax+1 );
  free_matrix (V, 0 , *imax+1 , 0 , *jmax );
  free_matrix (P, 0 , *imax+1 , 0 , *jmax+1 );
  free_matrix (RS, 1 , *imax , 1 , *jmax );
  free_matrix (F, 0 , *imax , 1 , *jmax );
  free_matrix (G, 1 , *imax , 0 , *jmax );
  free_imatrix(Flag, 0, *imax+1, 0, *jmax+1);
  free_matrix(Temp, 0, *imax+1, 0, *jmax+1);
  return 0;
}
