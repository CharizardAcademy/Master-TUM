#include "helper.h"
#include "init.h"

int read_parameters( const char *szFileName,       /* name of the file */
                    double *Re,                /* reynolds number   */
                    double *UI,                /* velocity x-direction */
                    double *VI,                /* velocity y-direction */
                    double *PI,                /* pressure */
                    double *GX,                /* gravitation x-direction */
                    double *GY,                /* gravitation y-direction */
                    double *t_end,             /* end time */
                    double *xlength,           /* length of the domain x-dir.*/
                    double *ylength,           /* length of the domain y-dir.*/
                    double *dt,                /* time step */
                    double *dx,                /* length of a cell x-dir. */
                    double *dy,                /* length of a cell y-dir. */
                    int  *imax,                /* number of cells x-direction*/
                    int  *jmax,                /* number of cells y-direction*/
                    double *alpha,             /* uppwind differencing factor*/
                    double *omg,               /* relaxation factor */
                    double *tau,               /* safety factor for time step*/
                    int  *itermax,             /* max. number of iterations  */
		                               /* for pressure per time step */
                    double *eps,               /* accuracy bound for pressure*/
		    double *dt_value,           /* time for output */
        int TypeQuestion,   /*Control a fluid with boundary condition question (=1) or heat question(=2)*/
        double *Ti,
        double *Pr,
        double *beta,
        double *INUI,
        double *INVI
        )
{
   READ_DOUBLE( szFileName, *xlength );
   READ_DOUBLE( szFileName, *ylength );

   READ_DOUBLE( szFileName, *Re    );
   READ_DOUBLE( szFileName, *t_end );
   READ_DOUBLE( szFileName, *dt    );

   READ_INT   ( szFileName, *imax );
   READ_INT   ( szFileName, *jmax );

   READ_DOUBLE( szFileName, *omg   );
   READ_DOUBLE( szFileName, *eps   );
   READ_DOUBLE( szFileName, *tau   );
   READ_DOUBLE( szFileName, *alpha );

   READ_INT   ( szFileName, *itermax );
   READ_DOUBLE( szFileName, *dt_value );

   READ_DOUBLE( szFileName, *UI );
   READ_DOUBLE( szFileName, *VI );
   READ_DOUBLE( szFileName, *GX );
   READ_DOUBLE( szFileName, *GY );
   READ_DOUBLE( szFileName, *PI );

   *dx = *xlength / (double)(*imax);
   *dy = *ylength / (double)(*jmax);

   if (TypeQuestion==2){
     READ_DOUBLE( szFileName, *Ti );
     READ_DOUBLE( szFileName, *Pr );
     READ_DOUBLE( szFileName, *beta );
   }
   READ_DOUBLE( szFileName, *INUI );
   READ_DOUBLE( szFileName, *INVI );
   return 1;
}

//initialize the U, V, P, T matrices
void init_uvp(
  double UI,
  double VI,
  double PI,
  int imax,
  int jmax,
  double **U,
  double **V,
  double **P,
  int TypeQuestion,
  double Ti,
  double **T)
{
    init_matrix(U,0,imax,0,jmax+1, UI);
    init_matrix(V,0,imax+1,0,jmax, VI);
    init_matrix(P,0,imax+1,0,jmax+1, PI);
    init_matrix(T,0,imax+1,0,jmax+1, Ti);
}

//initialize Flags. each Flag case is a int type between 0 and 256+15=241. Its bit representation represent:
//fluid/ no-slip/ free-slip/ outflow/ inflow/ B_N/ B_S/ B_W/ B_E
void init_flag(
  char* problem,
  char* geometry,
  int imax,
  int jmax,
  int **Flag,
  double **Geo
){
  int **aux=imatrix(0,imax+1,0,jmax+1);
  aux=read_pgm(geometry);
  for (int j=0;j<=jmax+1;j++){
    for (int i=0;i<=imax+1;i++){
      Geo[i][j]=(double)aux[i][j];
      Flag[i][j]=0;
      if (aux[i][j]==4){//fluid
        Flag[i][j]+=256;
      }
      if (aux[i][j]==0){//no-slip
        Flag[i][j]+=128;
      }
      if (aux[i][j]==1){//free-slip
        Flag[i][j]+=64;
      }
      if (aux[i][j]==2){//outflow
        Flag[i][j]+=32;
      }
      if (aux[i][j]==3){//inflow
        Flag[i][j]+=16;
      }
      if ((j<(jmax+1))&&(aux[i][j+1]==4)){//BN
        Flag[i][j]+=8;
      }
      if ((j>0)&&(aux[i][j-1]==4)){//BS
        Flag[i][j]+=4;
      }
      if ((i>0)&&(aux[i-1][j]==4)){//BW
        Flag[i][j]+=2;
      }
      if ((i<(imax+1))&&(aux[i+1][j]==4)){//BE
        Flag[i][j]+=1;
      }
      //assert only one field from 5 cases. This assertion is useless according to how we build Flag[i][j]
      //assert((Flag[i][j]&240==128)||(Flag[i][j]&240==64)||(Flag[i][j]&240==32)||(Flag[i][j]&240==16)||(Flag[i][j]&240==0));//&& strcat(strcat(strcat("Error! Fields set unproperly at point ",itoa(i)),", "),itoa(j)));
      //assert fluid or obstacle without 2 opposite fluid cells
      assert((Flag[i][j]>255)||((Flag[i][j]&12)!=12));//NS not same time
      assert((Flag[i][j]>255)||((Flag[i][j]&3)!=3));//WE not same time
      //The above 2 assetions can also make sure that we have not an obstacle cell with more than 2 adjacent fluid cells
    }
  }
  free_imatrix(aux, 0, imax+1, 0, jmax+1);
}
