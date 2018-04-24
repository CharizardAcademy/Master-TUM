#include "helper.h"
#include "visual.h"
#include "init.h"
#include "uvp.h"
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


int main(int argn, char** args){
    int imax_init = 0;
    int jmax_init = 0;
    int itermax_init = 0;
    double Re_init = 0;
    double UI_init = 0;
    double VI_init = 0;
    double PI_init = 0;
    double GX_init = 0;
    double GY_init = 0;
    double t_end_init = 0;
    double xlength_init = 0;
    double ylength_init = 0;
    double dt_init = 0;
    double dx_init = 0;
    double dy_init = 0;
    double alpha_init = 0;
    double omg_init = 0;
    double tau_init = 0;
    double eps_init = 0;
    double dt_value_init = 0;

    double *Re = &Re_init;
    double *UI = &UI_init;
    double *VI = &VI_init;
    double *PI = &PI_init;
    double *GX = &GX_init;
    double *GY = &GY_init;
    double *t_end = &t_end_init;
    double *xlength = &xlength_init;
    double *ylength = &ylength_init;
    double *dt = &dt_init;
    double *dx = &dx_init;
    double *dy = &dy_init;
    int *imax = &imax_init;
    int *jmax = &jmax_init;
    double *alpha = &alpha_init;
    double *omg = &omg_init;
    double *tau = &tau_init;
    int *itermax = &itermax_init;
    double *eps = &eps_init;
    double *dt_value = &dt_value_init;

    read_parameters("cavity100.dat",
                    Re, UI, VI, PI,
                    GX, GY, t_end, xlength,
                    ylength, dt, dx, dy,
                    imax, jmax, alpha, omg,
                    tau, itermax,eps,dt_value);

    // create matrices for U,V,P
    double **U = matrix(0, *imax+1, 0, *jmax+1);
    double **V = matrix(0, *imax+1, 0, *jmax+1);
    double **P = matrix(0, *imax+1, 0, *jmax+1);

    //create matrices for F,G
    double **F = matrix(0, *imax+1, 0, *jmax+1);
    double **G = matrix(0, *imax+1, 0, *jmax+1);

    //create matrix for RS
    double **RS = matrix(0, *imax+1, 0, *jmax+1);

    //creat variable for rs
    double rs_init = 0;
    double *rs = &rs_init;

    printf("\n \n");

    //perform the algorithm
    double t = 0;
    int step_n = 0;
    //initialize U,V,P
    init_uvp(*UI, *VI, *PI, *imax, *jmax, U, V, P);
    while(t<*t_end){
        calculate_dt(*Re, *tau, dt, *dx, *dy, *imax, *jmax, U, V);
        boundaryvalues(*imax, *jmax, U, V);
        calculate_fg(*Re, *GX, *GY, *alpha, *dt, *dx, *dy, *imax, *jmax, U, V, F, G);
        calculate_rs(*dt, *dx, *dy, *imax, *jmax, F, G, RS);
        for(int i=0;i<=*imax;i++){
            for(int j=0;j<=*jmax;j++){
                rs = &RS[i][j];
            }
        }
        int it = 0;
        while(it<*itermax && *rs>*eps){
            sor(*omg, *dx, *dy, *imax, *jmax, P, RS, rs);
            it += 1;
        }
        calculate_uv(*dt, *dx, *dy, *imax, *jmax, U, V, F, G, P);
        //output U,V,P
        t += *dt;
        step_n += 1;
    }
    //output U,V,P
    write_vtkFile("output_uvp", step_n, *xlength, *ylength, *imax, *jmax, *dx, *dy, U, V, P);
    //free the matrix storage
    free_matrix(U, 0, *imax+1, 0, *jmax+1);
    free_matrix(V, 0, *imax+1, 0, *jmax+1);
    free_matrix(P, 0, *imax+1, 0, *jmax+1);
    free_matrix(RS, 0, *imax+1, 0, *jmax+1);
    free_matrix(F, 0, *imax+1, 0, *jmax+1);
    free_matrix(G, 0, *imax+1, 0, *jmax+1);

    return 0;
}