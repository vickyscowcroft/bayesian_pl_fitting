#include <math.h>
#include <stdlib.h>
#include <gsl/gsl_integration.h>
//#include <stdio.h>

double f(double d, void* fdata){
    double * double_fdata = (double*) fdata;

    double par_obs = double_fdata[0];
    double var_par = double_fdata[1];
    double DM = double_fdata[2];
    double var_DM = double_fdata[3];
    double par_offset = double_fdata[4];
    double L = double_fdata[5];

    //printf("Data = %f, %f, %f, %f, %f\n", d, par_corrected, var_par, DM, var_DM);
    return pow(d, 2) * exp(-0.5 * (2*d/L + pow(par_obs - (1./d + par_offset), 2.)/var_par + pow(DM - (5*log10(d) - 5), 2.)/var_DM));
}

double integrate_likelihood(const double xmin, const double xmax, double *fdata, double rel_tolerance) {
    gsl_integration_workspace * w = gsl_integration_workspace_alloc (1000);

    gsl_function F;
    F.function = &f;
    F.params = fdata;

    double result, err;

    gsl_integration_qag (&F, xmin, xmax, 0, rel_tolerance, 1000, 1, w, &result, &err);

    gsl_integration_workspace_free (w);
  
    return result;
}
