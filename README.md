# bayesian_pl_fitting

Adaptation of the Bayesian fitting code from Sesar et al. (2017)

Original Sesar et al. paper: https://ui.adsabs.harvard.edu/#abs/2017ApJ...838..107S/abstract



### Requires GNU scientific library:

https://www.gnu.org/software/gsl/

On ubuntu may already be installed in /usr/include/gsl


### Compiling likelihood_function_quad:

See command in README file and change paths accordingly:

* For ubuntu, try:

`` -L/usr/local/lib and -I/usr/include/ ``

* If these don't work we can put in a ticket with BUCS.


## Running:

* This version currently only running in Python 2.7

* Can be run from command line as

`` ./fit_PLRs.py ``

or in ipython with 

`` ipython --pylab ``

``` python
run -i ./fit_PLRs.py 
```


