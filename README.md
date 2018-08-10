# BATDOG

### *The <font color=red>B</font>ayesian <font color=red>A</font>s<font color=red>t</font>rometric <font color=red>D</font>ust Cart<font color=red>og</font>rapher*

![THE BATDOG](BATDOG.jpg?raw=true)

A Python notebook for processing of parallaxes, comparison with variable star distances and inference of line of sight interstellar extinction to said variable stars. It supports a novel prior on distance that allows for existing distance data to be used as a prior on parallaxes, and can use calibration stars to solve for a systematic error in parallax measurements (such as the Gaia zero-point offset.) It also simulateously fits for/verifies Leavitt law parameters.


## Getting started

### Pre-requisites

* Python 3 (development was conducted on 3.6)
* Jupyter to open the main notebook file.

And the following packages - with brackets around the versions that were used in development. Lower (or higher) versions may cause unintended behaviour. *Italicised* packages are non-essential, but recommended.

* numpy (1.14.5)
* scipy (1.1)
* pandas (0.23.1)
* matplotlib (2.2.2)
* emcee (2.2.1)
* psutil (5.4.6)
* *corner* (2.0.1)
* *astropy* (3.0.3)

*Don't have admin privileges on your machine? It sucks, doesn't it! You can install packages to your user directory and solve this problem with the --user flag when installing packages with pip.*

### Optional extras

The following Jupyter notebook extensions were used when developing the code, and make the long notebook file *much* more user-friendly. Instructions on how to install nbextensions can be found [here](https://ndres.me/post/best-jupyter-notebook-extensions/).

* Table of Contents 2 (toc2)
* Collapsible headings

### Installing

Run batdog.ipynb. Data to use should be in the data_dir (`data/` by default.) Simples!


## Loading data

### Fake data schemes

A couple of fake data loading schemes are included in section 7: a Milky-Way style exponentially decreasing space density distribution with a single scale parameter, and a complicated Magellanic-Cloud style distribution that makes use of the 'finite cloud' prior system. They're good places to start to run the script on test data.

Errors are simulated using values from Scowcroft 2016, and estimates of Gaia parallax error from Gaia Collaboration 2018 (Gaia Data Release 2. Summary of the contents and survey properties). No outliers are simulated.

Data is written into a pandas.DataFrame aptly named `data`. Annoyingly, I used symbols following the following convention throughout as column labels, in part as a shorthand and in part because I didn't know better.

#### Suffixes:

* `_true suffix`: a true value, used in fake data tests.
* `_exp suffix`: an experimentally measured value.
* `_sigma suffix`: the standard deviation on an experimentally measured value.
* `_X`: the name of the band this parameter is in, where X is the band name.

#### Global variables: 

* `omega_0`: global parallax offset, in mas.
* `L`: scale length parameter in the parallax prior, in parsecs.
* `a`: the multiplier of $log(P)$ in the Leavitt law.
* `b`: the intercept of the Leavitt law.
* `s`: the scatter of the Leavitt law.

#### Variables for each star:

* `omega`: parallax, in mas.
* `P`: period.
* `m`: apparent magnitude.
* `A`: extinction (in magnitudes) in a specific band.

#### Calibration (binary) star variables:

* `omega_ext`: parallaxes measured *by an external source*, in mas.
* `omega_gaia`: parallaxes measured *by the Gaia satellite*, in mas.

#### Other symbols in the code:

* `ID`: the ID of a star. Maybe just an integer, or could be its name in your survey.
* `ra`: right ascension, in degrees.
* `dec`: declination, in degrees.
* `r`: the distance to a star from the observer, in parsecs. (Not to be confused with a shorthand for the `ranges` dictionary in the priors/likelihoods section.)
* `x, y, z`: cartesian co-ordinates, in parsecs.


## Setup of priors

### Parallax priors

The type of parallax prior must be specified at the top of the file, which initialises an instance of the `ParallaxPrior` class. This can either be currently of two forms: exponentially decreasing space density (Bailer-Jones IV 2018) or 'finite cloud,' a special prior developed for BATDOG designed for use with the Magellanic clouds, but also appropriate for irregular globular clusters. By convention in the code, the initialisation of this class is called `parallax_prior_repo`.

#### A note on the finite cloud prior type

The *finite cloud* prior uses existing data on the 3D positions of stars to construct a distribution from which stars are presumed to be drawn from. This is done by projecting a 'Gaussian beam' of a certain width (corresponding to the standard deviation of the Gaussian) to weight stars. 

An optimum bin width is estimated with Knuth's rule, which shouldn't be used on weighted histograms in theory, but seemed to give the optimum answer anyway/very close to it in all cases, so is tentatively used. Prior stars are then binned into weighted distance histograms. Error on sky position is assumed to be negligible.

They are then smoothed using a Savitzky-Golay filter, which by default uses a cubic scheme. A spline fit is then performed on the data to produce callable fit functions (which again by default are cubic polynomials.) The boundaries of the prior are locked at zero which a number of added points. This improves the performance of the filter, as without multiple boundary points, it doesn't realise that the edge should remain at zero.

An attempt was made to factor in errors on star position into the finite cloud priors. However, they were typically so large that they would cause all detail in the prior to be lost, and the prior would resort to looking like a Gaussian. For enough stars in the model, error on any one star shouldn't matter.

**It is strongly recommended before using the finite cloud prior to test its performance on a siumulated data set similar to your real one. No analytical way to derive optimum beam widths, filter or fit settings is implemented, so you need to do it with your eyes.**

### Other priors

The settings for other priors can be changed in the prior and likelihoods function definition area. You'll probably want to change parameters and constants here.

Some basic checking for errors has been added (for instance, it won't try to evaluate priors on binary/calibration stars if you haven't actually specified any.)

A number of parameters have uniform priors in reasonable ranges. The scatter parameter has a Jeffreys log-uniform prior of $p(x) = 1/x$.

### Debugging

All priors and likelihoods have a `debug` mode specified by a boolean that will highlight any array indices with non-finite values (such as `-inf`, which corresponds to 0 probability in log-probability space.) This can be very helpful in working out what's wrong.

When given even a single infinitely unlikely starting guess, emcee will get '[very stuck](http://weknowmemes.com/wp-content/uploads/2012/05/i-got-stuck-so-i-went-to-sleep.jpg)' and not move walkers at all, even when it looks like it should for 99.9% of the walkers. Debug mode is your friend in this case!


## Creation of a starting guess

A key part of the code is the starting guess maker. A 100-star, 10 band simulation will have over 2100 parameters (!!!), and as such, the Monte-Carlo method is extremely slow at moving through parameter space. To complete your work before your own death, it's hence necessary to start at the right answer, and only conduct sampling to get an idea of the errors/any bi-/tri-/etc-modality in the distributions.

The starting guess maker can exclude failed guesses to allow the script to continue. However, it is **strongly** recommended that you investigate and rectify them before proceeding, as they may indicate code/data issues. Excluding stars could result in selection effects. *Improve before you exclude!*

### Code conventions here

Data is transferred from its respective data frames to a single data dictionary, `data_emcee`. This is due to speed and compatibility issues encountered with pandas.

Emcee can only take parameters in a single, 1D position vector. As such, a dictionary calles `ranges` (sometimes abbreviated `r` in functions for readability) stores the array values of each parameter type. `create_new_ranges_dict()` must be modified to allow for any changes to how many parameters are in the model.

### Starting guess likelihood maximisation

#### Step 1: user-specified parameters

The user should specify Leavitt law parameters, the scale of the prior, and a guess on the zero-point offset if no binary/calibration stars are in the model.

#### Step 2: inference of the systematic parallax offset

`omega_0` can be inferred initially from the mean difference between calibration star external and Gaia measurements. The external parallax measurements are assumed to be accurate, and are hence added to the starting guess. This step will fail if calibration stars aren't very good/few in number.

#### Step 3: variable star parallax maximisation

Firstly, the true parallax is estimated with three methods: with the given Gaia parallaxes, from evaluations of the Leavitt law for the band with the least scatter, and from the mode of the prior. The likelihoods of each parallax are computed, and the best estimator is picked. In rare cases, all estimates will be infinitely unlikely, at which point you should probably check my code or write a brute-force parallax guesser.

Then, using the Nelder-Mead scheme (which was found to be the most stable), the parallax guess is refined to perfection. Fails may occur here if the initial guesses are extremely close to infinitely unlikely values.

#### Step 4: extinction inference

Using all parameter guesses made so far, the Leavitt law is evaluated to infer an extinction value to each star. This is forced to be positive. The total `posterior()` value is then evaluated for this star.

This step will fail if any of the above are incorrect/bad guesses, or if the scatter on the Leavitt law is too small.

#### Step 5: extension to guesses for all the walkers

Lastly, guesses are made for the walkers within specified standard deviations on parameters by adding Gaussian deviates pulled from distributions with user-set standard deviations, which should be roughly the same as the error on parameters (with the exception of extinction, which should be small, as the program does not correctly consider scatter in the Leavitt law until after it has been ran.)

By default, four times as many walkers are used than dimensions. It helps emcee run efficiently to have quite a few walkers.

Walker guesses are only added if they return a finite evaluation of `posterior()`, which is important so that we don't do something stupid like try to tell BATDOG to assume the Magellanic Clouds exist at a negative distance away from us.

Setting these standard deviations really high will make it take fucking forever because there will be so many rejects. Setting `debug` to `True` in the call of `posterior()` will let you see which parameters are constantly causing zero probabilities.


## Running with `emcee` and data output

Interfacing with emcee is controlled by a number of functions that simplify the process. 

#### `run_emcee_for_me`

This function controls the calling of `emcee`, helps to prevent out of memory errors, predicts finish times, and allows you to specify a cut-off time to end running after (in `end_time` hours.)

Out of memory errors are avoided with empirical numbers corresponding to model parameters (dimenions, walkers, and chain length) that corresponded to roughly one quarter memory usage on a machine with 15.5GB of RAM. Using `psutil.virtual_memory().total` and parameters of the input sampler, the function estimates a maximum chain length that your system can run before encountering memory issues. 

Generally, about one quarter memory usage remains stable and fast, giving Python enough room to copy arrays around. Upto a half could be used, but the program begins to slow to a crawl by this point. This can be changed with the `memory_fraction_to_use` parameter.

If more steps have been requested than can be stored at once, the function auto-refreshes the chain. It will attempt to return the longest possible chain - for instance, if you requested 500 steps but the max stable chain length was 300, then the program would refresh after 200 steps and return 300 steps at the very end.

Every time `sampler.run_mcmc()` is called, `emcee` has to copy to a new array longer than before (due to inefficiencies in .append() methods vs. array pre-allocation.) This can only be done on a single CPU core, and slows the program down for long chains. As such, the program runs silently for `reporting_time` minutes, calling `emcee` for multiple steps at a time and increasing program speed.

#### Parameter and chain plotting functions

Two plotting functions are built-in: `plot_chains` and `plot_corner`. Both should be fully automated in plotting, and will make a plot of global parameters and parameters for a single specified star in a single specified band. Also, `plot_chains`'s lnprob plot at the top is great for looking at how convergence is going.

#### Quantile computation and value comparison functions

Additionally, a function to call `corner.quantile()` in section 6 can be used to calculate the modal value of distributions of all parameters and their +- range to 1 sigma either side. This is then used a bit lower to make plots of parameters between measured and true values.


## Tips for successful running

A number of things are in the documentation above, but here are some extras:

### Include zero-point calibration stars in your model if you're inferring for a systematic parallax offset

Due to the degeneracy between all parameters - parallax, extinction, period, etc - BATDOG will often get cheeky and use the zero-point offset parameter to shift things around because it doesn't know any better. Having at least ~5 calibration stars would solve this.

For further guidance on why inferring for parallax offsets is necessary with Gaia data in small sections of the sky, see Bailer-Jones 2018 (IV) or Gaia Collaboration 2018 (Using Gaia parallaxes).


## Author details

BATDOG was originally written by Emily Hunt (eh594@bath.ac.uk, or emily.hunt.physics@gmail.com after she graduates, or @emilydoesastro on Twitter) under the supervision of Dr Victoria Scowcroft.


## Versioning

... was (maybe foolishly) not used during development - please see GitHub commit notes as they're mostly quite informative.


## License

MIT License Emily Hunt (2018) - see LICENSE file


## Acknowledgements

This code was heavily based on work by Branimir Sesar (see 2017 paper), made use of the excellent resources provided on the use of Gaia parallaxes with first author Coryn Bailer-Jones, used a helpful 2010 data analysis guide by David Hogg, and was helped in its development by a Bayesian statistics tutorial by [Jake Vanderplas](https://github.com/jakevdp/BayesianAstronomy).




