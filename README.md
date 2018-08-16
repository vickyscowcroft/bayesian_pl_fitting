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

### Symbols

Data is written into a pandas.DataFrame aptly named `data`. Annoyingly, I used symbols following the following convention throughout as column labels, in part as a shorthand and in part because I didn't know better.

#### Suffixes:

* `_true` suffix: a true value, used in fake data tests.
* `_exp` suffix: an experimentally measured value.
* `_inf` suffix: a value that has been inferred by BATDOG.
* `_sigma` suffix: the standard deviation on an experimentally measured value.
* `_X`: the name of the band this parameter is in, where X is the band name.
* `_u` or `_l`: upper/lower limits on a parameter at the 1-sigma level.

#### Global variables: 

* `omega_0`: global parallax offset, in mas.
* `L`: scale length parameter in the parallax prior, in parsecs.
* `Rv`: extinction law co-efficient.

#### Variables for each photometric band:

* `a`: the multiplier of $log(P)$ in the Leavitt law.
* `b`: the multiplier of metallicity in the Leavitt law.
* `c`: the intercept of the Leavitt law.
* `s`: the scatter of the Leavitt law.

#### Variables for each star:

* `omega`: parallax, in mas.
* `A`: extinction (in magnitudes) **in the V band.**

#### Nuisance parameters for each star:

* `P`: period.
* `m`: apparent magnitude, in however many photometric bands you're using.

#### Calibration (binary) star variables:

* `omega_ext`: parallaxes measured *by an external source*, in mas.
* `omega_gaia`: parallaxes measured *by the Gaia satellite*, in mas.

#### Other symbols in the code:

* `ID`: the ID of a star. Maybe just an integer, or could be its name in your survey.
* `ra`: right ascension, in degrees.
* `dec`: declination, in degrees.
* `r`: the distance to a star from the observer, in parsecs - not to be confused with a shorthand for the `ranges` dictionary in the priors/likelihoods section.
* `x, y, z`: cartesian co-ordinates, in parsecs.


## Setup of priors

### Parallax priors

The type of parallax prior must be specified at the top of the file, which initialises an instance of the `ParallaxPrior` class. This can either be currently of two forms: exponentially decreasing space density (Bailer-Jones IV 2018) or 'finite cloud,' a special prior developed for BATDOG designed for use with the Magellanic clouds, but also appropriate for irregular globular clusters. By convention in the code, the initialisation of this class is called `parallax_prior_repo`.

#### A note on the finite cloud prior type

The *finite cloud* prior uses existing data on the 3D positions of stars to construct a distribution from which stars are presumed to be drawn from. This is done by projecting a 'Gaussian beam' of a certain width (corresponding to the standard deviation of the Gaussian) to weight stars. 

An optimum bin width is estimated with Knuth's rule - which shouldn't be used on weighted histograms in theory - but seemed to give the optimum answer anyway/very close to it in all cases, so is tentatively used. Prior stars are then binned into weighted distance histograms. Error on sky position is assumed to be negligible.

Priors are then smoothed using a Savitzky-Golay filter, which by default uses a cubic scheme. A spline fit is then performed on the filtered data to produce callable fit functions (which again by default are cubic polynomials.) The boundaries of the prior are locked at zero with a number of added points. This improves the performance of the filter, as without multiple boundary points, it doesn't realise that the edge should remain at zero.

An attempt was made to factor in errors on star position into the finite cloud priors. However, they were typically so large that they would cause all detail in the prior to be lost, and the prior would resort to looking like a Gaussian. For enough stars in the model, error on any one star shouldn't matter.

**It is strongly recommended before using the finite cloud prior to test its performance on a siumulated data set similar to your real one. No analytical way to derive optimum beam widths, filter or fit settings is implemented, so you need to do it [with your eyes](https://i.pinimg.com/236x/51/0c/ba/510cba80e4554ec4836f3e4b768fce66--vulnerability-lp.jpg).**

### Other priors

The settings for other priors can be changed in the prior and likelihoods function definition area. You'll probably want to change parameters and constants here.

Some basic checking for errors has been added (for instance, it won't try to evaluate priors on binary/calibration stars if you haven't actually specified any.)

A number of parameters have uniform priors in reasonable ranges. The scatter parameter has a Jeffreys log-uniform prior of $p(x) = 1/x$.

### Debugging the posterior probability

All priors and likelihoods have a `debug` mode specified by a boolean that will highlight any array indices with non-finite values (such as `-inf`, which corresponds to 0 probability in log-probability space.) This can be very helpful in working out what's wrong.

When given even a single infinitely unlikely starting guess, emcee will get '[very stuck](http://weknowmemes.com/wp-content/uploads/2012/05/i-got-stuck-so-i-went-to-sleep.jpg)' and not move walkers at all, even when it looks like it should for 99.9% of the walkers. Debug mode is your friend in this case!


## Creation of a starting guess

A key part of the code is the starting guess maker. A 200-star, 5 band simulation will have ~422 parameters (!!!), and as such, the Monte-Carlo method is extremely slow at moving through parameter space. To complete your work before your own death, it's hence necessary to start at the right answer, and only conduct sampling to get an idea of the errors and any bi-/tri-/etc-modality in the distributions.

The starting guess maker can exclude failed guesses to allow the script to continue. However, it is **strongly** recommended that you investigate and rectify them before proceeding, as they may indicate code/data issues. Excluding stars could result in selection effects. *[Improve before you exclude!](https://i.redd.it/fu8qy0gy3r701.jpg)*

### On code conventions in the starting guess maker

Data is transferred from its respective data frames to a single data dictionary, `data_emcee`. This is due to speed and numpy compatibility issues encountered with `pandas`.

`emcee` can only take parameters in a single, 1D position vector. As such, a dictionary calles `ranges` (sometimes abbreviated `r` in functions for readability) stores the array values of each parameter type. `create_new_ranges_dict()` handles the creation of this dictionary. If `ranges` is wrong, the simulation may keep running quietly, yet the posterior functions will be unable to address/read parameters while the inference runs!

### Starting guess likelihood maximisation

#### Step 1: user-specified parameters

The user should specify Leavitt law parameters, the scale of the prior, Rv, and a guess on the zero-point offset if no binary/calibration stars are in the model.

#### Step 2: inference of the systematic parallax offset

`omega_0` can be inferred initially from the mean difference between calibration star external and Gaia measurements. The external parallax measurements are assumed to be accurate, and are hence added to the starting guess. This step will fail if calibration stars aren't very good/few in number.

#### Step 3: variable star parallax maximisation

Firstly, the true parallax is estimated with three methods: with the given Gaia parallaxes, from evaluations of the Leavitt law for the band with the least scatter, and from the mode of the prior. The likelihoods of each parallax are computed, and the best estimator is picked. In rare cases, all estimates for a star will be infinitely unlikely, at which point you should probably check my code, write a brute-force parallax guesser, or question the size of uncertainties on your input parallaxes.

Then, using the Nelder-Mead scheme (which was found to be the most stable), the parallax guess' probability is maximised. Fails may occur here if the initial guesses are extremely close to infinitely unlikely values.

#### Step 4: extinction inference

Using all parameter guesses made so far, the Leavitt law is evaluated to infer an extinction value to each star. The total `posterior()` value is then evaluated for this star. This step will fail if any of the above are incorrect/bad guesses, or if the scatter on the Leavitt law is too small.

In particular, bad data may produce bad extinction guesses. It's quite common for them to be negative in this case (and hence infinitely unlikely.) In this case, you can proceed in two ways:
1. Try guessing 0.0 as the extinction
2. Drop these stars (and introduce a selection effect in the data)

If large negative numbers are being guessed for extinctions, then this means either that the input Leavitt Law parameters or the parallax are implying the star is much further away than it actually is. Keeping a lot of these stars in the model may cause `emcee` to be unstable (see the Tips for Successful Running section.)

#### Step 5: extension to guesses for all the walkers

Lastly, guesses are made for the walkers by adding Gaussian deviates pulled from distributions with user-set standard deviations. It is often helpful to start with very small standard deviations and allow `emcee` to expand out into parameter space during burn-in of the Markov chains. Using large standard deviations can take longer, or may cause walkers to get stuck in modes of the posterior that are far from the MAP value.

By default, ten times as many walkers are used than dimensions. It helps emcee run efficiently and correctly with a higher acceptance fraction to have quite a few walkers.

Walker guesses are only added if they return a finite evaluation of `posterior()`, which is important so that we don't do something stupid like try to tell BATDOG to assume the Magellanic Clouds exist at a negative distance away from us.

Setting these standard deviations really high will also make it take fucking forever to make guesses for each walker because there will be so many rejects. Setting `debug` to `True` in the call of `posterior()` will let you see which parameters are constantly causing zero probabilities. This step would often take ~20 seconds for me even for thousands of walkers.


## Running with `emcee` and data output

Interfacing with emcee is controlled by a number of functions that simplify the process. 

#### `run_emcee_for_me`

This function controls the calling of `emcee`, helps to prevent out of memory errors, predicts finish times, reports the mean acceptance fraction of all the chains so far, and allows you to specify a cut-off time to end running after (in `end_time` hours.)

Out of memory errors are avoided with empirical numbers corresponding to model parameters (dimenions, walkers, and chain length) that corresponded to roughly one quarter memory usage on a machine with 15.5GB of RAM. Using `psutil.virtual_memory().total` and parameters of the input sampler, the function estimates a maximum chain length that your system can run before encountering memory issues. 

Generally, about one quarter memory usage remains stable and fast, giving Python enough room to copy arrays around. Upto a half could be used, but the program begins to slow to a crawl by this point. This can be changed with the `memory_fraction_to_use` parameter.

If more steps have been requested than can be stored at once, the function auto-refreshes the chain. It will attempt to return the longest possible chain - for instance, if you requested 500 steps but the max stable chain length was 300, then the program would refresh after 200 steps and return 300 steps at the very end.

Every time `sampler.run_mcmc()` is called, `emcee` has to copy to a new array longer than before (due to inefficiencies in .append() methods vs. array pre-allocation.) This can only be done on a single CPU core, and slows the program down for long chains. As such, the program runs silently for `reporting_time` minutes, calling `emcee` for multiple steps at a time and increasing program speed.

#### Parameter and chain plotting functions

Multiple plotting functions are built-in, of which two of the most important for diagnostics are `plot_chains` and `plot_corner`. Both make a plot of global parameters and parameters for a single specified star in a single specified band. Also, `plot_chains`'s lnprob plot at the top is great for looking at how convergence is going.

#### Quantile computation and value comparison functions

Additionally, a function to call `corner.quantile()` in section 6 can be used to calculate the modal value of distributions of all parameters and their +- range to 1 sigma either side. This is then used a bit lower to make a number of interesting plots of parameters between measured and true values.


## Tips for successful running

A number of things are in the documentation above, but here are the highlights:

### Make sure that the mean acceptance fraction is between 0.2 and 0.5

BATDOG reports the mean acceptance fraction when running, and it ideally should be in the range 0.2 to 0.5. If it is significantly lower then this (like, 0.10 or worse, with 0.01 being [terrible](https://media.makeameme.org/created/this-is-going-bj0y3f.jpg)), that can mean two things:
1. There aren't enough walkers.
2. The model is too complicated (with too many parameters) - their degeneracies will cause `emcee` to get very stuck. (There should be a few times more data inputs than parameters to infer.)
3. Underestimated measurement uncertainties on some parameters for some stars (especially parallax) can cause `emcee` to get stuck. Remove these stars from the model (which is a selection effect - but will at least mean that `emcee` is able to run properly and actually produce statistically significant results.) Adding more data to the model (and hence having better-defined likelihoods) will help to deal with some poorer data. Try adding more photometric bands if you can.

More guidance can be found in [this paper](https://ui.adsabs.harvard.edu/#abs/2013PASP..125..306F/abstract) made by the creators of `emcee`.

### You may need to include zero-point calibration stars in your model if you're inferring for a systematic parallax offset

Due to the degeneracy between all parameters - parallax, extinction, etc - BATDOG may get cheeky and use the zero-point offset parameter to shift things around because it doesn't know any better. Having at least ~5 calibration stars helps to solve this. This is likely to be more of an issue for runs with a small number of stars, with few photometric bands, or when the distance prior is strong and pushes the zero-point offset away from its true value.

For further guidance on why inferring for parallax offsets is necessary with Gaia data in small sections of the sky (e.g. the Magellanic clouds), see Bailer-Jones 2018 (IV) or Gaia Collaboration 2018 (Using Gaia parallaxes).


## Author details

BATDOG was originally written by Emily Hunt (eh594@bath.ac.uk, or emily.hunt.physics@gmail.com after she graduates, or @emilydoesastro on Twitter) under the supervision of Dr Victoria Scowcroft.


## Versioning

... was (maybe foolishly) not used during development - please see GitHub commit notes as they're mostly quite informative.


## License

MIT License Emily Hunt (2018) - see LICENSE file


## Acknowledgements

This code was heavily based on work by Branimir Sesar (see 2017 paper), made use of the excellent resources provided on the use of Gaia parallaxes by Coryn Bailer-Jones et al., used a helpful 2010 data analysis guide by David Hogg, and was helped in its development by a Bayesian statistics tutorial by [Jake Vanderplas](https://github.com/jakevdp/BayesianAstronomy).




