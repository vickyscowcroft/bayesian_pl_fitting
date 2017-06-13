# bayesian_pl_fitting

Adaptation of the Bayesian fitting code from Sesar et al. (2017)

Original Sesar et al. paper: https://ui.adsabs.harvard.edu/#abs/2017ApJ...838..107S/abstract

Original code: fit_PLRs.py

VS modularised version **in progress**: modularised_sesar_fitting_code.ipynb

IN PROGRESS: 

  * Convert to python3
    - imports all work in python3 in main code - check submodules
    - print statements in later functions?
  * Convert dust map read to IRSA astroquery calls


TO DO:

  * Switch from Dambis/Klein & Bloom data to Monson, Beaton & Scowcroft data
  * Change initial PLs
    - Marconi? Want to use consistent set of PL/PLZs
    - Could also consider Neeley update for Spitzer
  * Different priors on stellar distribution - see Astraatmadja & Bailer-Jones papers

DONE:
  
  * Convert data reads to pandas
  * Move to jupyter notebook
  

