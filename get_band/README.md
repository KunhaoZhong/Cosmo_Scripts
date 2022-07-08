# Get Band

After getting chains of cosmological parameters, it is useful to see if the physical observables(Cl, Pk, distances, etc..) are changed or not. Get the observables from the best-fit cosmological parameters is not always good enough, rather, we want a band of observables from the chains.

For example, the CMB anisotropies at large scale seem to be lower than expected from the current data, the so-called low-l anomaly. If we propose a new model and get a 2-sigma preference over LCDM, we would like to see if the posterior of our chains generate a lower Cl band(not just the best-fit model), or if the preference is actually driven by other cosmological parameters.

For example, we get the following Cl_band from the posterior of a model with dark energy anisotropic stress.

![](https://github.com/KunhaoZhong/Cosmo_Scripts/blob/main/get_band/cl_astress.jpg)<!-- .element height="9%" width="10%" -->

The script utilize MPI python wrapper mpi4py, so the execution is fast. A sample slurm script is also provided.

The sctipt can be easily generated to calculate other variables such as matter power spectrum Pk.
