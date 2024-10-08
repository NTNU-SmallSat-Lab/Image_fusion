# HYPSO CNMF data fusion
**Unfinished code still in development, messy, buggy and poorly documented**
The code currently only runs on simulated data, it requires Hypso L1B data as input.

In order to use, determine variables in Hypso_CNMF.py
Decide number of endmembers for output
for most scenes the variables can be kept as:

delta = 0.15
tol = 0.00005
PPA = False #PPA implementation is problematic and unfinished, do not trust it
Unhaze = True
loops = (300, 5)

Write in the number of endmembers and desired coordinates.

The program discards hypso bands 0-3 and 115-119