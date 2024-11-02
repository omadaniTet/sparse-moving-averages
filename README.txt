

This repository contains code for Sparse Moving Averages (SMAs), or efficient 
space-bounded categorical predictors, for online open-ended probabilistic prediction
under non-stationarity, following the paper 'Tracking Changing Probabilities via Dynamic
Learners'.

See SMAs.py for a number of SMAs such as Sparse EMA and Qs, and see
synthetic_experiments.py for evaluating and comparing these techniques
on a few types of synthesized sequences.  The begining of
synthetic_experiments.py gives examples of runs and describes
functionality. See wrapper_synthetic.py, for how to generate some of
the tables entries in the paper, and the two test files (test_SMAs.py and
test_evaluations.py) for insights on the behaviour of the various
functions.

TODO: Add a few real-world sequences from the paper, as well as code
for experiments on those sequences.


