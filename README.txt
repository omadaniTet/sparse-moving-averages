

This repository contains code for Sparse Moving Averages (SMAs), or efficient 
space-bounded categorical predictors, for online open-ended probabilistic prediction
under non-stationarity, following the paper 'Tracking Changing Probabilities via Dynamic
Learners' ( https://arxiv.org/abs/2402.10142v2 ). 'Open-ended' means the set of 
items that would be seen in the input stream is not known to the predictor, and can grow
unbounded with passage of time. The predictor, being space-bounded (finite space, independent 
of streamm length), can only predict a relatively small subset of items, each with an estimated 
probability (a 'moving average'), at any given time.

- See SMAs.py for a number of SMAs such as Sparse EMA and Qs

- Use synthetic_experiments.py for evaluating and comparing these techniques
on a few types of synthesized sequences.  The begining of
synthetic_experiments.py gives examples of runs and describes
functionality. 

- See dyal_on_binary.py for an example of a function that generates
arrays of output (learned) probabilities from running DYAL on a binary 
oscillating sequence (gives an example of plotting the data).

- See wrapper_synthetic.py, for how to generate some of
the tables' entries in the paper. 


- See the two unit test files (test_SMAs.py and
test_evaluations.py) for insights on the behaviour of the various
functions.

TODO: Add a few real-world sequences from the paper, as well as code
for experiments on those sequences.


