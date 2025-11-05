

This repository contains code for Sparse Moving Averages (SMAs), or efficient 
space-bounded categorical predictors, for online open-ended probabilistic prediction
under non-stationarity, following the paper 'Tracking Changing Probabilities via Dynamic
Learners' ( https://arxiv.org/abs/2402.10142v2 ). 'Open-ended' means the set of 
items that would be seen in the input stream is not known to the predictor, and can grow
unbounded with passage of time. The predictor, being space-bounded (finite space, independent 
of stream length), can only predict a relatively small subset of items, each with an estimated 
probability (a 'moving average'), at any given time.

- See SMAs.py for a number of SMAs such as Sparse EMA and Qs

- Use synthetic_experiments.py for evaluating and comparing these techniques
on a few types of synthesized sequences.  The beginning of
synthetic_experiments.py gives examples of runs and describes
functionality. 

- See dyal_on_binary.py for an example:  a function that generates
arrays of output (learned) probabilities from running DYAL on a binary 
oscillating sequence (gives an example of plotting the data).

The picture, dyal_001_differentBinomThrshs_syn_oscillates.png, plots DYAL's estimates when 
the binomial tail parameter is changed, one of {1, 3, 5, 10}, 
min learning-rate of 0.001, on the same binary oscillating (0.025<->0.25) sequence (setting to 1 leads
to high variance (close the Qs), and 10 can be too conservative (close to static EMA, 
fixed rate of 0.001).

- See wrapper_synthetic.py, for how to generate some of
  the tables' entries in the paper. 

- realworld_seqs.py reads from Expedition sequences (108) and reports on that (either 
  one SMA or compare a pair like above).  

For Masquerade Unix sequences, please get the sequences from: 
https://www.schonlau.net/masquerade/MasqueradeDat.gz
(and we used the first uncorrupted 5k from each of the 50 sequences)

- See the two unit test files (test_SMAs.py and test_evaluations.py) for additional 
  insights on the behavior of the various functions.

