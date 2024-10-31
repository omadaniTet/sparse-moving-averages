


Sparse Moving Averages (SMAs), or efficient space-bounded categorical
learners/predictors, for online open-ended probabilistic prediction,
following the paper 'Tracking Changing Probabilities via Dynamic
Learners'.

See SMAs.py for a number of SMAs such as Sparse EMA and Qs, and see
synthetic_experiments.py for evaluating and comparing these techniques
on a few different types of synthesized sequences.  The begining of
synthetic_experiments.py gives examples of runs and describes
functionality. See wrapper_synthetic.py, for how to generate some of
the tables in the paper, and the various tests (test_SMAs.py and
test_evaluations.py) for insights on the behaviour of the various
functions.
