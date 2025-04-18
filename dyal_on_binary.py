#
# To support generation of data for plots in synthetic experiments.
# For starters, plot of DYAL probability outputs, on a binary
# oscillating sequence, as a function of time (fig ~15 of the paper in
# synthetic experiments). One can change the binomial tail threshold,
# the minimum learning rate, etc.
#

import SMAs
from sma_eval_utils import filter_and_cap
from synthetic_experiments import make_sequence, make_args, get_DYAL_sma

#####


# Get a binary stream, or generate one, and run DYAL on it.  Report
# (return) estimated/computed probability and learning rate at each
# time point, as well as the observation (the sequence), as lists.
#
# Example use in a Python notebook, plotting:
#
#
#class my_args:
#    def __init__(self):
#        self.minlr = 0.001
#        self.seqtype = 'binary'
#        self.probs = '0.025,0.25'
#        self.seqlen = 20000
#        self.minticks = 2500 # None # 1000
#        self.brier = None
#        self.minobs = 10
#
# args = my_args() # This is needed for making a binary sequence.
#
# (change the binomial threshold: low value such as 1 leads to high
#  variance and makes it closer to Qs, while high value such as 10
#  leads to static EMA, or slow convergence when minimum learning-rate
#  is low).
# 
# ts, probs_001_1, rates_001_1, seq1 = dyal_probs_etc_on_binary_seq(args, seq=None, binom=1)
# ts, probs_001_3, rates_001_3, seq1 = dyal_probs_etc_on_binary_seq(args, seq=seq1, binom=3)
#
# import matplotlib.pyplot as plt
# plt.plot(  ts , probs_001_1 , linestyle="",  marker='.', color='yellow', label = 'thrsh 1' )
# plt.plot(  ts , probs_001_3,  linestyle="",  marker='.', color='blue', label = 'thrsh 3' ) 
#
def dyal_probs_etc_on_binary_seq(args, method=None, seq=None, binom=None):
    #
    # NOTE: assumes the sequence is binary.
    if seq is None:
        seq, ideal_loss, other_info = make_sequence(args)
 
    # predictor = get_predictor_using_args(args, method=method)
    dyal_sma = get_DYAL_sma(args)

    # play with (set the) binomial threshold, etc.
    if binom is not None:
        SMAs.SMA_Constants.binomial_thrsh = binom # 1 # 100 # 3, 5, 7

    print('\n', dyal_sma.get_name())
    ts, probs, rates = [], [], []

    t = 0
    for obs in seq:
        t += 1
        # get item -> PR map ('raw distro')
        # raw_predictions = predictor.get_distro()
        raw_predictions = dyal_sma.get_distro()

        # predicted_SD = raw_predictions
        predicted_SD, NS_prob = filter_and_cap(raw_predictions)
        
        prob = predicted_SD.get( '1', 0.0 ) # gets its prediction for '1'.

        # NOTE: this works for a DYAL predictor only
        rate_map = dyal_sma.get_lrs(  )
        rate = rate_map.get('1', 0)
        
        dyal_sma.update( obs ) # update the predictor.
        
        probs.append(prob)
        rates.append(rate)
        ts.append(t)
        
    return ts, probs, rates, seq
 
########

#  main 
if __name__ == "__main__":

    args = make_args()

    # These could be set in the command line, but for shorter command
    # lines, can be set here.
    args.seqtype = 'binary'

    print('#  seqlen: %d' % args.seqlen)
    print('#  seqtype: %s' % args.seqtype)
    print('#  probs: %s' % args.probs)

    times, probs, lrs, seq = dyal_probs_etc_on_binary_seq(args)

