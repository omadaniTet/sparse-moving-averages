import random, math
import dir_code_backups.dir_for_tmlr2025_paper.SMAs as SMAs
# import drift
from river import drift
import numpy as np
# to run on command line
import argparse


# To compare ADWIN with Qs and DYAL, on binary sequences (capped
# logged loss and Brier).


# See compare(...) below.
#
#
# Example run:
#
# python3 compare_to_ADWIN.py -seed 1 -n 5000 -k 5 -delta 0.1
#
# or once in a Python3 environment (notebook, etc), and imported compare_to_ADWIN, run:
# compare()
#
# # This one does 30 sequences, 10000 long each.
# compare(k=30, n=10000, seed=1)
#

#########

# To install river:

# created dir_venv
# python3 -m venv dir_venv
# source dir_venv/bin/activate
# python3 -m pip install river  # now this worked! (in virtual env)

###

# Comparing ADWIN to Qs and DYAL.
#
# Compare on k sequences of n events (observations) long.
# 
# Both capped logloss and Brier (quadratic loss) are reported.
#
def compare(
        k=10, minobs=10, n = 1000, p1=None, p2=None, pit = 0,
        # maximum loss for logloss (so 0 probs don't affect it ).
        max_ll = 5, ignore = 0,
        # Parameters for ADWIN
        delta=.01, clock=10, grace_period=5, max_buckets=5, mwl=5,
        # seed for seq. generation (if None or 0, diffent output each time)
        seed=None):

    # wins based on logloss and Brier.
    br_wins_qs, br_wins_dy, wins_dyal_ll, wins_qs_ll = 0, 0, 0, 0
    
    qs_perf, adw_perf, dyal_perf = [], [], []
    qs_lls, adw_lls, dyal_lls = [], [], []
    if seed:
        random.seed(seed)
        
    print('\n# Averaging over %d sequences, %d long each, minobs:%d\n' % (k, n, minobs) )
    
    for i in range(k):
        #if p1 is None:
        seq = gen_seq( minobs, n, p1, p2)
        vals_qs = apply_qs(seq)
        qs_brier = simpler_brier(vals_qs)

        vals_dyal = apply_dyal(seq)
        dyal_brier = simpler_brier(vals_dyal)
        
        vals_adw = adwin_estimates(
            seq,
            delta=delta, max_buckets=max_buckets,
            mwl = mwl, 
            clock=clock, grace_period=grace_period )
        
        if pit and len(vals_adw ) < 200:
            print('\n# adw_vals: ', zip_str(seq, vals_adw))
            
        if pit and len(vals_dyal ) < 200:
            print('\n# dyal_vals: ', zip_str(seq, vals_dyal))
            
        if pit and len(vals_dyal ) < 200:
            print('\n# qs_vals: ', indxed_str(vals_qs))
            
        adw_brier = simpler_brier(vals_adw)
        if qs_brier < adw_brier:
            br_wins_qs += 1
        if dyal_brier < adw_brier:
            br_wins_dy += 1
            
        if pit:
            print('# %d. adw_brier:%.2f (on %d items)  qs_brier:%.2f dyal:%.2f (%d/%d wins)' % (
                i, adw_brier, len(vals_adw), qs_brier, dyal_brier, br_wins_dy, i+1))

        dyal_ll = simpler_ll(vals_dyal, max_loss=max_ll, ignore=ignore)
        qs_ll = simpler_ll(vals_qs, max_loss=max_ll, ignore=ignore)
        adw_ll = simpler_ll(vals_adw, max_loss=max_ll, ignore=ignore)
        qs_lls.append(qs_ll)
        adw_lls.append(adw_ll)
        dyal_lls.append(dyal_ll)

        if dyal_ll < adw_ll:
            wins_dyal_ll += 1
            
        if qs_ll < adw_ll:
            wins_qs_ll += 1
            
        if pit:
            print('#   %d. adw_ll:%.2f (on %d items)  qs_ll:%.2f dyal:%.2f (%d/%d ll_wins)' % (
                i, adw_ll, len(vals_adw), qs_ll, dyal_ll, wins_dyal_ll, i+1))
            
        qs_perf.append(qs_brier)
        adw_perf.append(adw_brier)
        dyal_perf.append(dyal_brier)
        


    print(('\n# Done! (on %d sequences, each %d long)\n\n----------\n\n# Results:\n\n' + 
           '# logloss: ADWIN mean:%.3f, Qs mean:%.3f, DYAL: %.3f') % (
               k, n,
               np.mean(adw_lls), np.mean(qs_lls), np.mean(dyal_lls)  ) )
    
    print('\n# Brier: ADWIN mean:%.3f , Qs  mean:%.3f, DYAL: %.3f' % (
        np.mean(adw_perf), np.mean(qs_perf), np.mean(dyal_perf)  ) )

    # number of win compared to ADWIN
    print('\n# Qs logloss wins (compared to ADWIN, on each sequence): %d/%d' %  (wins_qs_ll, i+1))
    print('\n# DYAL logloss wins: %d/%d' %  (wins_dyal_ll, i+1))

          
    print('\n# trials:%d ...   Brier wins of Qs over ADWIN:%d, of dyal over ADWIN:%d\n' % (
        k, br_wins_qs, br_wins_dy) )

########

# Generate binary sequences, either using p1<->p2, ie oscillation, or
# picking the probability from unifromly at random.
#
def gen_seq( minobs, n, p1, p2, pit=0):
    probs = [p1, p2]
    pr = p1
    if p1 is None:
        pr = random.uniform(0, 1)
    pos, seq = 0, []
    idx, i = 0, 0
    while i < n:
        item = 0
        if random.uniform(0, 1) < pr:
            item = 1
        if item == 1:
            pos += 1
        if pos >= minobs:
            pos = 0
            if p1 is not None:
                idx += 1
                idx %= 2
                pr = probs[idx]
            else:
                pr = random.uniform(0, 1)

            if pit:
                print('# changed! pr:%.2f' % pr, 'i', i )
            
        seq.append(item)
        i += 1
    return seq

####

# Get the prob of item before observing it. Use Qs.
def apply_qs(seq):
    qs = SMAs.Qs(q_capacity=5)
    ps = []
    for s in seq:
        p1 = qs.get_prob(s)
        p2 = qs.get_prob(1 - s)
        if 0 and j < 20:
            print( ' sum: %.2f' % (p1+p2))
        if p1 > 0:
            p1 = p1 / (p1+p2)
        ps.append( p1 )
        qs.update(s)
        
    return ps

###

# reports RMSE (equivalent to Brier on binary outcomes)
#
def simpler_brier(ps, do_str=0):
    e = 0
    for p in ps:
        e += (1-p) * (1-p)
    # root mean squared error
    if do_str:
        return '%.3f' % math.sqrt( e / len(ps) )
    return  math.sqrt( e / len(ps) )

# simple capped logloss.
def simpler_ll( ps, max_loss=5, ignore=0, do_str=0 ):
    e = 0
    i = 0
    num = 0
    for p in ps:
        i += 1
        if i < ignore:
            continue
        num += 1
        l = max_loss
        if p > 0:
            l =  -math.log(p)
        e += min(l, max_loss)
        
        
    e = e / num
    if do_str:
        return '%.3f' % e
    return  e

###

def apply_dyal(seq):
    SMAs.SMA_Constants.ignore_sd = False
    dyal = SMAs.DYAL( min_lr=0.001, q_capacity = 3 )
    # print( dyal.get_name() )
    ps = []
    j = 0
    for s in seq:
        p1 = dyal.get_prob(s)
        p2 = dyal.get_prob(1 - s)
        if 0 and j < 20:
            print( ' sum: %.2f' % (p1+p2))
        if p1 > 0:
            p1 = p1 / (p1+p2)
        ps.append( p1 )
        dyal.update(s)
        j += 1
        
    return ps

###

def adwin_estimates(
        seq, delta=0.05, clock=10, grace_period=5,
        max_buckets=5, mwl = 5, 
        pit = 0 ):
     adwin = drift.ADWIN( delta=delta, clock=clock,
                          max_buckets=max_buckets,
                          min_window_length = mwl, 
                          grace_period=grace_period )
     
     # values output  (sequence of probabilities)
     i , vals = 0, []
     #     for i, val in enumerates1):
     for val in seq:
         i += 1
         # print('# %d, %.3f' % (i, adwin.estimation))
         p_1 = adwin.estimation
         if val == 1:
             vals.append( p_1 )
         else:
             vals.append( 1.0 - p_1 )
             
         adwin.update(val)
         if pit and adwin.drift_detected:
             print(f"Change detected at index {i}, input value: {val}")
     return vals

###

def indxed_str(vals):
    str = ''
    i = 0
    for s in vals:
    	i += 1
    	str += '(%d, %.2f)' % (i, s)
	
    return str

def zip_str(seq, ps):
    str = ''
    i = 0
    for s in zip(seq, ps):
    	i += 1
    	str += '(%d, %.2f)' % (s[0], s[1])
	
    return str

###

def make_args(msg='Comparisons with ADWIN (binary sequences).'):
    parser = argparse.ArgumentParser(msg)

    # general options.
    parser.add_argument('--seed', '-seed',  default=0, type=int,
                        help='seed for the generation of sequences.')

    parser.add_argument('--k', '-k', default=10, type=int,
                        help='Number of sequences to compare on.')
    parser.add_argument('--n', '-n', default=2000, type=int,
                        help='Number of observations or sequence length.')
    parser.add_argument('--minobs', '-mo', default=10, type=int,
                        help='Number of observations of 1 before its probability can change.')
    # for ADWIN
    parser.add_argument('--delta', '-delta', default=0.01, type=float,
                        help='Confidence delta for ADWIN.')

    # Add more options..
    
    args = parser.parse_args()
    return args

############


if __name__ == "__main__":

    args = make_args()

    compare(k=args.k, n=args.n, seed=args.seed,
            minobs=args.minobs,
            delta=args.delta
            )
    
