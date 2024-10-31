import numpy as np
import argparse

#
# An example wrapper around synthetic_experiments.py, performing
# experiments on binary sequences (synthetic single-item
# non-stationary, eg oscillating or uniform, and also stationary
# experiments, by setting a large minobs).
#
# Example runs:
#
# python3 wrapper_synthetic.py
#
# python3 wrapper_synthetic.py --seqlen 500
#
from synthetic_experiments import explore_loglossNS_etc, make_args

#  main 
if __name__ == "__main__":

    args = make_args()

    # These could be set in the command line, but for shorter command
    # lines, can be set here.
    args.seqtype = 'binary'
    args.probs = 'uniform'
    args.dev = 1.5 # 2 # can change  to 2, etc. (on command line, or here)

    print('\n#  deviation thresh: %.1f' % args.dev)
    print('#  num trials: %d' % args.ntrials)
    print('#  seqlen: %d' % args.seqlen)
    print('#  seqtype: %s' % args.seqtype)
    print('#  probs: %s' % args.probs)
    
    args.dv = True # report deviation thresholds.

    for sma in ['ema', 'qs', 'dyal']:
        args.sma = sma
        print('\n\n# ------------------ \n# SMA:', sma, '\n')
        for minobs in [10, 50]: #, 100]:
            args.minobs = minobs
            args.minticks = 0 # for uniform
            if args.probs != 'uniform': # Turn off (when uniform)
                if minobs == 10:
                    args.minticks =  400
                elif minobs == 50:
                    args.minticks = 2000
                elif minobs == 100:
                    args.minticks = 4000
            print('\n ------- minticks:%d' % args.minticks)
            args.minlr = 0.01 # for dyal
            args.qcap = 5
            res = explore_loglossNS_etc(args, report = 0)
            res2 = ['%.2f' % x for x in res]
            if sma == 'qs':
                print('# qcap:', args.qcap)
            else:
                print('# minlr:', args.minlr)
            
            print('# minobs:%d %s ..' % ( args.minobs, res2[:5]))
            print('\n# mean:%.3f  std:%.3f\n' % (np.mean(res), np.std(res)) )
    
            print('\n ----- \n')
            
            args.qcap = 10
            args.minlr = 0.001 # for dyal
            res = explore_loglossNS_etc(args, report = 0)
            res2 = ['%.2f' % x for x in res]
            if sma == 'qs':
                print('# qcap:', args.qcap)
            else:
                print('# minlr:', args.minlr)
            
            print('# minobs:%d %s ..' % (
                args.minobs, res2[:5]))

            print('\n# mean:%.3f  std:%.3f\n' % (np.mean(res), np.std(res)) )

