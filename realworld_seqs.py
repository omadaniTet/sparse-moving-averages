#
# For exploring sparse (multiclass) moving avg techniques (SMAs).
#
# Read a set of sequences and run one SMA on it.  Report loglossNS.
#
# Currently reads from Expedition sequences (104 sequences above length 75).
#
# TODO: add a few other sequences.
#

# Example runs:
#

# ( report only sequences above 5000 length )
# python3 realworld_seqs.py --sma dyal  --minlr 0.001 --minlen 5000
#
#  (where sma could be dyal, ema, qs, or box )
#
######     or compare a pair of SMAs:
# 
# python3 realworld_seqs.py --pair  dyal,qs  --params 0.01,5  --ref_window 500
#
# (output contains: # numwins1: 92 numwins2: 12 (mean1:2.071, mean2:2.269) (on loglossNS)  )
# ( ie dial won on 92 of the 104 sequences )
#
#
#  ( for ema setting --minlr,  to around  0.01  often works best)
#
#  ( for box set --cap, or capacity, from 100 to  1000  )
# 
#  ( for queues set --cap, somewhere in 3 to 10, etc)
#
#



# only on  sequences with length above 5k
# python3 realworld_seqs.py --minlen 5000  
#

import gzip, argparse
from SMAs import *
from sma_utils import *

# For evaluating (loss on) an SMA's predictions.
from sma_eval_utils import filter_and_cap, NsReferee, \
    compute_log_loss_NS,  EVAL_Constants

from synthetic_experiments import get_predictor_using_args, compute_loss, get_num_wins_etc

#########

# Gradually move these

# Could be moved into args.
ignore_NS = False # Ignore/skip  NS (noise) items in computing logloss, etc??

##

###

def make_args(msg='Synthetic experiments for SMAs'):
    parser = argparse.ArgumentParser(msg)

    # general options.
    parser.add_argument('--pit', '-pit', action='store_true', default=False,
                        help='pit=print it! or print extra information.')

    # By default, report on one method, unless -pair is specified. Reports the
    # number of wins/losses according to a criterion.
    parser.add_argument('--pair', '-pair', default='', help='If not empty, ' +
                        'a pair, csv, of techniques to compare (eg "qs,dyal").')
    
    # Evaluation options.
    #
    # By default, use/return loglossNS results (eg when comparing a pair).
    parser.add_argument('--brier', '-br', action='store_true', default=False)

    # For 'short' experiments, such as sequences of 1000s long, and to
    # keep it simple, no window size (ie None) is fine (and used for
    # most of the paper).
    parser.add_argument('--ref_window', '-rw', type=int, default=None,
                        help='The window used by the referee in loglossNS (if none, no limit).')
    
    # An item is marked NS by the referee iff the observation count,
    # within the referee window, is not greater than this threshold.
    parser.add_argument('--ref_count', '-rc', type=int, default=2,
                        help='The referee observation-count threshold t: marked NS iff <= t.')
    
    parser.add_argument('--fc_minpr', '-rminpr', type=float, default=0.01,
                        help='Min prob threshold in filter & cap function, in (0,1).')
    
    # Experimental size parameters.
    #
    parser.add_argument('--params', '-params', default=None,
                        help='options, csv, for each technique in pairs, eg "10,0.01".')

    parser.add_argument('--minlen', '-minlen', default=75, type=int,
                        help='Minimum sequence length (ignore shorter ones).')
    
    parser.add_argument('--maxlen', '-maxlen', default=0, type=int,
                        help='Maximum sequence length (ignore longer ones).')
    
    # Prediction methods
    parser.add_argument('--sma', '-sma', default='EMA',
        help='SMAs (sparse moving average techniques): EMA (default), Qs, DYAL')
    
    parser.add_argument('--harmonic', '-harmonic', action='store_true', default=False)
    
    parser.add_argument('--minlr', '-minlr', default=0.001, type=float,
                        help='min rate or fixed-rate for EMA variants and DYAL')
    parser.add_argument('--maxlr', '-maxlr', default=1.0,  type=float,
                        help='max rate for harmonic EMA.')
    
    parser.add_argument('--cap', '-cap', default=3, type=int,
                        help='queue capacity for Qs and Box.')
    
    args = parser.parse_args()
    return args

###

# Oct 2023 (read from files dumped above!)
#
# These observation sequences where collected during
# Expedition training. See README in dir_expedition_seqs/
def read_seqs(args, fname=None, min_len=None, max_len=None, pit=1, sort_it=1):
    if args.minlen > 0:
        min_len = args.minlen
    if args.maxlen > 0:
        max_len = args.maxlen
        
    if fname is None:
        fdir = './'
        # fdir += 'dir_500thrsh/' # which subdir
        fname = fdir + 'Expedition_108_sequences.txt.gz'
    seqs = []
    with gzip.open(fname, "rt") as fp:
        for line in fp:
            if line.startswith('#'):
                continue # skip comment lines
            if ',' not in line: # must be empty, etc.
                continue
            parts = line.split(',')
            lenp = len(parts)
            if min_len is not None and lenp < min_len:
                continue
            if max_len is not None and lenp > max_len:
                continue
            seq = []
            for tup in parts:
                if ':' not in  tup:
                    continue
                seq.append(tup.split(':')[1])
            seqs.append(seq)
        if pit:
            print_seq_info(seqs)
        if sort_it: # sort by increasing len
            seqs.sort( key=lambda x: len(x)  )
        return seqs

###

def print_seq_info(seqs, more_info=0):
    print('\n# num seqs read and kept: ', len(seqs))
    lens =  [len(x) for x in seqs]
    if len(lens) < 10:
        print('# seq lens read: ',  lens)
    print('# min and max len seq:', min(lens), max(lens), ', median:%.1f' % np.median(lens) )
    if more_info: # get num unique in each, print in decreasing order, etc
        lens =  [(len(x), len(set(x))) for x in seqs]
        lens.sort(key=lambda x: -x[0]  )
        i = 1
        for pair in lens:
            print(i, pair)
            i += 1
    print('\n# 1st few of 1st seq: ', seqs[0][:2])

####

def report_prediction_results(args, seqlens, losses):
    print('\n\n# done.. Reporting averages over %d sequences (minlen:%d, maxlen:%d)\n' % (
        len(seqlens), np.min(seqlens), np.max(seqlens)))

    loss_type = '(loglossNS)'
    if args.brier:
        loss_type = '(Brier loss)'
        
    print('# loss type %s' %  loss_type  )
    print('# avg of predictor mean loss: %.3f std:%.3f max:%.3f %s' % (
        np.mean(losses), np.std(losses), np.max(losses), loss_type))

    print()

###

def explore_loglossNS_etc(
        args, seqs, pit=0, 
        method=None, predictor_num='', report=1):
    global ignore_NS
    EVAL_Constants.min_pr, EVAL_Constants.NS_allocation  = args.fc_minpr, args.fc_minpr
    SMA_Constants.min_prob = args.fc_minpr

    losses,  seqlens, i = [], [], 0
    start_time = time.time()
    for seq in seqs:
        i += 1
        seqlens.append(len(seq))

        # Get a fresh predictor and referee.
        predictor = get_predictor_using_args(
            args, method=method, method_num=predictor_num)
    
        # This is the NS arbiter/referee, for NS status.
        NS_ref = NsReferee(args.ref_count, args.ref_window)
    
        if i == 1:
            print('# predictor %s is: %s\n\n' % (
                str(predictor_num), predictor.get_name()))
            sys.stdout.flush()

        t, loss, counted_loss = 0, 0.0, 0
        for obs in seq:
            t += 1
            # NOTE: here, we could also use an ideal referee in these
            # synthetic experiments (we are using a practical
            # referee).
            is_NS = NS_ref.get_is_NS_and_update(obs)
            # get item -> PR map ('raw distro')
            raw_predictions = predictor.get_distro()
            predicted_SD, NS_prob = filter_and_cap(raw_predictions)
            prob = predicted_SD.get( obs, 0.0 ) # evaluate this probability.
            predictor.update( obs ) # update the predictor.
            if not ignore_NS or not is_NS:
                loss += compute_loss(args,
                    obs, is_NS, predicted_SD, NS_prob, pit=0) # pit and i==0)
                counted_loss += 1
            #counted += 1
        
        # collect the lossses.
        losses.append( loss / counted_loss )

    if predictor_num == '' and report: # evaluating a single predictor?
        report_prediction_results( args, seqlens, losses )
        print('\n# Time taken: %.2f mins\n' % get_mins_from(start_time))

    return losses

###

def compare_pairs_of_methods( args, seqs ):
    start_time = time.time()
    print('\n# Comparing two methods on the same %d sequences.\n' % len(seqs) )
    m1 = args.pair.split(',')[0]
    res1 = explore_loglossNS_etc(
        args, seqs, method=m1, predictor_num=1)
        
    m2 = args.pair.split(',')[1]
    res2 = explore_loglossNS_etc(
        args, seqs, method=m2, predictor_num=2)
        
    nw1, nw2, _, mll1, mll2 = get_num_wins_etc(res1, res2)

    #print('# m1:', get_method_name_etc(m1[0], m1[1]), '  m2:',
    #      get_method_name_etc( m2[0], m2[1] ))

    loss_type = '(on loglossNS)'
    if args.brier:
        loss_type = '(on Brier loss)'
    
    print('# numwins1: %d numwins2: %d (mean1:%.3f, mean2:%.3f) %s\n' % (
        nw1, nw2, mll1, mll2, loss_type))
    
    print('\n# Time taken: %.2f minutes.\n' % get_mins_from(start_time))

        
####


#  main 
if __name__ == "__main__":

    args = make_args()

    seqs = read_seqs(args)
    
    if ',' in args.pair: # A pair of predictors have been specified?
        # Compare the two techniques (two SMAs).
        compare_pairs_of_methods(args, seqs)
    else: # this should be the default ..
        # report on the performance of a single method (SMA)
        explore_loglossNS_etc(args, seqs)
        

