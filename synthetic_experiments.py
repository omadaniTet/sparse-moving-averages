#
# For exploring sparse (multiclass) moving avg techniques (SMAs).
#
# Generate a few synthesized sequences and evaluate one or two SMAs on
# it.
#
# There are options to change how the sequences are generated, eg how
# long they are and how long before the underlying (semi)distribution
# (SD) is changed, as well as options for evaluation (choice of loss),
# and changing an SMA technique's parameters.
#
#

# Example runs:
#
# # On 5 sequences of length 1000 each, use Sparse EMA (single method)
# python3 synthetic_experiments.py -nseq 5 -sl 1000 -sma ema
#
#  ( where --sma could be dyal, ema, qs, or box )
#
# The default is reporting on a single method, where a method's
# parameters can be specified, and a number of loss types are
# reported.
#
# (pairing mode: compate 2 methods) Compare Qs with capacity 5 to Qs
# with capacity 10.  Reports the number of wins (lower losses) by each
# method.  python3 synthetic_experiments.py -pair qs,qs -params 5,10
#
# (pairing) Compare DYAL (min rate 0.01) and Box precictors (window
# # size 50).
# python3 synthetic_experiments.py  -pair dyal,box -params 0.01,50
#
# use -br to compare on Brier and -dv to compare using deviation-rate.
#
# use --fc_minpr 0.001 to change Pns and Pmin in filter and cap
# function to 0.001 when evaluating (default is 0.01).

# Compare on sequences where minimum probability is 0.001 and maximum is 0.005
# (and the parameters of filter-cap function are also set to 0.001)
# (use -dv for deviation, or -br for Brier/quadratic)
# 
# python3 synthetic_experiments.py  -sl 40000  -mo 20 -minticks 0  -seed 2 -nseq 5 \
#  -maxpr .005 -minpr .001 --fc_minpr 0.001   --pair dyal,dyal  --params 0.01,0.001 -dv
#

#
# NOTE: in the pairing mode, currently only one main parameter can be
# specified (e.g. the minimum learning rate for DYAL or EMA, and queue
# capacity for Qs and Box).
#
#


import random, numpy as np, sys, math, time
from collections import Counter, defaultdict

import argparse


from SMAs import *
from sma_utils import *

# For evaluating (loss on) an SMA's predictions.
from sma_eval_utils import filter_and_cap, brier_loss, NsReferee, \
    compute_log_loss_NS, ideal_llNS_values, ideal_brier_losses, \
    EVAL_Constants

### globals

positive_symbol = '1' # for 'binary' sequence generation
negative_symbol = '0' # for 'binary' sequence generation
noise_id = 1 # available id for an NS (noise) item 

# Gradually move these

# Could be moved into args.
ignore_NS = False # Ignore/skip  NS items in computing logloss, etc??

# when generating a sequence (noise in it), noise items should appear
# exactly once (otherwise, they may get higher prob, but only just
# below the min prob)
seen_once = 1

# some minimum amount allocated for noise items (when creating a distro)
#noise_allotment = 0.01 # could be 0, and must be less than 1
#noise_allotment = 0.1  # could be 0, and must be less than 1

# The ceiling (max) prob (1.0 or 0.1, etc), when creating a distro to draw from..
ceiling_prob =   1.0 # 0.1 ( max_prob )

# When generating a new prob distro, should all items be new??
all_new = 1 # vs recycle or reuse items when creating underlying SDs (if 1, do NOT recycle)

###

def make_args(msg='Synthetic experiments for SMAs'):
    parser = argparse.ArgumentParser(msg)

    # general options.
    parser.add_argument('--seed', '-seed',  default=1, type=int,
                        help='seed for the generation of sequences.')

    parser.add_argument('--pit', '-pit', action='store_true', default=False,
                        help='pit=print it! or print extra information.')

    # By default, report on one method, unless -pair is specified. Reports the
    # number of wins/losses according to a criterion.
    parser.add_argument('--pair', '-pair', default='', help='If not empty, ' +
                        'a pair, csv, of techniques to compare (eg "qs,dyal").')
    
    # Sequence type or how a sequence is generated (SDs type).
    parser.add_argument('--seqtype', '-sds', default='phases',
                        help='Type of SDs (distributions) used' +
                        ' for sequence generation (phases|binary).')

    # For when the sequence type is binary.
    parser.add_argument('--probs', '-probs', default='0.25,0.025',
                        help='probabilities for binary sequence generation' +
                        ' (use "uniform" for uniform interval).')

    # For distribution (SD) generation (the true underlying SD). A
    # sequence is drawn from such SDs.
    #
    # Determines the stability period for an item, eg 10 ticks, or 50, etc. 
    parser.add_argument('--minobs', '-mo', default=10, type=int,
                        help='Number of observations of an item before its PR can change.')
    parser.add_argument('--minticks', '-minticks', default=0, type=int,
                        help='Number of time points before a SD can change.')
    parser.add_argument('--noise_alloc', '-no', default=0.01, type=float,
                        help='Probability mass allocated for noise (NS) items.')
    # Ceiling or maximum for probability of a salient item, when generating SDs.
    parser.add_argument('--pr_ceiling', '-maxpr', default=1.0, type=float)
    # minimum probability of a salient item, when generating SDs.
    parser.add_argument('--pr_floor', '-minpr', default=0.01, type=float)
    # 'Recycle' or use brand new items when changing the SD?
    parser.add_argument('--all_new', '-all_new', action='store_true', default=False)

    # Evaluation options.
    #
    # By default, use/return loglossNS results (eg when comparing a pair).
    parser.add_argument('--brier', '-br', action='store_true', default=False)
    parser.add_argument('--dv', '-dv', action='store_true', default=False)
    
    parser.add_argument('--dev', '-dev', type=float, default=1.5,
                        help='The deviation threshold for reporting deviation rates.')

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
    # Num trials, or the number of sequences to generate and evaluate on.
    parser.add_argument('--ntrials', '-nseq', default=5, type=int)
    # Sequence length (length of each sequence).
    parser.add_argument('--seqlen', '-sl', default=1000, type=int)
    
    parser.add_argument('--params', '-params', default=None,
                        help='options, csv, for each technique in pairs, eg "10,0.01".')

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


# When we are handling for noise/NS, how low or high can the optimal
# loss (minimum achievable) get, the bounded version (logloss_NS), as
# we change number of salients (each distributed identically having
# equal propbability p)?  This is used to generate the plot (for
# understanding the behavior of bounded logloss).
def explore_opt_bounded_logloss(
        incr=0.02, pmin=0.01, num_salient=1, pnoise=0.01):
    p = 0 # p is the probability of each item, and is increased, by
          # incr, to generate the numbers.
    maxp = 1.0 / num_salient
    ps , lls = [], [] # the probabililities and the losses arrays.
    while p <= maxp:
        if p < pmin: # the items remain non-salient?  all mass is
            # allocated to noise (and the referee agrees).
            loss = 0
        else:
            left = 1.0 - p * num_salient
            if left < pnoise:
                break
            # Some mass, 'left', is given to noise, and the rest
            # num_salient * p is for the salient items.
            loss = left * -log(  left  ) + num_salient * p * -log(p)
        print( 'p:%.3f\tloss:%.3f' % (p  , loss))
        ps.append(p)
        lls.append(loss)
        p += incr

    # One last point
    p = (1.0 - pnoise) / num_salient
    left = pnoise
    loss = left * -log(  left  ) + num_salient * p * -log(p)
    ps.append(p)
    lls.append(loss)
    return ps, lls

####
        
def get_predictor_using_args(args, method=None, method_num=None):
    sma, params = None, None
    if method is None:
        method = args.sma.lower()
    elif method_num is not None: # paired experiments.
        if args.params is not None:
            params = args.params.split(',')[method_num-1]
        
    if method == 'ema' or method == 'sema':
        sma = get_EMA_sma(args, params)
    elif method == 'qs':
        sma = get_Qs_sma(args, params)
    elif method == 'dyal':
        sma = get_DYAL_sma(args, params)
    elif method == 'box':
        sma = get_Box_sma(args, params)
    return sma

def get_EMA_sma(args, params=None):
    min_rate = args.minlr
    if params is not None:
        min_rate = float(params)
    return EMA(use_harmonic=args.harmonic,
               min_rate=min_rate, max_rate=args.maxlr)

def get_Qs_sma(args, params=None):
    cap = args.cap
    if params is not None:
        cap = int(params)
    return Qs(q_capacity=cap)

def get_Box_sma(args, params=None):
    cap = args.cap
    if params is not None:
        cap = int(params)
    return Box(capacity=cap)

def get_DYAL_sma(args, params=None):
    min_rate = args.minlr
    if params is not None:
        min_rate = float(params)
    return DYAL(min_lr=min_rate)

def get_predictor(method, q_capacity=3):
    if method == 1:
        # static EMA (fixed-rate)
        prob_predictor = EMA(use_harmonic=False)
    elif method == 2:
        # With Harmonic decay
        prob_predictor = EMA(use_harmonic=True)
    elif method == 3:
        prob_predictor = Qs(q_capacity=3)
    elif method == 4:
        prob_predictor = DYAL(use_binomial_tail=1)
    return prob_predictor

###

# from 1 to noise range, picked uniformly at random.
# (old) def get_a_noise_item(noise_range):
#
def get_a_noise_item():
    # i = str(random.randint(1, noise_range))
    global noise_id
    noise_id += 1
    return 'noise_' + str(noise_id)

####

# Creates a semidistro (SD), a map of item -> probability
# (PR). (generate_distro or generate a distribution or make_distro,
# generate_synthetic_.. ). Items are string ids ('1', '2', etc.)
def generate_semidistro_probs(args, prev_distro=None):
    # For salient items, don't generate PRs below
    # SMA_Constants.min_prob. Always leave at least 'noise_allotment'
    # for noise items.
    sump, left, min_prob = 0.0, 1.0, args.pr_floor
        # max(SMA_Constants.min_prob, args.pr_floor)
    noise_allotment = args.noise_alloc
    ceiling_prob = min(1.0, args.pr_ceiling) # max PR in an SD
    probs = []
    while left > noise_allotment + min_prob:
        max_prob = min(left - noise_allotment, ceiling_prob)
        assert min_prob <= max_prob
        p = random.uniform(min_prob, max_prob)
        #print(p, min_prob, left - noise_allotment)
        sump += p
        left = 1.0 - sump
        probs.append(p)
    # Because the first prob might be highest.
    random.shuffle( probs )
    i, distro, sump = 0, {}, 0.0
    if args.all_new and prev_distro is not None:
        # start i after the last index in prev_distro
        i = 1 + max([0] + [int(x) for x in list(prev_distro.keys())])
    for p in probs:
        i += 1
        distro[ str(i) ] = p
        sump += p
    assert 1 - sump >= noise_allotment
    return distro

#

# Draw an item randomly according to the given SD (and from its
# support).  Return the selected item and its PR. It may return None
# if no item is picked. (if None, a noise item is to be drawn).
def draw_item(sd, counts):
    p = random.uniform(0, 1.0)
    picked, prob = None, None
    sump = 0.0
    for c2, p2 in sd.items():
        sump += p2
        if p < sump:
            picked, prob = c2, p2
            counts[c2] += 1
            break
    return picked, prob

# generate a random sequence by drawing iid from the given distro
# (which is an SD).
def generate_random_seq(
        args, sd, min_obs, seqlen, min_ticks=None):
    # total ideal loss ..
    #min_ticks = 1000
    #print('# min ticks is:', min_ticks)
    seq, min_count, tot = [], 0, 0.0
    counts = Counter() # informational
    min_prob, sump = 1.1, 0.0 # in the sd
    for c, p in sd.items():
        sump += p
        if p < min_prob:
            min_item = c
            min_prob = p
            
    if args.brier: # compare on quadratic loss?
        item_losses, noise_loss = ideal_brier_losses(sd)
    else: # loglossNS (default)
        item_losses, noise_loss = ideal_llNS_values(sd)
        # -log(1-sump) # loss on an NS (noise) item
    ticks = 0
    #print('# noise prob: %.4f, %d' % (1-sump, min_obs))
    while (min_count < min_obs or (min_ticks is not None and ticks < min_ticks)) \
          and ticks < seqlen:
        ticks += 1
        outcome, prob = draw_item(sd, counts)
        if outcome is None:
            # a noise item is picked.
            seq.append(get_a_noise_item())
            tot += noise_loss
        else:
            seq.append(outcome)
            tot += item_losses.get(outcome)
            # loglossNS
            # tot += -log( prob )                
        min_count = counts[min_item]
    return seq, tot

##

def pick_distro( SDs ):
    random.shuffle(SDs)
    return SDs[0]

###

# generates a sequence, composition of ('stable') subsequences, where
# in each subseq, distro (of multiple items) remains fixed (see
# above).
#
# Also returns the ideal loss.
def generate_seqs( args, do_uniform_ticks=1, SDs=None ):
    desired_len = args.seqlen
    min_obs = args.minobs
    # actual length generated, ideal loss on the sequence, etc.
    loss, slen, seq = 0.0, 0, []
    distro, distro_seq = {}, [] # distro is one SD
    num_SDs, sd_sizes, lefts = 0, [], []
    min_ticks = None
    if do_uniform_ticks and SDs is not None:
        min_ticks = 0
        for distro in SDs:
            min_ticks = max(min_ticks, int( min_obs / min(distro.values())) )
        if pit:
            print('\n# min_ticks was set to', min_ticks)
        
    while slen < desired_len:
        prev_distro = distro
        if SDs is None:
            distro = generate_semidistro_probs(args, prev_distro )
        else:
            distro = pick_distro( SDs )
        #if prev_distro != {} and prev_distro is not None:
        #    abs_diffs = get_distro_diffs(prev_distro, distro)

        sd_sizes.append(len(distro))
        lefts.append(1 - sum(distro.values())) # what's available/left, for noise
        seq2, ideal_loss = generate_random_seq(
            args, distro, min_obs, desired_len - slen, min_ticks=min_ticks )
        slen += len(seq2)
        if args.pit:
            print('# sub seq len:', len(seq2) )
            #print('# sub seq len:', len(seq2), 'distro:', distro, 'min_ticks:', min_ticks)
        seq.extend(seq2)
        # Keeping track of the distro generating the seq
        distro_seq.extend([distro for i in range(len(seq2))])
        loss += ideal_loss # sum of ideal losses
        num_SDs += 1

    if args.pit:
        print('# num SDs: %d, avg_distro_entries:%.1f, avg_for_noise:%.2f, len:%d\n' % (
            num_SDs, np.mean(sd_sizes), np.mean(lefts), len(seq) ))

    return seq, loss / len(seq), distro_seq, num_SDs

####

# Used in generating binary sequences.
def extract_probs(args):
    if  args.probs == 'uniform':
        return None
    probs = args.probs.split(',')
    return [float(x) for x in probs]
    
###
    
# Sample, draw a boolean/binary item (0 or 1), from the given
# old_prob, and from time to time, change the old_prob (take another
# prob from probs or uniformly from prob_interval) ...
def draw_one(probs, old_prob, min_obs, ticks, nposes, t,
             min_ticks=1000):
    prob_interval = None
    if probs is None or probs == []:
        prob_interval = [0.01, 1] # 'uniform' means this range.
    #print('# probs:', probs)
    # if min_ticks > 0, beyond min_obs, we also require at least
    # min_ticks time points before changing the distribution
    # (old_prob).
    if min_ticks is None:
        min_ticks = 0
    p = old_prob
    if p is None or (nposes >= min_obs and ticks >= min_ticks) or \
       (p == 0 and ticks >= min_ticks):
        #if ticks > 0:
        #    print('# num pos:%d current ratio:%.3f' % (nposes, 1.0 * nposes / ticks))
        ticks = 0 # reset 
        if prob_interval is not None:
            p = random.uniform(prob_interval[0], prob_interval[1])
            #print('# sampled p: [%.2f %.2f] p:%.2f' % (prob_interval[0], prob_interval[1], p))
            nposes = 0
        elif len(probs) > 1: # take another prob from probs
            #print("\n# changing target prob.. time & nposes: ", t, nposes)
            nposes = 0
            kept = [] # probs to take a new prob from
            for p in probs:
                if p != old_prob:
                    kept.append(p)
            if len(kept) > 1:
                random.shuffle(kept)
            p = kept[0]
        else:
            p = probs[0]
    # p = probs[prob_index]
    outcome = 0
    ticks += 1
    if p > 0 and random.uniform(0, 1.0) < p:
        outcome = 1
        nposes += 1
    return outcome, ticks, nposes, p # prob_index


# binary, ie two items in the sequence, and one changes PRs among the
# given PRs (probs).. the other item gets the complement PR.
#
# k is the number of observations/draws, or sequence length. The
# sequence oscillates between the given different PRs.. (unless one PR
# is given in probs array)..
def generate_binary_oscillating_seq(args, probs):
    global negative_symbol, positive_symbol
    assert negative_symbol != positive_symbol
    seqlen, min_ticks = args.seqlen, args.minticks
    prob_index = 0
    num_positives = 0 # number of positives
    seq = [] # The sequence of observations.
    used_probs = []
    p, ticks, i, num_pos = None, 0, 0, 0
    #for i in range(k):
    prev, num_sds = None, 0 # number of different SDs
    sd, sds, total = {}, [], 0.0 # total ideal loss, etc.
    while True:
        symbol = negative_symbol
        # num_positives gets reset if/when there is a change in prob.
        outcome, ticks, num_positives, p = draw_one(
            probs, p, args.minobs, ticks, num_positives, i,
            min_ticks=min_ticks)
        if outcome == 1: # positive outcome?
            symbol = positive_symbol
            num_pos += 1
        seq.append(symbol)
        used_probs.append(p) # probs[prob_index])
        if prev != p:
            num_sds += 1
            sd = {} # allocate new SD
            sd[positive_symbol] = p
            sd[negative_symbol] = 1-p
            if args.brier:
                item_losses, noise_loss = ideal_brier_losses(sd)
            else: # loglossNS (default)
                item_losses, noise_loss = ideal_llNS_values(sd)
            
        total += item_losses.get(symbol)
        sds.append(sd)
        prev = p
        i += 1
        # wait till positives are above min_obs, unless the generated
        # sequence is already somewhat bigger than seqlen.
        if i >= seqlen and (num_positives >= args.minobs or i >= 1.1*seqlen):
            break

    assert i > 0
    return seq, total / i, sds, num_sds

###

# Returns the sequence and other information.
def make_sequence(args):
    seq_choice = args.seqtype.lower() # SDs type or sequence type
    other_info = {}
    if seq_choice == 'binary':
        probs = extract_probs(args) # probs = [0.025, 0.15]
        seq, ideal_loss, gt_sd_seq, num_sds = generate_binary_oscillating_seq(
            args, probs)
        #seq, ideal_loss, gt_sd_seq, num_sds = generate_binary_oscillating_seq(
        #    args, [0.025, 0.25, .05], prob_interval=None)
    else: # seq_choice == 'phases': # goes through 'stability phases'
        # Set distros (SDs) to a list of SDs if you
        # want specific distribution switches.. otherwise, they'll be created
        # at random ... 
        # For instance: SDs = [ {'1':.1, '2':.05, '3':.5} ,
        #                       {'1':.1, '2':.4, '3':.05} ]
        SDs = None
        seq, ideal_loss, gt_sd_seq, num_sds = generate_seqs(
            args, SDs=SDs )
        if args.pit: #  and i == 0:
            print('# len of seq in 1st trial: %d' % len(seq))
        sys.stdout.flush()

    other_info['gt_sd_seq'] = gt_sd_seq
    other_info['avg_sd_size'] = np.mean([len(x) for x in gt_sd_seq])
    other_info['num_sds'] = num_sds
    return seq, ideal_loss, other_info

############


def compute_loss(
        args, obs, is_NS, predicted_sd, NS_prob, pit=False):
    if args.brier: # Use Brier scoring (quadratic loss)?
        return brier_loss(predicted_sd, obs)
    else:
        return compute_log_loss_NS(
            obs, is_NS, predicted_sd, NS_prob, pit=pit)

###

def update_max_deviations(predicted_distro, gt_distro, max_devs):
    all_dev, violates = False, set() # all deviate?
    max_dev = max(list(max_devs.keys()))
    for i, p1 in gt_distro.items():
        p2 = predicted_distro.get(i, 0.0)
        if p2 <= 0.0:
            all_dev = True # all deviate!
            break
        else:
            r = max( p1 / p2, p2 / p1 ) # we assume p1 > 0
            if r > max_dev:
                all_dev = True
                break
            else:
                for dev in max_devs:
                    if dev < r:
                        violates.add(dev)
    for dev in max_devs:
        if all_dev or dev in violates:
            max_devs[dev] += 1

####

def increment_deviation_counts(
        args, obs, predicted_SD, gt_SD, deviation_counts,
        max_devs, t=None, count_not_in=0 ):
    global positive_symbol, negative_symbol
    update_max_deviations(predicted_SD, gt_SD, max_devs)
    # for the binary sequences case, look at the probability
    # on the positive symbol
    if args.seqtype.lower() == 'binary':
        p1 = predicted_SD.get(positive_symbol, 0)
        p2 = gt_SD.get(positive_symbol, 0)
        if 0:
            print('# HERE, obs:%s p1 (predicted prob):%.3f gt_p2:%.3f dev_count:%d t:%d' % (
                obs, p1, p2, deviation_counts[args.dev], t) )
            print('#   HERE2: %.3f' % predicted_SD.get(negative_symbol, 0))
            #print('#   HERE2 gt pos prob: %.3f' % gt_SD.get(positive_symbol, 0))
    else:
        p1 = predicted_SD.get(obs, 0)
        p2 = gt_SD.get(obs, 0)
    
    if 0:
        print('timepoint:', t, obs, '%.3f' % p1, 'gt:', p2, deviation_counts,
              predicted_SD)
        
    if p2 == 0: # ground-truth does not have the item (NS item).
        if count_not_in:
            for dev in deviation_counts:
                deviation_counts[dev] += 1
        return # Done.
    r = 1000 # some large number
    if p1 > 0:
        r = max( p1 / p2, p2 / p1 )
    for dev in deviation_counts: # go over each threshold
        if r > dev:
            deviation_counts[dev] += 1

##############

# Given an online prediction/learning (SMA) method, creates a sequence
# (from a choice of distribution) and report/returns the performance
# of that method (loglossNS and others) in a map structure.
#
# Uses bounded logloss (and other performance measures): we need to
# handle NSs (NS=Not Salient, or OOV=out-of-vocab, ie items not seen
# recently) which is tricky in a non-stationary setting and we want to
# allow 'grace periods' ...
#
# 
# (used for table on oscillation or non-stationary, beginning of
#  multi-item experiements.., and remaining multi-item synthetic
#  experiments!)
def explore_loglossNS_etc(
        args, seed=None, pit=0, 
        method=None, predictor_num='', report=1):
    global ignore_NS, noise_id
    num_trials, seqlen = args.ntrials, args.seqlen
    EVAL_Constants.min_pr, EVAL_Constants.NS_allocation  = args.fc_minpr, args.fc_minpr
    SMA_Constants.min_prob = args.fc_minpr

    if seed is not None:
        random.seed( seed )
    elif args.seed is not None:
        random.seed( args.seed )

    #if devs is None: # deviation threshold(s)
    devs = [args.dev]

    # if not empty, this function is being invoked by pair predictors.
    if predictor_num == '' and report:
        seqtype =  args.seqtype
        if seqtype == 'binary':
            seqtype = 'binary (%s)' % args.probs # (also show the PRs)
        print(('\n# seq. type: %s, num_trials (sequences):%d\n# min observations:%d ' +
               '.. desired sequence len: %d, min_prob:%.3f') % (
            seqtype, num_trials, args.minobs, seqlen, SMA_Constants.min_prob))
        
    sys.stdout.flush()
    losses, dev_losses, num_SDs, seqlens = [], [], [], []
    start_time, ideal_losses, avg_sd_sizes = time.time(), [], []
    deviations, max_deviations = [], [] # deviations averages for a few deviation thresholds.
    for i in range(num_trials):
        noise_id = 1 # reset noise ids, before sequence generation.
        seq, ideal_loss, other_info = make_sequence(args)
        seqlens.append(len(seq))
        # ideal_loss is the lowest achievable loss
        ideal_losses.append(ideal_loss)
        # ground-truth SD sequence (one for each time point t)
        gt_sd_seq = other_info.get('gt_sd_seq', None)
        if 'num_sds' in other_info:
            num_SDs.append(other_info['num_sds'])
        if 'avg_sd_size' in other_info:
            avg_sd_sizes.append(other_info['avg_sd_size'])

        #if i == 0:
        #    print('# size:', len(gt_sd_seq), '  sd seq: ', gt_sd_seq[:5], '\n', gt_sd_seq[-1]  )
        
        predictor = get_predictor_using_args(
            args, method=method, method_num=predictor_num)
            
        # This is the NS arbiter/referee, for NS status.
        NS_ref = NsReferee(args.ref_count, args.ref_window)
        if i == 0:
            print('# predictor %s is: %s\n\n' % (
                str(predictor_num), predictor.get_name()))
            sys.stdout.flush()

        #  {1.1:0 , 1.5:0, 2.0:0 }, { 1.1:0 , 1.5:0, 2.0:0 }
        deviation_counts, max_devs = {}, {} 
        for dev in devs:
            deviation_counts[dev] = 0
            max_devs[dev] = 0 

        counted, counted_loss = 0, 0
        loss, t = 0, 0 # time point t, (total) loss of the predictor
        # compute and collect a few performance scores (losses) on the
        # given sequence. Average over sequence length.
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
            increment_deviation_counts(
                args, obs, predicted_SD,
                gt_sd_seq[ t-1 ],
                deviation_counts, max_devs, t=t, count_not_in=0 )
            counted += 1

        # collect the lossses.
        losses.append( loss / counted_loss )
        dev_losses.append(1.0 * deviation_counts[devs[0]] / counted )
        # Collect for other deviation thresholds and for other
        # deviation rates (max rate).
        for dev in devs: # [1.1, 1.5, 2.0]:
            deviation_counts[dev] /= counted
            max_devs[dev] /= counted
        deviations.append( deviation_counts )
        max_deviations.append( max_devs )

    if predictor_num == '' and report: # evaluating a single predictor?
        report_prediction_results(
            args, seqlens, num_SDs, avg_sd_sizes, losses, devs,
            deviations, max_deviations, ideal_losses)
        print('\n# Time taken: %.2f mins\n' % get_mins_from(start_time))
        
    if args.dv: # for paired comparisons
        #print('\n# returning dev loss..')
        return dev_losses
    else:
        return losses
    
##

def report_prediction_results(
        args, seqlens, num_SDs, avg_sd_sizes, losses, devs,
        deviations, max_deviations, ideal_losses):
    print(('\n\n# done.. Reporting averages over %d sequences (minlen:%d, maxlen:%d).. avg.' +
           '\n  num SDs generating a seq: %.1f (min number:%d max:%d) ' +
           'avg. num salients (support size): %.1f') % (
               len(num_SDs), np.min(seqlens), np.max(seqlens),
               np.mean(num_SDs), np.min(num_SDs), np.max(num_SDs), np.mean(avg_sd_sizes)))
    loss_type = '(loglossNS)'
    if args.brier:
        loss_type = '(Brier loss)'
        
    print('# optimal (lowest achievable) loss:%.3f std:%.3f %s' % (
        np.mean(ideal_losses), np.std(ideal_losses), loss_type))
    print('# avg of predictor mean loss: %.3f std:%.3f max:%.3f %s' % (
        np.mean(losses), np.std(losses), np.max(losses), loss_type))

    # Report on deviation performance.
    for dev in devs: # [1.1, 1.5, 2.0]:
        sumds, max_sumds = [], []
        for dev_map in deviations:
            sumds.append( dev_map[dev] )
        for dev_map in max_deviations:
            max_sumds.append( dev_map[dev] )
        # average (over all sequences) of average deviation rate.
        print('# For dev:%.1f, mean deviation on observed:%.3f (std:%.3f)' %  (
            dev, np.mean(sumds), np.std(sumds)))
        print(('# For dev:%.1f, mean deviation-rate on any item (or max): ' +
               ' %.3f (std:%.3f)') % (
                   dev, np.mean(max_sumds), np.std(max_sumds)))
    print()
    
##

# Compare two methods (SMAs) on the same sequences.
# Report number of wins and losses.
def compare_pairs_of_methods( args ):
    start_time = time.time()
    print('\n# Comparing two methods on the same %d sequences.\n' % args.ntrials)
    m1 = args.pair.split(',')[0]
    res1 = explore_loglossNS_etc(
        args, method=m1, predictor_num=1)
        
    m2 = args.pair.split(',')[1]
    res2 = explore_loglossNS_etc(
        args, method=m2, predictor_num=2)
        
    nw1, nw2, _, mll1, mll2 = get_num_wins_etc(res1, res2)

    #print('# m1:', get_method_name_etc(m1[0], m1[1]), '  m2:',
    #      get_method_name_etc( m2[0], m2[1] ))

    loss_type = '(on loglossNS)'
    if args.brier:
        loss_type = '(on Brier loss)'
    if args.dv: # over ride!
        loss_type = '(Dev. rate, on observed item, threshold %.1f)' \
            % args.dev
    
    print('# numwins1: %d numwins2: %d (mean1:%.3f, mean2:%.3f) %s\n' % (
        nw1, nw2, mll1, mll2, loss_type))
    
    print('\n# Time taken: %.2f minutes.' % get_mins_from(start_time))

###

def get_num_wins_etc(res1, res2):
    nwins1, nwins2, ties = 0, 0, 0
    # extract the bounded loglosses
    #for ll1, ll2 in zip(res1['lls'], res2['lls']):
    for ll1, ll2 in zip(res1, res2):
        if ll1 < ll2: # which one is lower?
            nwins1 += 1
        elif ll1 > ll2:
            nwins2 += 1
        else:
            ties += 1
    return nwins1, nwins2, ties, np.mean(res1), np.mean(res2)
            
####


#  main 
if __name__ == "__main__":

    args = make_args()

    if ',' in args.pair: # A pair of predictors have been specified?
        # Compare the two techniques (two SMAs).
        compare_pairs_of_methods(args)
    else: # this should be the default ..
        # report on the performance of a single method (SMA) given choice
        # of SDs (sequence types generated), number of sequences (trials),
        # etc..
        explore_loglossNS_etc(args)

    if 0: # how high/low can lowest achievable bounded loglossNS get?
        explore_opt_bounded_logloss(incr=0.01, pmin=0.01, num_salient=1)
