# Functions for evaluating/comparing SMAs.

from math import log
from collections import Counter

# For the referee, the Box predictor could be used.
from SMAs import Box

# >class
class EVAL_Constants:
    # NOTE: min_pr and NS_allocation should be equal in general.
    #

    # For detemining NS items.. (in (0, 1) ).
    min_pr = 0.01
    # This should be in general equal to  NS_min_pr.
    NS_allocation = 0.01

    # should we up-norm (or only down-norm when necessary) during
    # extraction of NS sd (a distribution that leaves some PR mass for
    # NS events)?..  In general, we should not up-norm to promote
    # truth telling (propriety).
    up_norm_NS = False #  by default, it should be False..

    # For detemining NS items..
    # should be an interger >= 0 (never None). For results to make
    # sense, it should be 2 or higher, that is, to estimate
    # probabilities, some grace period is needed. But for
    # experimental purposes we allow 1 or 0.
    NS_freq = 2
    NS_window = None # window size (if None, no limit).

    
# Returns a pair: the 'normed' and 'capped' distro (a map: item -> prob), and
# also the probability assigned to NS items (all put/pooled
# together), ie the probs in map should sum at most to
# 1 - NS_minprob, ie strictly less than 1 (we assume
# NS_minprob in (0, 1), and we make extensive use of
# that here).
def filter_and_cap(c_to_p): # cap and filter We assume each
    # entry in values of c_to_p is a prob (PR), ie in [0, 1], but
    # the sum in c_to_b could be above 1..
    sump = 0.0
    distro = {}
    for c, prob in c_to_p.items():
        # should it be '<=' or just '<' when comparing to min_prob??
        if prob is None or prob <= EVAL_Constants.min_pr:
            continue # should be treated as NS
        sump += prob
        distro[c] = prob

    max_allowed = 1.0 - EVAL_Constants.NS_allocation
    if EVAL_Constants.up_norm_NS:
        # If it's near enough to max_allowed, no change (up norming) necessary
        if (sump <= max_allowed and sump > max_allowed - 0.001) or sump <= 0.0:
            return distro, 1 - sump
    elif sump <= max_allowed: # (leave as is.. no capping necessary)
        return distro, 1 - sump

    # Capping is necessary.  Normalize/reduce, so the sum is
    # exactly max_allowed
    r = max_allowed / sump
    normed_distro = {}
    sump = 0.0 # recompute sump
    for c, p in distro.items():
        p = r * p
        if p > EVAL_Constants.min_pr:
            normed_distro[c] = p
            sump += p
                
    assert sump < max_allowed + .001
    return normed_distro, 1 - sump

###

# Computes loglossNS (log loss in the presence of NS items).  Obs is
# the observation, and the probability assigned to it is in the distro
# map (possibly 0). NOTE: in general, we assume filter_and_cap() has
# been applied to the distro, using NS_min_prob (so distro does not
# have probabilities below min_prob, and the sum of its entries is at
# most 1-min_prob). Set fc_applied to False otherwise.
def compute_log_loss_NS(obs, is_NS, distro, NS_prob, pit=0, fc_applied=1):
    # The maximum loss that can be incurred.
    NS_loss = -log(EVAL_Constants.NS_allocation)
    prob = distro.get(obs, 0.0)
    if pit:
        print('# (in loss) obs:%s prob:%.3f' % (obs, prob))
    min_prob = EVAL_Constants.min_pr
    if prob < min_prob and fc_applied:
        # should we assume this?? (if we assume FC is applied, yes)
        assert prob == 0.0
    # Predicted as NS ?
    if prob <= min_prob:
        if is_NS:
            loss = min(-log(NS_prob), NS_loss)
            if pit:
                print('# (in loss) both are saying NS, NS_prob: %.3f' % NS_prob)
        else: # take the bigger loss, if it's marked as NS .. (eg. if
            # the predictor is putting all its prob-mass to NS, then take the
            # cost from loss from assigning min_prob...
            loss = max(-log(NS_prob), NS_loss)
            if pit:
                print('# (in loss) ---> predictor says NS, but not the marker..')            
    else: # predicted as a "modeled", or salient, item
        if pit:
            print('# (in loss) -> predictor says the item is IV.. markers is_NS:', is_NS)
        loss = -log(prob)
    if pit:
        print('# (in loss) loss: %.2f' % loss)
    return loss

####

# With k=2, this is standard Brier or quadratic loss.  obs is the
# observed item.  NOTE: we need **all** the predictions (all the
# probabilities here, not just the PR for the observed), which should
# form a sd.
def brier_loss(predictions, obs, k=2):
    lk = 0 # the loss (for k)
    found, sump = 0, 0.0
    for s2, p2 in predictions.items(): # Brier form of scoring
        sump += p2
        if s2 == obs:
            lk += pow(1.0 - p2, k)
            found = True
        else:
            lk += pow(p2, k)
    if not found: # Make sure you punish
        lk += 1   # if not there (OOV case), maximum loss is 1.
        # NOTE: if not there, and no other items predicted,
        # loss is 1.0, but otherwise, Brier score (loss) can be up to 2.0.

    # NOTE: the sd property.
    assert sump <= 1.001, '# sump was %.5f' % sump
    return lk

#################
#
## These below are used for computing lowest achievable ('ideal')
## losses on synthesized sequences.

# Given a SD (the true SD generating the data), compute and return
# loglossNS map for various items (the ideal case). Returns those
# ideal losses for each item (in a map), and also the loss on a noise
# item.
def ideal_llNS_values(sd):
    losses, sump = {}, 0
    for c, p in sd.items():
        sump += p
        losses[c] = -log(p)
    assert sump <= 1.0001
    sump = min(sump, 1.0)
    noise_loss = 10000 # 
    if sump < 1.0:
        noise_loss = -log(1-sump)
    return losses, noise_loss

# Like above, for Brier losses.
def ideal_brier_losses(sd, k=2):
    sump = sum(list(sd.values()))
    assert sump <= 1.0001
    sump = min(sump, 1.0)
    assert 'noise' not in sd
    # copy the values, plus 'noise' item in sd2
    sd2 = dict(sd)
    sd2['noise'] = 1.0 - sump
    losses = {}
    for c, _ in sd2.items():
        losses[c] = brier_loss(sd2, c, k=k)
    return losses, losses['noise']

########

# A referee (a simple algorithm) for marking items as NS (noise) or
# not, to be used in evaluations.
# 
# If window is None (both here and in Constants) use no window (use a
# simple item->count map, no expiration), otherwise the simple Box
# predictor is used.
#
# NS_queue or (3rd-party) referee or arbiter for what should be
# labeled NS ..
#
# >class NS
# 
class NsReferee:
    def __init__(self, freq_threshold=None, window=None):
        self.NS_freq  = freq_threshold
        if freq_threshold is None:
            self.NS_freq  = EVAL_Constants.NS_freq

        # If <= NS_freq, then it's marked NS (NS is true).  NS_freq
        # should be an interger >= 0 (never None). For results to make
        # sense, it should be 2 or higher, that is, to estimate
        # probabilities, some grace period is needed. But for
        # experimental purposes we allow NS_freq=1 or 0.
        assert self.NS_freq >= 0
        
        self.box = None # This one can be None.
        if window is None:
            window = EVAL_Constants.NS_window

        if window is not None:
            # allocate a box predictor when window remains None
            self.box = Box(capacity=window)
        else:
            self.c_to_count = Counter() # map of concept/item to count

    def get_name(self):
        if self.box is None:
            return "Referee for NS marking, unlimited window, freq threshold:%d" % (
                self.NS_freq )
        else:
            return "Referee for NS marking  window size:%d freq threshold:%d" % (
                str(self.box.get_capacity()), self.NS_freq )

    def reset(self):
        if self.box is None:
            self.c_to_count = Counter() # map of concept/item to count
        else:
            self.box.reset()

    def get_is_NS_and_update(self, obs ):
        label = self.is_NS(obs)
        self.update(obs)
        return label

    # Returns true if deemed NS.
    def is_NS(self, obs):
        if self.box is None:
            count = self.c_to_count.get(obs, 0)
        else:
            count = self.box.get_count(obs)
        return count <= self.NS_freq

    def update(self, obs ):
        if self.box is None:
            self.c_to_count[obs] += 1
        else:
            self.box.update(obs)

