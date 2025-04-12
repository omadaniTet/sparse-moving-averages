import numpy.testing as npt
from SMAs import *

# Unit tests for SMAs.
#
# NOTE: some of the asserts (tests) below are with equality and pass
# eventhough the quantities compared are float.  In case the tests do
# not pass in other environments: npt could be used for approximate
# equality, and is used in some places below already.
#


#########  EMA and related tests

def test_harmonic_decay():
    npt.assert_almost_equal(harmonic_decay(1/3.0), 1/4, decimal=4)
    npt.assert_almost_equal(harmonic_decay(1), 1/2.0, decimal=4)
    npt.assert_almost_equal(harmonic_decay(1/5), 1/6.0, decimal=4)
    npt.assert_almost_equal(harmonic_decay(1/5, 0.01), 1/6.0, decimal=4)
    # Test when a minimum is set.
    npt.assert_almost_equal(harmonic_decay(1/5, 1), 1, decimal=4)
    npt.assert_almost_equal(
        harmonic_decay(0.6, 0.01), 1.0/(10/6+1.0), decimal=4)
    npt.assert_almost_equal(
        harmonic_decay(0.6, 0.5), 0.5, decimal=4)

def test_plain_ema_update():
    sd = {}
    plain_ema_update('a', sd, 0.4)
    assert sd == {'a': 0.4}
    plain_ema_update('b', sd, 0.55)
    assert sd['a'] == (1.0 - 0.55) * 0.4
    assert sd['b'] == 0.55
    assert len(sd) == 2

# test EMA updating, status of lr, etc, with and without using
# harmonic decay.
def test_ema_updates_etc():
    ema = EMA(use_harmonic=False, min_rate=0.1,
              max_rate=None, min_prob=0.02)
    assert ema.get_lr() == 0.1
    assert ema.min_prob == 0.02
    assert ema.max_map_entries >= int(1.5 * 1.0/0.02)
    assert ema.get_distro() == {}

    pr = ema.predict_and_update('o')
    assert pr == 0.0 # before the update, it's 0.
    assert ema.get_prob('a') == 0.0
    assert ema.get_prob('o') == 0.1
    assert ema.get_distro() == {'o': 0.1}
    pr = ema.predict_and_update('o')
    assert pr == 0.1
    pr = 0.9 * 0.1 + 0.1
    assert ema.get_prob('o') == pr
    assert ema.get_lr() == 0.1 # No change in lr.
    ema.update('b')
    assert ema.get_distro() == {'b':0.1, 'o':0.9*pr}
    # Now, with changing rates.
    ema2 = EMA(use_harmonic=True, min_rate=0.26,
               max_rate=0.5, min_prob=0.015)
    assert ema2.get_lr() == 0.5
    pr = ema2.predict_and_update('c')
    assert ema2.min_prob == 0.015
    assert pr == 0.0 # before the update, it's 0.
    assert ema2.get_prob('c') == 0.5
    assert ema2.get_lr() == 1.0/3.0
    assert ema2.get_prob('b') == 0.0
    pr = ema2.predict_and_update('c')
    assert pr == 0.5
    npt.assert_almost_equal(ema2.get_prob('c'), 2.0/3.0 * 0.5 + 1.0/3, decimal=5)
    assert ema2.get_lr() == 0.26 # reached the minimum.

def test_ema_prune():
    ema = EMA(min_rate=0.,
               max_rate=0.5, min_prob=0.01)
    ema.updates_since_last_check = 3
    ema.max_map_entries = 5
    ema.ema_sd = {1:.1, 2:.1, 3:0.05, 4:.2}
    ema.prune_map()
    assert ema.updates_since_last_check == 0
    assert len(ema.ema_sd) == 4
    ema.max_map_entries = 3
    ema.min_prob = 0.4 # raise the min_prob, so tiny is 0.16
    ema.prune_map()
    assert ema.ema_sd == {4:.2}
    ema.min_prob = 0.8 # raise the min_prob even higher.
    ema.prune_map()
    assert ema.ema_sd == {}

#########  Qs and related tests

# test the essential functions of ObsQ (a single queue)
def test_ObsQ():    
    q = ObsQ(3)
    # Empty queue
    assert q.get_prob() == 0.0
    assert q.get_pos_count() == 0
    q.positive_update()
    assert not q.is_full()
    assert q.get_prob() == 0.0
    assert q.get_pos_count() == 1 # one cell.
    assert q.get_count() == 1
    q.positive_update()
    assert not q.is_full() # Not at capacity yet.
    assert q.get_prob() == 1.0
    assert q.get_pos_count() == 2 # two cells.
    assert q.get_count() == 2
    q.positive_update()
    assert q.is_full() # At capacity now.
    assert q.get_prob() == 1.0
    assert q.get_pos_count() == 3 # 3 cells.
    assert q.get_count() == 3
    q.negative_update()
    # Uses all 3 cells, but the unbiased formula.
    assert q.get_prob() == 2.0 / 3.0
    assert q.get_count() == 4
    assert q.get_pos_count() == 3 # remains 3 cells.
    assert q.get_counts() == (4, 3)
    q2 = ObsQ(2) # Capacity 2.
    q2.negative_update()
    assert q2.get_pos_count() == 0 # no cell, no effect.
    assert q2.get_count() == 0
    q2.positive_update()
    q2.negative_update()
    assert q2.get_pos_count() == 1
    assert q2.get_count() == 2
    assert q2.get_prob() == 0
    q2.negative_update()
    assert q2.get_prob() == 0
    assert q2.get_count() == 3
    assert not q2.is_full()
    q2.positive_update()
    assert q2.get_prob() == 1.0 / 3.0
    assert q2.get_pos_count() == 2
    assert q2.get_count() == 4
    assert q2.is_full()
    q2.positive_update()
    assert q2.get_pos_count() == 2
    assert q2.get_count() == 2
    assert q2.get_counts() == (2, 2)

####

# test the essential functions of Qs
def test_Qs(cap=3):
    qs = Qs(q_capacity=cap)
    assert qs.get_distro() == {} # Empty SD
    assert qs.get_distro(normalize=1) == {} # Empty SD
    assert qs.get_prob('a') == 0
    pr = qs.predict_and_update('a')
    assert pr == 0.0 # before the update, it's 0.
    assert qs.get_prob('a') == 0.0
    assert qs.get_prob('o') == 0.0
    assert qs.get_distro(normalize=1) == {} # Empty SD
    assert qs.get_distro() == {} # Empty SD
    assert len(qs.get_items()) == 1
    qs.predict_and_update('a')
    assert pr == 0.0 # before the update, it's 0.
    assert qs.get_prob('a') == 1.0 # now, it's 1
    pr = qs.predict_and_update('o')
    assert pr == 0.0 # before the update, it's 0.
    assert qs.get_prob('a') == 1.0 / 2.0 # lowered.
    pr = qs.predict_and_update('o')
    assert pr == 0.0 # (before the update) it's still 0.
    assert qs.get_prob('a') == 1.0 / 3.0 # lowered again.
    assert qs.get_prob('o') == 1.0 
    assert len(qs.get_items()) == 2
    sd = qs.get_distro()
    assert sd.get('o') == 1.0 
    qs.update('e')
    assert len(qs.get_items()) == 3

def test_lowest_prs():
    c_to_pr = {1:0.5, 3:0.2, 4:0.7}
    # Keep top 2
    assert Qs.lowest_prs(c_to_pr, 2) == set([3])
    # Keep top 1
    assert Qs.lowest_prs(c_to_pr, 1) == set([3, 1])
    # Keep all
    assert Qs.lowest_prs(c_to_pr, 10) == set([])
    # Keep top 1
    assert Qs.lowest_prs(c_to_pr, 1, 0.3) == set([3, 1])
    # Below 0.8? all qualify (to be dropped).
    assert Qs.lowest_prs(c_to_pr, 1, 0.8) == set([1, 4, 3])
    # Below 0.8? all qualify (to be dropped).
    assert Qs.lowest_prs(c_to_pr, 10, 0.8) == set([1, 4, 3])
    assert Qs.lowest_prs(c_to_pr, 10, 0.21) == set([3])
    
def test_Qs_prune():
    qs = Qs(q_capacity=2, min_prob=0.01)
    qs.updates_since_last_check = 3
    qs.max_map_entries = 5
    qs.update('d')
    for i in range(2):
        qs.update('a')
        qs.update('b')
        qs.update('c')
    qs.prune_map()
    assert qs.get_prob('a') == 1.0/5. 
    assert qs.updates_since_last_check == 0
    assert len(qs.get_items()) == 4
    qs.max_map_entries = 3
    qs.prune_map()
    # item 3 with lowest weight is dropped.
    assert len(qs.get_items()) == 3
    assert qs.get_prob('d') == 0.0 # 'd' is dropped.
    # 'a' is kept.
    assert qs.get_prob('a') == 1.0/5.0 
    assert qs.get_prob('c') == 1.0/3.0
    # Change min PR..
    qs.min_prob = 0.21
    qs.prune_map()
    assert len(qs.get_items()) == 3 # no change.
    qs.min_prob = 0.9 # tiny PR is now high.. tiny is 0.81 ..
    qs.prune_map()
    assert len(qs.get_items()) == 0
    assert qs.get_prob('c') == 0

######

def test_TimeStampQs_updates_etc():
    tsq = TimeStampQs(q_capacity=3)
    assert tsq.get_prob('a') == 0.0
    assert tsq.get_distro() == {}
    tsq.update('a')
    assert tsq.get_prob('a') == 0.0
    pr = tsq.predict_and_update('a')
    assert tsq.get_prob('a') == 0.0 and pr == 0.0
    assert tsq.get_distro() == {}
    pr = tsq.predict_and_update('a')
    # Requires 3 observations after which it shows positive probability.
    assert tsq.get_prob('a') == 1.0 and pr == 0.0
    assert tsq.get_distro() == {'a':1}
    pr = tsq.predict_and_update('b')
    assert tsq.get_prob('b') == 0.0 and pr == 0.0
    pr2 = tsq.get_prob('a')
    npt.assert_almost_equal(tsq.get_prob('a'), 0.5, decimal=5)
    pr = tsq.predict_and_update('b')
    assert tsq.get_prob('b') == 0.0 and pr == 0.0
    pr2 = tsq.get_prob('a')
    npt.assert_almost_equal(tsq.get_prob('a'), 1/3.0, decimal=5)
    pr = tsq.predict_and_update('b')
    pr1, pr2 = tsq.get_prob('a'), tsq.get_prob('b')
    assert pr2 == 1.0 and pr == 0.0
    npt.assert_almost_equal(pr1, 1/4.0, decimal=5)
    assert tsq.get_distro() == {'a':pr1, 'b':pr2}

# Test when the observations are themseleves fractions (not 0 and 1)
def test_TimeStampQs_fractions():
    tsq = TimeStampQs( q_capacity=3, with_proportions=True )
    assert tsq.get_prob('a') == 0.0
    tsq.update('a', 0.3)
    assert tsq.get_prob('a') == 0.0
    tsq.update('a', 0.5)
    assert tsq.get_prob('a') == 0.3
    tsq.update('a', 0.2)
    assert tsq.get_prob('a') == 0.4 # Average of 0.5 and 0.3
    tsq.update('b', 0.7)
    assert tsq.get_prob('a') == 0.4 # Still, average of 0.5 and 0.3
    assert tsq.get_prob('b') == 0.0
    tsq.update('b', 0.6)
    assert tsq.get_prob('b') == 0.7
    npt.assert_almost_equal(tsq.get_prob('a'), 0.2/3, decimal=5)

#######
#### Box tests

def test_Box():
    box = Box(capacity=5)
    assert box.get_prob('a') == 0.0
    box.update('a')
    assert box.get_prob('a') == 1.0
    assert box.get_distro() == {'a':1.0}
    pr = box.predict_and_update('b')
    assert pr == 0.0
    assert box.get_prob('a') == 0.5
    assert box.get_prob('b') == 0.5
    assert box.get_distro() == {'a':0.5, 'b':0.5}
    box.update('c')
    assert box.get_prob('a') == 1.0/3.0
    assert box.get_prob('c') == 1.0/3.0
    box.update('a')
    assert box.get_prob('a') == 2.0/4.0
    box.update('a')
    assert box.get_prob('a') == 3.0/5.0
    assert box.get_prob('b') == 1.0/5.0
    box.update('a')
    assert box.get_prob('a') == 3.0/5.0
    assert box.get_prob('b') == 1.0/5.0
    box.update('a')
    assert box.get_prob('a') == 4.0/5.0
    assert box.get_prob('b') == 0.0
    sd = box.get_distro()
    assert sd == {'a':4/5.0, 'c':1/5.0}

######
### DYAL tests


def test_DYAL_updates_etc():
    dyal = DYAL(q_capacity=3, min_lr=0.01, min_prob=0.03)
    assert dyal.get_distro() == {} # Empty SD
    assert dyal.min_prob == 0.03
    assert dyal.get_prob('a') == 0.0
    assert dyal.get_lr('a') == None
    pr = dyal.predict_and_update('e')
    assert dyal.get_lr('e') == None
    assert pr == 0.0
    pr = dyal.predict_and_update('e')
    assert dyal.get_lr('e') == None
    assert pr == 0.0 and dyal.get_prob('e') == 0.0
    assert dyal.get_distro() == {}
    pr = dyal.predict_and_update('e')
    assert pr == 0.0
    pr = dyal.get_prob('e')
    assert pr > 0.9
    assert dyal.get_distro() == {'e':pr}
    lr = dyal.get_lr('e')
    assert lr > 0.1 
    assert dyal.get_max_rate_tup() == (lr, 'e', pr)
    pr2 = dyal.predict_and_update('d')
    assert pr2 == 0.0
    assert dyal.get_prob('e') < pr # 'e' is lowered
    pr = dyal.predict_and_update('d')
    assert pr == 0.0
    assert dyal.get_prob('e') < 0.75 # 'e' is lowered
    assert list(dyal.get_distro().keys()) == ['e']
    pr = dyal.predict_and_update('d')
    assert pr == 0.0
    pr2, lr2 =  dyal.get_prob('d'), dyal.get_lr('d')
    pr3 =  dyal.get_prob('e')
    assert dyal.get_distro() == {'d':pr2, 'e':pr3}
    assert lr2 > dyal.get_lr('e')
    assert dyal.get_max_rate_tup() == (lr2, 'd', pr2)
    lrs = dyal.get_lrs()
    assert lrs['d'] == lr2

    
def test_dyal_prune():
    dyal = DYAL(q_capacity=3, min_lr=0.01, min_prob=0.01)
    dyal.updates_since_last_check = 3
    dyal.max_map_entries = 5

    # Items 1 through 4, updated 4 times in sequence.
    for t in range(4):
        for item in range(1, 5):
            dyal.update(item)

    dyal.prune_maps()
    assert dyal.updates_since_last_check == 0
    assert len(dyal.get_distro()) == 4
    
    dyal.max_map_entries = 3
    dyal.min_prob = 0.4 # raise the min_prob, so tiny is 0.16
    dyal.prune_maps()
    # print(dyal.get_distro())
    # One item is dropped.
    assert len(dyal.get_distro()) == 3
    dyal.min_prob = 0.6 # raise min prob. All will be dropped.
    dyal.prune_maps()
    assert len(dyal.get_distro()) == 0

####

def tests_EMA():
    test_harmonic_decay()
    test_plain_ema_update()
    test_ema_updates_etc()
    test_ema_prune()

###

def tests_Box():
    test_Box()

###

def tests_Qs():
    test_ObsQ()
    test_Qs(2)
    test_Qs(5)
    test_lowest_prs()
    test_Qs_prune()

####

def tests_TimeStamp():
    test_TimeStampQs_updates_etc()
    test_TimeStampQs_fractions()

###

def tests_DYAL():
    test_DYAL_updates_etc()
    test_dyal_prune()
    
###

if __name__ == "__main__":


    tests_EMA()
    tests_Box()
    tests_Qs()
    tests_TimeStamp()
    tests_DYAL()

