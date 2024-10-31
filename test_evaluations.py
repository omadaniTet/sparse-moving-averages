# Test FC(), logloss, Quadloss/Brier, etc.

import numpy.testing as npt
from math import log
from sma_eval_utils import *


# test_filter_and_cap()
def test_fc():
    EVAL_Constants.min_pr = 0.1
    EVAL_Constants.NS_allocation = 0.01
    sd = {'a':0.2}
    assert filter_and_cap(sd) == ({'a':0.2}, 0.8)
    EVAL_Constants.min_pr = 0.22 
    assert filter_and_cap(sd) == ({}, 1.0)
    EVAL_Constants.min_pr = 0.22 
    sd = {'a':0.23, 'b':0.1}
    assert filter_and_cap(sd) == ({'a':.23}, 0.77)
    sd = {'a':0.23, 'b':0.8}
    sd2, remainder = filter_and_cap(sd)
    npt.assert_almost_equal(sd2['a'], 0.221, decimal=3)
    npt.assert_almost_equal(remainder, 0.01, decimal=3)
    assert sum(list( sd2.values() )) <= .9901
    EVAL_Constants.NS_allocation = 0.02
    # This time, 'a' is dropped because it goes below 0.22
    sd2, remainder = filter_and_cap(sd)
    npt.assert_almost_equal(remainder, 0.239, decimal=3)
    npt.assert_almost_equal(sd2['b'], 0.761, decimal=3)
    assert len( sd2 ) == 1 # it only has 'b'
    EVAL_Constants.min_pr = 0.01 # we lower min pr threshold.
    sd2, remainder = filter_and_cap(sd)
    # Remainder is 0.02 and both are kept.
    npt.assert_almost_equal(remainder, 0.02, decimal=3)
    assert len( sd2 ) == 2
    assert sum(list( sd2.values() )) <= .9801

###

def test_log_loss_NS():
    # compute_log_loss_NS('a', is_NS, distro, NS_prob, pit=0)
    sd = {'a':0.23, 'b':0.8}
    NS_prob = 0.01
    assert compute_log_loss_NS('a', False, sd, NS_prob) == -log(0.23)
    assert compute_log_loss_NS('a', True, sd, NS_prob) == -log(0.23)
    assert compute_log_loss_NS('a', True, sd, 0.3) == -log(0.23)
    assert compute_log_loss_NS('c', True, sd, 0.3) == -log(0.3)
    assert compute_log_loss_NS('c', True, sd, 0.15) == -log(0.15)
    EVAL_Constants.min_pr = 0.24
    assert compute_log_loss_NS('a', True, sd, 0.3, fc_applied=0) == -log(0.3)
    NS_loss = -log(EVAL_Constants.NS_allocation)
    # In case the sd and referee don't agree, incur NS_loss
    assert compute_log_loss_NS('c', False, sd, 0.15) == NS_loss

###

def test_brier_loss():
    # brier_loss(predictions, obs, k=2)
    assert brier_loss({'b': 1.0}, 'a') == 2 # maximum loss
    assert brier_loss({'b': 0.5}, 'a') == 1 + 0.5*0.5
    assert brier_loss({'b': 0.1}, 'a') == 1.01
    assert brier_loss({'a': 0.1}, 'a') == 0.9 * 0.9
    assert brier_loss({'a': 1.}, 'a') == 0
    assert brier_loss({'a': .5, 'b':.2 , 'c':.3}, 'a') == 0.25 + .04 + 0.09
    # loss on 'c'
    assert brier_loss({'a': .5, 'b':.2 , 'c':.3}, 'c') == 0.25 + .04 + 0.49

##

def test_ideal_llNS_values():
    EVAL_Constants.min_pr = 0.01
    EVAL_Constants.NS_allocation = 0.01
    sd = { 1:.1, 2:.2 }
    NS_prob = 0.01
    l1 = {}
    for c, p in sd.items():
        l1[c] = compute_log_loss_NS(c, False, sd, NS_prob )
    l2, noise_loss = ideal_llNS_values(sd)
    assert l1 == l2
    npt.assert_almost_equal(noise_loss, -log(0.7), decimal=3)
    sd = { '1':.99 }
    l2, noise_loss = ideal_llNS_values(sd)
    npt.assert_almost_equal(noise_loss, -log(0.01), decimal=3)
    npt.assert_almost_equal(l2['1'], -log(0.99), decimal=3)
    sd = {  }
    l2, noise_loss = ideal_llNS_values(sd)
    npt.assert_almost_equal(noise_loss, -log(1.0), decimal=3) # 0 loss
    assert l2 == {} # empty

##

def test_ideal_brier_losses():
    sd = { 1:.1, 2:.2 }
    sd2 = { 1:.1, 2:.2, 'noise':.7 }
    l1 = {}
    for c, p in sd.items():
        l1[c] = brier_loss(sd2, c)
    l2, noise_loss = ideal_brier_losses(sd)
    assert l2['noise'] == noise_loss
    l2.pop('noise')
    assert l1 == l2
    sd = { 1:1.0 }
    l2, noise_loss = ideal_brier_losses(sd)
    assert l2 == {1:0, 'noise':2.0}
    assert noise_loss == 2.0
    sd = { }
    l2, noise_loss = ideal_brier_losses(sd)
    assert l2 == { 'noise':0.0} and noise_loss == 0.0

###

def  test_referee():
    ref = NsReferee(freq_threshold=2)
    ref.update('o' )
    assert ref.is_NS('o') and ref.is_NS('a')
    ref.update('a' )
    ref.update('o' )
    assert ref.is_NS('o') and ref.is_NS('a')
    ref.update('a' )
    ref.update('o' )
    # 'o' is no longer NS.
    assert not ref.is_NS('o') and ref.is_NS('a')
    ref.update('a' )
    assert not ref.is_NS('o') and not ref.is_NS('a')
    # A few corner (special or near special) cases.
    ref2 = NsReferee(freq_threshold=0)
    assert ref2.is_NS('o') and ref2.is_NS('a')
    ref2.update('o' )
    assert not ref2.is_NS('o') and ref2.is_NS('a')
    ref2.update('o' )
    assert not ref2.is_NS('o') and ref2.is_NS('a')
    ref3 = NsReferee(freq_threshold=1)
    assert ref3.is_NS('o') and ref3.is_NS('a')
    ref3.update('o' )
    assert ref3.is_NS('o') and ref3.is_NS('a')
    ref3.update('o' )
    assert not ref3.is_NS('o') and ref3.is_NS('a')
    ########
    # With a finite window specified (uses Box).
    ref4 = NsReferee(freq_threshold=0, window=3)
    assert ref4.is_NS('o') and ref4.is_NS('a')
    ref4.update('o' )
    assert not ref4.is_NS('o') and ref4.is_NS('a')
    ref4.update('a' )
    assert not ref4.is_NS('o') and not ref4.is_NS('a')
    ref4.update('a' )
    assert not ref4.is_NS('o') and not ref4.is_NS('a')
    ref4.update('a' ) # Now, 'o' is ejected.
    assert ref4.is_NS('o') and not ref4.is_NS('a')
    ref5 = NsReferee(freq_threshold=1, window=2)
    ref5.update('o' )
    assert ref5.is_NS('o') and ref5.is_NS('a')
    ref5.update('o' )
    assert not ref5.is_NS('o') and ref5.is_NS('a')
    ref5.update('a' )
    assert ref5.is_NS('o') and ref5.is_NS('a')
    ref5.update('o' )
    assert ref5.is_NS('o') and ref5.is_NS('a')
    ref5.update('a' )
    assert ref5.is_NS('a')
    ref5.update('a' )
    assert ref5.is_NS('o') and not ref5.is_NS('a')

    

###

if __name__ == "__main__":


    test_fc()
    test_log_loss_NS()
    test_brier_loss()
    test_ideal_llNS_values()
    test_ideal_brier_losses()
    test_referee()

