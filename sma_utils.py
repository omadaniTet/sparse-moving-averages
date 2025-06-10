#

#
# This is on top of Concept, Expedition, etc.

# So utilities that use fns from Concept file, ema_variants file,
# etc.. So if a class from Concepts file wants to use these, then put
# them in pg_utils.

import string, random, time
import gzip, re, math, numpy as np

from math import log

from os import listdir
from os.path import isfile, join

from collections import defaultdict, Counter

# from concept2 import *

##

# timer!
def get_mins_from( from_time ):
    delta = time.time() - from_time
    return delta / 60.0

def get_hrs_from( from_time ):
    delta = time.time() - from_time
    return delta / 3600.0

##

# KL(p || q), where both are maps..
# divergence of distro q from p.
#
def kl_general(ps, qs):
    s = 0
    for i, p in ps.items():
        q = qs.get(i, 0.0)
        assert q > 0 # what to do with 0s?
        s += p * log(p / q)
    return s
##

# bounded kl() or relative_entropy() ..  for the binary case..
def kl_bounded(a, b):
    """a and b are probabilities, and their order matters.  So
    relatively entropy of b with respect to a (or divergence of b from
    a)..  Using natural log or ln (for 'nats').

    """
    assert(b > 0)
    if a >= 1.0:
        return a * log(a / b)  # short circuit.  
    if b >= 0.999: # so 1-b is not zero..
        b = 0.999
    return a * log(a / b) + (1 - a) * log(  (1-a) / (1-b) )

####### Losses

# In these, we assume probs1 and probs2 are distributions or at least
# semidistributions..

# sample according to probs1, but incur costs accoring to probs2. This
# is the logloss scoring rule.
def expected_logloss(probs1, probs2):
    s = 0.0
    for c, p in probs1.items():
        s += p * -log(probs2.get(c))
    return s

# This turned out to not be proper!!
def expected_quadloss(probs1, probs2):
    s, sump = 0.0, 0.0
    for c, p in probs1.items():
        d = (1.0 - probs2.get(c)) * (1.0 - probs2.get(c)) * p
        s +=  d
        print(d, '  sum=', s)
        sump += p
    assert sump == 1.0
    return math.sqrt(s)

def brier_score(preds_list, obs, k=2):
    if type(preds_list) == dict:
        preds_list = list(preds_list.items())
    lk = 0
    found, sump = 0, 0.0
    for s2, p2 in preds_list: # Brier form of scoring
        sump += p2
        if s2 == obs:
            lk += pow(1.0 - p2, k)
            found = 1
        else:
            lk += pow(p2, k)
    if not found: # Make sure you punish
        lk += 1   # if not there (OOV)..
        # NOTE: if not there, and no other items predicted,
        # loss is 1.0, but otherwise, Brier score (loss) can be up to 2.0.
    assert sump <= 1.001, '# sump was %.5f' % sump
    return lk

# Brier score, or scoring, rule for multiclass, min is 0, max is 2.
# (Brier distances or diffs.. Brier distance is a better name than
# Brier score.. but i guess if you are assessing the calibration of a
# model.. a 'score' is a better term..).. k is the power, and k=2,
# is the standard Brier.
def expected_brier_score(probs1, probs2, k=2):
    s, sump = 0.0, 0.0
    for c1, p1 in probs1.items():
        d, found = 0, False
        # Brior reqiure you to go over all elements in p2.
        for c2, p2 in probs2.items():
            if c2 == c1:
                d += pow(abs(1.0 - p2), k) # * (1.0 - p2)
                found = 1
            else:
                d += pow(p2, k) # * p2
        if not found:
            d += 1
        s +=  d * p1
        #print(d, '  sum=', s)
        sump += p1
    assert sump <= 1.001, 'sump was %.3f' % sump
    return s


#######

# Should we move these to its own classes? 'encoders' or 'coders',
# etc..

# For converting to a smaller alphabet, eg binary (only two
# primitives!).
#
# TODO: also support varying widths instead of only fixed widths for
# the code: with a fixed width, all 8 bits, the learner my cheat and
# somehow discover the width to lower its bad ratio :) (thought it is
# unlikely?). One way to do this (but below, the 'progressive
# counting' is easier/better): keep the lower alphabet at lower
# capacity and then when you run out of codes, use the original
# chars.. Example, with width k=2, instead of 10 possibilities, to
# generated 100 or fewer unique possibilities (number of unique chars
# is below 100 in all our experiments so far), use 9 or 8 chars
# only... so that yield 64 or 81 unique. Randomly assign the codes
# (don't do this by freq of char), and when run out, use the original
# length 1 chars! (some frequent chars should get themseleves as their
# code..). This method yields k length code or 1 length codes. Perhaps
# we can extend it to get more variety of lengths. Here's one: say we
# have binary (or two primitives), then '0', '1', '10', '11', '110',
# as all codes... assign them randomly to the original chars (so the
# most frequent one should not necessarily get 0 or a low count)...
# 
#
# >class 
class binary_coder: # binary or larger alphabet size..

    # Generate  codes (binary or higher up), one code per char.
    # The codes are generated all.
    # The codes are assigned in an online manner.
    def __init__(self, n=100, l=2, do_shuffle=True, fixed_width=True, pit=0):
        """n is total number of (different) characters we need to
        support. In most our files, it's no more than 100.  Each
        character is converted to a code of string length l, when
        fixed_width is true, or otherwise, up to l. If l=2, then we
        get an alphabet size closest to original size. If l=7 or 8,
        then we get binary..

        """
        self.n = n # number to generate for...
        self.char_to_code = {} # map from char to code str
        self.char_to_count = Counter() # just for stats, eg alignment info
        self.assigned_codes = set() # set of assigned codes
        self.fixed_width = fixed_width
        self.l = l # length of code (in case of fixed_width)
        self.codes, self.k = self.generate_codes(n, self.l, do_shuffle, pit=pit)
        # indexes into codes array, points to the next available code.
        self.avail = 0
        # for computing bad  splits..
        self.do_left_ends = False # left end or right end?
        print('\n#')

    def set_fixed_width(self, status=True):
        self.fixed_width = status
        self.codes, self.k = self.generate_codes(
            self.n, self.l, do_shuffle, pit=pit)
    
        # usa2616welcome
    # generate codes, each of fixed length l, or otherwise variable
    # length. They should be at least n such, to go around.
    def generate_codes(self, n, l=None, do_shuffle=True, k=None, pit=0):
        # k will be size of alphabet set ('0', '1', '2', ..., k-1).
        # k * k * .. = k^l >= n, or l * log(k)  >= log(n)
        #    exp( log(k)) > exp(log(n)/l)
        if k is None:
            assert l is not None
            # k = int( math.exp(math.log(n)/l) + 1 ) # size of alphabet
            k = 2
            while True:
                prod = 1
                for i in range(l):
                    prod *= k
                if prod >= n:
                    break
                else:
                    k += 1
            assert k <= 10 # can't handle more than 10 (otherwise, need to
            # use something like 'a', 'b', 'c', too)!
        if  self.fixed_width:
            print (('\n# coding (to smaller alphabet) is invoked, alphabet size ' +
                    'of %d for fixed code of len=%d ') % (k, l))
            # generate codes, each of length l.
            codes =  self.fixed_len_l_codes(k, l)
        else: # variable width
            codes =  self.variable_len_codes(k, n)
            
        # random permute the codes..
        if do_shuffle:
            random.shuffle(codes)
        print('\n# num codes generated:', len(codes), '\n')
        if pit:
            for i in range(len(codes)):
                print('# code:', i+1, ' is: "%s"' % codes[i])
        return codes, k

    ###

    def variable_len_codes(self, k, num_to_gen):
        last_loop, j = [],  2
        print('# alphabet size k is:', k, '  num to gen:', num_to_gen)
        #exit(0)
        for i in range(k):
            last_loop.append(str(i))
        codes = last_loop
        while True:
            if len(codes) >= num_to_gen:
                break
            print('# loop:', j, ' num codes so far:', len(codes))
            new_codes = []
            for c in last_loop:
                for i in range(k):
                    new_codes.append(c + str(i))
            codes.extend(new_codes)
            last_loop = new_codes
            j += 1
        return codes

    ###
    
    # Generate all possible codes of fixed length l, alphabet set of
    # size k.
    @classmethod
    def fixed_len_l_codes(cls, k, l):
        codes = []
        for i in range(k):
            codes.append(str(i))
        for _ in range(l-1):
            new_codes = []
            for c in codes:
                for i in range(k):
                    new_codes.append(c + str(i))
            codes = new_codes
        return codes

    # get the (binary) code for character ch.
    def get_code_for(self, ch):
        return self.char_to_code[ch] 
    
    # get code for char c
    def convert(self, c, update_counts=1):
        code = self.char_to_code.get(c, None)
        if update_counts:
            self.char_to_count[c] += 1
        if code is None: # Assign a code
            code = self.codes[ self.avail ]
            self.assigned_codes.add(code)
            self.char_to_code[c] = code # get_code_for
            print('\n# in coder..  char "', c, '" got code:',
                  code, ' index=', self.avail)
            self.avail += 1 # Advance available id.
        return code

    # Convert a line or a string to a string (list/array) of codes
    def convert_str(self, s):
        cs = ''
        for c in s:
            cs += self.convert(c)
        return cs

    # nc = num_chars to convert
    def convert_a_sample(self, line, nc):
        ll = len(line)
        # pick a random location, from 0, to len(line) - nc
        nc = min(ll, nc)
        i = 0
        if nc < ll:
            i = random.randint(0, ll - nc)
        sampled = line[i : i + nc]
        coded = self.convert_str(sampled)
        # print('\n# ** coded and sampled:', coded, sampled)

        # With fixed_width, this should be the case.
        if  self.fixed_width:
            assert len(coded) == len(sampled) * self.l
        return coded, sampled

    #########

    # Just for info..
    def get_chars_priors(self, k=None, pit=1):
        tot = sum(list(self.char_to_count.values()))
        pairs = []
        for ch, c in self.char_to_count.items():
            pairs.append((ch, 1.0 * c / tot))
        if pit:
            print('\n# in coder: num chars seen: ', len(pairs), '\n')
        pairs.sort(key=lambda x: -x[1]) # descending prior
        if k is not None:
            pairs = pairs[:k] 
        return pairs
            
    ###
    
    ### for computing alignment
    
    # advance the indexes for code/chunk, and empty props list
    def advance_code(self, j, prev, lc):
        j += 1
        prev += lc
        return j, prev, []
    
    def advance_char(self, i, l, r, w):
        i += 1
        r += w
        l += w
        return i, l, r

    def measure_overlap(self, l1, r1, l2, r2):
        #r2 = l2 + size2 - 1
        ov_start = max(l1, l2)
        ov_end = min(r1, r2)
        ov = ov_end - ov_start + 1
        return ov

    ###

    # orig should be the original line, an array/seq of characters.
    # make a position-to-char map, i -> ch (which character ch is
    # covering position i map).
    def make_pos_to_char(self, orig):
        pos_to_char = {}
        i = 0
        for ch in orig:
            code = self.get_code_for(ch)
            lc = len(code)
            for pos in range(i, i+lc):
                pos_to_char[pos] = ch
            i += lc
        return pos_to_char
                
    # This should work for both variable and fixed width, and handle
    # overlaps among chunks.  Returns for each chunk (code) that
    # corresponds to a char, its lists of proportion overlaps.
    def get_alignment_info_variable_width(
            self, chunks, orig, index_pairs):
        pos_to_char = self.make_pos_to_char(orig)
        codes = self.assigned_codes # for a fast check..
        # Go over each chunk that corresponds to a char, and update
        # its overlap stats (fractions).
        i = 0
        prop_lists = defaultdict(list) # map of chunks to list of proportions.
        for ch_list in chunks:
            # make into a string (from a list)
            chunk = ''.join(ch_list)
            if chunk not in codes:
                i += 1
                continue
            # coverage pointers (indices) of chunk i.
            l, r = index_pairs[i]
            ovs = {} # map of ch -> overlap counts
            prev_ch, count = None, 0
            # because the chunk can presumably overlap the same char
            # in non-continguous positions, we will keep track of the
            # max contiguous overlap count..
            for j in range(l, r+1):
                ch = pos_to_char.get(j)
                if prev_ch is None or prev_ch != ch:
                    if prev_ch is not None:
                        c2 = ovs.get(prev_ch, 0)
                        ovs[prev_ch] = max(c2, count)
                    prev_ch = ch
                    count = 1
                else:
                    count += 1
            # One last time (for the last seen char).
            assert prev_ch is not None
            c2 = ovs.get(prev_ch, 0)
            ovs[prev_ch] = max(c2, count)
            props = [] # now make it a list of proportion
            lc = 1.0 * (r - l + 1)
            for ch, count in ovs.items():
                props.append((ch, count / lc )) # insert ratio
            prop_lists[chunk].append(props)
            i += 1
        return prop_lists


    # Works for fixed width chunks only, currently.
    def get_alignment_info_fixed_width(self, chunks, orig, min_factor=2):
        w = self.l # the width
        minw = w / min_factor
        l2, l1, r1 = 0, 0, w - 1 # l2 points to first char (or bit) of current chunk
        # there can be multiple appearance of a chunk, so this will
        # be a list of lists.
        al = defaultdict(list)
        # i indexes into orig, j into chunks..
        # sc is size of chunks array..
        i, j, lo, sc = 0, 0, len(orig), len(chunks)
        props = [] # proportions list for a chunk
        while i < lo and j < sc:
            # for c in chunks
            ch = orig[i]
            # make into a string (from a list)
            chunk = ''.join(chunks[j])
            lc = len(chunk) 
            if lc <= minw: # skip this chunk (too short)
                j, l2, props = self.advance_code(j, l2, lc) 
                if l2 > r1:
                    i, l1, r1 = self.advance_char(i, l1, r1, w)
                    assert l1 <= l2 and l2 <= r1
                continue
            r2 = l2 + lc - 1
            ov = self.measure_overlap(l1, r1, l2, r2)
            assert ov > 0 , 'i,j:%d,%d l2:%d l1:%d r1:%d' % (
                i, j, l2, l1, r1)
            assert r1 >= l2, 'chunk len:%d l2:%d l1:%d r1:%d' % (lc, l2, l1, r1)
            #print('# 333 chunk:', chunk, 'ov:', ov, 'lc:', lc, 'i', i, 'j', j)
            p = 1.0 * ov / lc
            assert p > 0
            props.append ( (ch, p) )
            # at least  one of the two should be advanced, possibly both.
            do_advance_chunk = r1 >= r2 # l2+lc-1
            do_advance_char = r1 <= r2 # l2+lc-1
            if do_advance_chunk:
                al[chunk].append(props)
                j, l2, props = self.advance_code(j, l2, lc)
            if do_advance_char:
                i, l1, r1 = self.advance_char(i, l1, r1, w)
            # at least one should be true
            assert do_advance_char or do_advance_chunk

        # last one not empty? add it.
        if props != []:
            al[chunk].append(props)
        #if len(al) > 0:
        #   print('# len al:', len(al))
        return al

    ############
    ##### Return number of splits, bad split ratio, etc (eg alignment
    ##### info)

    def get_split_info_variable_width(self, index_pairs, orig):
        # if self.do_left_ends:
        #    return self.get_split_info_left_ends(index_pairs, orig)
        if self.do_left_ends:
            ends = [x[0] for x in index_pairs] # left ends
        else:
            ends = [x[1] for x in index_pairs] # right ends
        num_origs = len(orig)
        # The last or right-most split is often easy to catch as
        # good. but, we'll leave it (just in case index_pairs doesn't
        # go all the way to end..)
        num_splits = len(ends)
        i, j, num_bad = 0, 0, 0
        split = ends[j] # current char, and proposed split
        good_idx = 0
        if not self.do_left_ends: # do right ends?
            good_idx = len(self.get_code_for(orig[0]))-1
        while True:
            # j points to the proposed splits
            num_bad += split < good_idx # (if two match, it's not bad!)
            advance_i = split >= good_idx
            advance_j = split <= good_idx
            if advance_i:
                i += 1
                if i >= num_origs:
                    num_bad += num_splits - (j + advance_j)
                    break
                # the index of the next good split.
                if self.do_left_ends: # do right ends?
                    good_idx += len(self.get_code_for(orig[i-1]))
                else:
                    good_idx += len(self.get_code_for(orig[i]))
            if advance_j: # go to next proposed split..
                j += 1
                if j >= num_splits: 
                    break
                split = ends[j]
        return num_splits, num_bad

    #
    #
    # We may not need this! (both handled within same fn above)
    """
    def get_split_info_left_ends(self, index_pairs, orig):
        ends = [x[0] for x in index_pairs] # left ends
        num_origs = len(orig)
        # the 1st (left-most) split counts, as good or bad?
        # this could be easily good split..
        num_splits = len(ends)
        i, j, num_bad = 0, 0, 0
        # current char, and current proposed split
        ch, split = orig[i], ends[j]
        good_idx = 0 #   The 1st good index.
        while True:
            # j points to the proposed splits
            num_bad += split < good_idx # (if two match, it's not bad!)
            advance_i = split >= good_idx
            advance_j = split <= good_idx
            if advance_i:
                i += 1
                if i >= num_origs:
                    num_bad += num_splits - (j + advance_j)
                    break
                ch = orig[i-1]
                # the index of the next good split.
                good_idx += len(self.get_code_for(ch))
            if advance_j: # go to next proposed split..
                j += 1
                if j >= num_splits: 
                    break
                split = ends[j]
        return num_splits, num_bad
    """
    
    ##
    
    # Report/return num splits, and ratio of bad splits, and also
    # alignment info.. orig is the original line, with normal
    # characters (sequence of chars)! chunks is the interpretation, ie
    # a sequence of intervals/phrases/coverages, possibly overlapping
    # (l2r). index_pairs is the array of coverage indices/pointers by
    # these chunks.
    def get_split_info(
            self, index_pairs, chunks, ov_list, orig, do_variable=None):

        # assert self.fixed_width # for now..
        al = None
        # Note: as of this writing we could get split info using
        # variable width assumption, even if the encoding is fixed
        # width.. but currently not supporting al (alignment
        # info/eval).
        if do_variable or not self.fixed_width: # handle variable width
            # return
            num_splits, bad = self.get_split_info_variable_width(
                index_pairs, orig)
            al = self.get_alignment_info_variable_width(
                chunks, orig, index_pairs) 
        else:
            num_splits, bad, al = self.get_split_info_fixed_width(
                index_pairs, orig, chunks, ov_list)
        bad_ratio = 1.0 * bad / num_splits
        return num_splits, bad_ratio, al

    # NOTE1: Only works if the encoding is fixed width..
    # (returns al info too, if the chunks, is not None or empty..  )
    # NOTE2: Cant handle overlaps..
    def get_split_info_fixed_width(
            self, index_pairs, orig,
            chunks=None, ov_list=None, misc_constraints=1):
        # For now, assume we didn't allow overlaps when
        # interpretting (and with fixed-width encoding)..
        assert ov_list is None or ov_list == []
        # without overlap allowed, and fixed-width sum of chunk length
        # should be |orig|*l
        
        #assert sum( [len(x) for x in chunks] ) == len(orig) * self.l, \
        #    '"%s", chunk len=%d, sum:%d' % (
        #        orig, self.l * len(orig), sum( [len(x) for x in chunks] )  )

        lens = [ (x[1] - x[0]+1) for x in index_pairs ]
        if misc_constraints:
            # misc constraints check (eg if we assume
            # the splits could be a subset of original phrases
            # we could
            assert sum( lens ) == len(orig) * self.l, \
                '"%s", chunk len=%d, sum lens:%d' % (
                    orig, self.l * len(orig), sum( lens )  )

        al = None
        if chunks is not None and chunks != []:
            al = self.get_alignment_info_fixed_width(chunks, orig)
        
        # with fixed length, should be easy to compute bad split count.
        num_splits = len(lens) # - 1 (should we ignore the last one?) variable-width does not..
        # num_splits = len(chunks) 
        if num_splits <= 0:
            if num_splits >= 0:
                print('# no splits:', chunks[0], ' ,  orig: "%s"' % orig)
            else:
                print('# no chunk ..  orig:', orig)                
            return 0, 0, al

        s, bad, i = 0, 0, 0 # sum, num bad
        #for chunk in chunks:
        for (l, r) in index_pairs:
            #print((l, r), r % self.l)
            if self.do_left_ends: #  = False # left end or right end?
                idx = l
            else:
                idx = r + 1
            if idx % self.l > 0:
                bad += 1
                    

        if 0:
            print('# l:', self.l, 'num splits:', num_splits, ' bad:', bad,
                  # ' diff:', num_splits - bad,
                  'ratio: %.3f' % ((1.0 * bad) / num_splits))
        #bad_ratio = 1.0 * bad / num_splits
        return num_splits, bad, al

    
####

# This is an older version of above: could be deprecated because of
# above ( i think the above is more versatile).
#
# (coding/encoding, deciphering, etc)
# 
# >class
class BinaryConvert:

    # One way to convert to binary is to keep a counter: every time a
    # new char is seen, the counter is incremented and translated to a
    # string (new binary code). Other ways: random string allocation,
    # hoffman codes, etc.
    available = 0 # the counter
    width = 8 # number of bits in every binary code..
    mapper = {} # chars to code
    
    @classmethod
    def convert_line(cls, line, pick_a_term=False, insert_spaces=True):
        newl = ''
        phrase = ''
        if pick_a_term: # pick one term
            parts = line.split(' ')
            i = EG.rand.randint(0, len(parts) - 1) # inclusive
            line = parts[i]
            phrase = line
            # print('\n# index ', i, 'and term picked it:', line)
        for x in line:
            if insert_spaces and newl != '':
                newl += ' ' # put a space separating the codes for chars!
            newl += cls.convert_char(x)
        #print('line:', line)
        #print('newl:', newl)
        return newl, phrase

    @classmethod
    def convert_to_str(cls, x):
        fstr = "{0:" + str(cls.width) + "b}"
        # replace all the (trailing white) spaces
        # with '0'
        return fstr.format(x).replace(' ', '0')

    @classmethod
    def allocate_code(cls, x):
        code = cls.convert_to_str(cls.available)
        cls.mapper[x] = code
        cls.available += 1 # next available id
        return code
    
    @classmethod
    def convert_char(cls, x):
        code = cls.mapper.get(x, None)
        if code is None:
            code = cls.allocate_code(x)
        return code

    # expects either a list of pairs or a dict
    # where keys are the characters, values are
    # code (binary strings).
    @classmethod
    def set_map(cls, bmapper, is_pairs=True):
        if is_pairs: # convert to dict!
            bmapper = dict(bmapper)
        cls.mapper = {}
        for x, code in bmapper.items():
            cls.mapper[x] = code
        print('\n# Num entries in new map is %d' % len(cls.mapper))        
        cls.available = len(cls.mapper) + 1 # next available id
        print('# Available id is:', cls.available)
        print

    @classmethod
    def print_map(cls, pairs_format=True):
        print
        if not pairs_format:
            for x, code in cls.mapper.items():
                print('"%s"\t"%s"' % (x, code))
        else: # in format that can be used set_map()
            # and/or dict .. 
            print("[")
            for x, code in cls.mapper.items():
                print('("%s"\t"%s"),' % (x, code))
            print("]")

    
#####

# for reading newsgroups (nws)
#path_nws = '/Users/omadani/work/data/dir_newsgrps2019/dir_test/'
#fname = path_nws + 'docs_contents_in_order.txt.gz'
#import re
# keep the long lines (read_nws)
def read_newsgrps_output_selected(fname, min_len=15, maxnum=500):
    with gzip.open(fname, 'r') as fp:
        i = 0
        for line in fp:
            line = str(line)
            if not satisfies_nws_criteria(line, min_len=min_len):
                continue
            i += 1
            if i > 500:
                break
            #print(line)
            print(i, 'now:', line)
        # return

def satisfies_nws_criteria(line, min_len=None):
    # print(type(line), line)
    if line.startswith('>'):
        return False
    line = line.strip() # strip space
    line = line.strip('\n') # strip space
    parts = line.split()
    if min_len is not None and len(parts) < min_len:
        return False
    line = line[2:-3] # first few and last few chars are weird
    if line.startswith('>'):
        return False
    line = line.strip() # strip space
    line = line.strip('\n') # strip space
    if not re.match(r"^[a-z]", line):
        return False
    return True

#path_nws = '/Users/omadani/work/data/dir_newsgrps2019/dir_test/'
#fname = path_nws + 'docs_contents_in_order.txt.gz'

# for processing lines from NSF abstract files
# basic processing..
def preprocess_line(line, convert_to_binary=False):
    line = line.strip()
    if line == "":
        return line, True
    # for abstracts files.. these can happen..
    skip = line.startswith("Not Available") or \
        line.startswith("====")
    # If you forget the 't' when opening then you need the bytes(..),
    # as below, and in any case, it looks weird with bytes, so dont
    # forget the 'rt' when opening.
    #
    #skip = line.startswith(bytes("Not Available", 'utf-8')) or \
    #    line.startswith(bytes("====", 'utf-8')) 

    if not skip and convert_to_binary:
        line = BinaryConvert.convert_line(line)
    return line, skip

###

def open_file(fname):
    if 'gz' in fname:
        f = gzip.open(fname, mode='rt') # open in rt=read-text mode
    else:
        f = open(fname, mode='r') 
    return f

###

# do news file(s) or NSF abstracts, etc?
def set_path_and_files(do_nws=0):
    if do_nws:
        print('\n# reading from NEWS groups dataset.')
        #path = '/Users/omadani/work/data/dir_newsgrps2019/dir_test'
        path = '/Users/omidmadani/work/data/dir_newsgrps2019/dir_test'
        files = [ 'docs_contents_in_order.txt.gz' ]
    else:
        print('\n# reading from NSF abstracts.')
        # path = '/Users/omadani/projects/dir_prediction_systems/dir_data_nsf_abstracts'
        path = '/Users/omidmadani/work/data/dir_data_nsf_abstracts'
        files = [ 'abstracts_7.gz',
                  'abstracts_8.gz', 'abstracts_9.gz', 'abstracts_10.gz',
                  'abstracts_11.gz', 'abstracts_1.gz', 'abstracts_2.gz',
                  'abstracts_3.gz',
                  'abstracts_4.gz', 'abstracts_5.gz', 'abstracts_6.gz' ] 
    return path, files

##

def read_nsf_files(
        max_lines_to_read=None, read_sample_rate=0.1, pit=1, rand=None):
    path, files = set_path_and_files(do_nws=0)
    if path is not None:
        files = [ path + '/' + f   for f in files ]
        nfiles = len(files)
    # all_lines = read_process_lines(
    all_lines = read_process_nsf_lines(
        files, nfiles, max_lines=max_lines_to_read,
        sample_rate=read_sample_rate, rand=rand)
    if pit >= 1:
        print('\n# read ', len(all_lines), 'lines' )
    return all_lines

# This is for nsf files too (i think!).. maybe i should change the name..
# 
def read_process_nsf_lines(
        files, nfiles, max_lines=None, sample_rate=None, rand=None):
    nfiles=len(files)
    lines = []
    nf = 0
    if sample_rate is None:
        sample_rate = Expedition.process_rate
    if rand is None:
        rand = random.Random()
    print('\n# num files, max_lines and sample rate: ', nfiles, max_lines, sample_rate)
    while True:
        if nfiles > 1:
            rand.shuffle(files)
        for tfile in files: # text file!
            f = open_file(tfile)
            nf += 1
            try:  # comment out the try if you want to see the exception..
                #if 1:
                # for the utf error (python3 on abstracts data) occurs here at
                # the loop level when
                # a line is read, so we want to handle it here...
                print(('\n# %d. Now reading from file: %s lines_so_far=%d, ' +
                       'sample_rate:%.3f') % (
                    nf, tfile, len(lines), sample_rate))
                for line in f:
                    line, skip = preprocess_line(line)
                    if skip:
                        continue
                    # print(len(line))
                    # if num_included == 0:
                    if sample_rate < 1  and rand.uniform(0, 1.0) > sample_rate:
                        continue
                    lines.append(line)
            except Exception as e: # This catch is the more useful one for utf err..!
                print("\n# EXCEPTION=\"%s\" OCCURRED WHILE PROCESSING A LINE!!"
                      % str(e))
                # comment exception.. if you want to see where errors occurred
            if f is not None:
                f.close()
            if max_lines is not None and len(lines) >= max_lines:
                break
            # Nothing to do..
            if sample_rate >= 1.0 and nfiles == 1:
                break
        print('#   num lines just read: ', len(lines))
        if max_lines is None: # one itr is good.
            break
        if max_lines is not None and len(lines) >= max_lines:
            break
        # Nothing to do..
        if sample_rate >= 1.0 and nfiles == 1:
            break
        
    #rand.shuffle(lines)
    return lines




###

##########

# news groups .. (NOTE: read_nws2 is the preferred way!)
def read_nws1(fname=None, pit=1):
    if fname is None:
        path_nws = '/Users/omadani/work/data/dir_newsgrps2019/dir_test/'
        fname = path_nws + 'docs_contents_in_order.txt.gz'
    lines = []
    f = open_file(fname)
    print(('\n# Reading from file: %s ') % (fname))
    n_exceps = 0
    for line in f:
        try:  # comment out the try if you want to see the exception..
            #if 1:
            # for the utf error (python3 on abstracts data) occurs here at
            # the loop level when
            # a line is read, so we want to handle it here...
            line, skip = preprocess_line(line)
            if skip:
                continue
            # print(len(line))
            lines.append(line)
        except Exception as e: # This catch is the more useful one for utf err..!
            #print("\n# EXCEPTION=\"%s\" OCCURRED WHILE PROCESSING A LINE!!"
            #      % str(e))
            n_exceps += 1
        # comment exception.. if you want to see where errors occurred
    if f is not None:
        f.close()
    if pit:
        print('# num lines read:', len(lines),
              ' num exceptions:', n_exceps)
    return lines

# When 'rt' is used when opening,  270495 lines are read.
# NOTE: .. i think this is the preferred way..!
def read_nws2(fname=None, pit=1, min_len=None, nws_quality=0):
    if fname is None:
        #path_nws = '/Users/omadani/work/data/dir_newsgrps2019/dir_test/'
        #path_nws = '/Users/omidmadani/work/data/dir_newsgrps2019/dir_test/'
        path_nws, files = set_path_and_files(do_nws=1)
        fname = path_nws + '/' + files[0] # 'docs_contents_in_order.txt.gz'

    lines = []
    # perhaps with gzip open, lines are read in a weird way (utf, etc)
    # (starts with b'..) unless we use 't', ie 'rt'...
    with gzip.open(fname, 'rt') as fp:
        for line in fp:
            line, skip = preprocess_line(line)
            if skip:
                continue
            if nws_quality: # check additional quality??
                if not satisfies_nws_criteria(line, min_len=min_len):
                    continue
            line = str(line)
            lines.append(line)
    if pit:
        print('\n# Read %d lines.\n' % len(lines))
    return lines

#read_newsgrps_output_selected(fname)

## for Frontiers, reporting various line stats (I wanted the lines to be
# long enough, so interpretation/segmentation would be more meaningful.

def report_line_stats_from(fname, maxl=None, small=5):
    lens = []
    num_small = 0
    with open(fname, 'r') as fp:
        i = 0
        for line in fp:
            parts = line.split()
            lenl = len(parts)
            lens.append(lenl)
            if lenl < small:
                num_small += 1
            i += 1
            if maxl is not None and i >= maxl:
                break
    print('\n# num lines below %d size is: %d' % (small, num_small))
    print('# med num tokens:%.1f mean:%.1f min:%d max:%d' % (
        np.median(lens), np.mean(lens), np.min(lens), np.max(lens)))

def report_line_stats(lines, maxl=None, small=5):
    lens = []
    num_small = 0
    i = 0
    for line in lines:
        parts = line.split()
        lenl = len(parts)
        lens.append(lenl)
        if lenl < small:
            num_small += 1
        i += 1
        if maxl is not None and i >= maxl:
            break
    print('\n# num lines below %d size is: %d' % (small, num_small))
    print('# med num tokens:%.1f mean:%.1f min:%d max:%d' % (
        np.median(lens), np.mean(lens), np.min(lens), np.max(lens)))          

# keep lines that are long! (white space or have blank spaces)
def drop_short_lines(lines, too_small=5):
    print('\n# orig num lines:', len(lines))
    newlines = []
    for line in lines:
        parts = line.split()
        lenl = len(parts)
        if lenl < too_small:
            continue
        newlines.append(line)
    print('# new num lines:', len(newlines))
    return newlines
     
# By default, order by key, increasing (for span stats of concepts)
# Return results in string format.
def get_count_stat_string(counter):
    cnts=list(counter.items())
    cnts.sort(key=lambda x: x)
    scnt = [] 
    for cnt in cnts:
        scnt.append(str(cnt[0]) + ':' + str(cnt[1]))
    return ', '.join(scnt) #  the string version


###

# What is the performance of picking random splits in lines, where we
# assume a blank space is a good split at the picked position or next
# position (otherwise, when both are other characters, it's a bad
# split) ? (so it assumes we have not removed blank spaces).. (June
# 2023)
def random_split_performance(lines, maxk=100):
    num , bad_split = 0, 0
    while num < maxk:
        for l in lines:
            j = random.randint(0, len(l)-2 )
            bad_split += not l[j].isspace() and not l[j+1].isspace()
            num += 1
    return bad_split / num, num

###

# to_float_str, etc. (returns a string rep, possibly '0.0'), for a
# prob. k is the power of 10 in how low we want to go in the
# representation .. add_more if you want additional significant digits
# to show..
def float_str(p, k=5, add_more=1):
    if p is None:
        return '0.0'
    val, j = 1.0, 1
    for i in range(k):
        j += 1 # j = 1,2,3,..
        val /= 10.0
        if p > val:
            str_rep = '%.' + str(j+ add_more) + 'f'
            return str_rep % p
    return '0.0'

###

def pick_random_letter():
    chars = string.ascii_lowercase # this is 'abcdex...'
    i = random.randint(0, len(chars)-1)
    return chars[i]

def generate_synthesic_lines(nlines):
    lines = []
    for i in range(nlines):
        lines.append(synthesize_line())
    return lines


###

# Add noise to sentence s (or piece of text)
def corrupt(s, p=.05, remove_space=0):
    t = ''  # the new transoformed text
    for i in range(len(s)):
        if random.uniform(0.0, 1.0) > p:
            t += s[i]
            continue
        todo = random.uniform(0.0, 1.0)
        if todo < 0.3:
            continue # skip
        elif todo < 0.6:
            t += pick_random_letter()
        else:
            t += s[i]
            t += pick_random_letter()
    if remove_space:
         t = t.replace(" ", "")
         t = t.replace("\n", "")
    return t

####

# Took this from 'my_wrapper' in slm_nn_ dir

# (for 'Positional loss' or word boundary loss)
# return index into the line, close to middle (but where a word
# ends, or end of the word, or beginning of another, etc. )
# ( at word boundary , split of a line, split of a sentence)
# ( break of a sentence )
def get_line_idx(line,  pos_type=1, rand_obj=None):
    num_chars = len(line)
    if rand_obj is None:
        rand_obj = random
    idx = int(num_chars / 2) # half_way ( middle of a line )
    if pos_type == 3: # pick a random position around middle.
        ntry = 0
        while ntry < 5:
            i = rand_obj.randint(-5, 5)
            if i + idx > 0 and i + idx < num_chars-1:
                return i + idx
            ntry += 1
        return None # not successful

    
    #print('idx, line_len, and line:', idx, num_chars, line)
    # That means predict first charc of next word!
    while idx >= 0 and line[idx] != ' ':
        idx -= 1
    if idx < 0:
        return None
    # the next char is beginning of a word!    
    if pos_type == 1:
        return idx
    # Go before the space(s)?
    while idx >= 0 and line[idx] == ' ':
        idx -= 1
    #print('idx after all the updating:', idx)
    if idx < 0: # No character.. skip..
        return None
    # Now idx is pointing to the last char of a word.
    # predict first letter of next word
    if pos_type <= 2:
        return idx
    idx -= 1 # go one position before (point to ending char of a word!)
    # (so it's the remainder of word, predicting last char!)
    # (this one should be easiest!)
    if idx < 0 or line[idx] == ' ':
        return None
    return idx


#######

# Oct 2023
# Given observations, immediately following a predictor
# (during Expedition training) for a few
# selected predictors, dump them to a file..
# (dump the obs sequences)
def dump_seqs( obs_lists, seed=None, fname='seqs_dump.txt' ):
    obs_to_list = defaultdict(list)
    obs_to_str = {}
    for eps, tup_map in obs_lists:
        for obs, obs_info in tup_map.items(): # tup is a map
            # This will in effect keep the last time
            # the obs was seen in the info map, to be later dumped.
            obs_to_str[obs] = obs_info['concept']
            after = obs_info['after'] # list of what it coocurs with
            for coocs in after:
                cid = coocs[0]
                obs_to_list[obs].append((eps, cid))
    cs = [ (x[0], len(x[1])) for x in obs_to_list.items()]
    cs.sort(key=lambda x: x[1])
    with open(fname, "a") as f:
        # Write to the text file
        #f.write("This is a test message.")
        if seed is not None:
            f.write("\n# seed was: %s\n" % str(seed) )
        for obs, size in cs:
            obs_str = obs_to_str[obs]
            f.write("\n# predictor='%s', cid=%d, seq_len=%d\n" % (obs_str, obs, size))
            seq = obs_to_list[obs]
            seq_str = ''
            for tup in seq:
                seq_str += '%d:%d,' % (tup[0], tup[1]) 
            seq_str += '\n'
            f.write(seq_str)

# Oct 2023 (read from files dumped above!)
#
# These observation sequences where collected during
# Expedition training. See README in dir_expedition_seqs/
def read_seqs_expedition(fname=None, min_len=None, max_len=None, pit=1, sort_it=1):
    if fname is None:
        fdir = '/Users/omadani/work/data/dir_expedition_seqs/'
        fdir += 'dir_500thrsh/' # which subdir
        #fname = fdir +  'predictor_sequence_dumps.5k_episodes.txt'
        #fname = fdir +  'predictor_sequence_dumps.20k_episodes.1.txt'
        #fname = fdir +  'predictor_sequence_dumps.20k_episodes.2.txt'
        #fname = fdir +  'predictor_sequence_dumps.25k_episodes.1.txt'
        #fname = fdir +  'predictor_sequence_dumps.30k_episodes.1.txt'
        #fname = fdir +  'predictor_sequence_dumps.30k_episodes.2.txt'
        fname = fdir + 'predictor_sequence_dumps.combined_30k_etc.txt'
    seqs = []
    with open(fname, "r") as fp:
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

def print_seq_info(seqs, more_info=1):
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
    
###
            
# reading greenebrg, etc.

# encodings: 'cp852' , 'iso8859_13', 'cp1251', 'cp737', .. 
# 'iso8859_16' , 'cp865'    seem to all work..

def read_greenberg( f, stubs=1 ):
    # encodings = all_encodings()
    encodings = ['cp852', 'iso8859_13' , 'cp865']
    for enc in encodings:
        try:
            k = 0
            seq = []
            with open(f, encoding=enc) as f:
                for l in f:
                    l = l.strip()
                    #print(l)
                    if not l.startswith('C '):
                        continue
                    if stubs:
                        l = l.split()
                        l = l[1]
                    seq.append(l)
                    k += 1
            #print('\n# encoding ', enc, ' worked.. ')
            #print('# num commands read (seq size):', k)
        except Exception:
            print(' .. some exception .. file was:', f)
            pass # continue
        break
    return seq

# read seqs from greenberg directory
def read_seqs_greenberg( stubs=1, sort_it=1, pit=1, min_len=None,
                         max_len=None, stop_at_half=None ):
    mypath = '/Users/omadani/work/data/dir_unix_commands/' + \
        'dir_unix_data_saul_greenberg/computer-scientists/'
    special =  '-'
    files = [f  for f in listdir(mypath) if \
             isfile(join(mypath, f)) and special in f ]
    seqs = []
    for f in files:
        f = join(mypath, f)
        seq = read_greenberg( f, stubs=stubs )
        lsq = len(seq)
        if stop_at_half:
            seq = seq[: int(lsq / 2)] # return half the seq (eg for non-stationary experiments)
        if min_len is not None and lsq < min_len:
            continue
        if max_len is not None and lsq > max_len:
            continue
        seqs.append(seq)
    if pit:
        print_seq_info(seqs)
    if sort_it: # sort by increasing len
        seqs.sort( key=lambda x: len(x)  )
    return seqs

####

# import random

# given a sequence, compute freq stats, and if above minc,
# get number of such symbols in the seq
def get_frq_stats(seq, minc=2):
    frqs = {} #Counter()
    for s in seq:
        c = frqs.get(s, 0)
        frqs[s] = c + 1
    tot = 0
    for s, c in frqs.items():
        if c >= minc:
            tot += 1
    return tot

# Get sequences whose num symbols with frq above minc is not many, below maxf
# (for non-stationarity experiments)
def get_small_seqs(seqs, minc, maxf):
    pairs = []
    print('\n# orig num seqs:', len(seqs))
    for seq in seqs:
        numf = get_frq_stats(seq, minc)
        if numf > maxf:
            continue
        pairs.append((seq, numf))
    return pairs

# Pick a sequence and plot max-rate of dyal.
def pick_and_plot(seqs, i = 0, k=1, xlabel='', minc=None, maxf=None):
    if minc is not None:
        pairs = get_small_seqs(seqs, minc, maxf)
        print('\n# num seqs with minc:%d, maxf:%d is %d ' % (minc, maxf, len(pairs)))
        i = 0
        for nf in pairs:
            print('# nf:', i, ' num symbs. above thrsh:', nf[1], 'seq len:', len(nf[0]))
            i += 1
        i = 0
        random.shuffle(pairs)
        seqs = [pairs[0][0]]
        print('# selected with nf:', pairs[0][1])
        
    seq = [] #seqs[i]
    #if k > 1:
    print('# len is at first:', len(seqs[i]))
    for j in range(k):
        seq += seqs[i]
    print('# len is now:', len(seq))
    maxlrs = predict_next.explore_lr_changes([seq], j = 0)
    plt.plot( maxlrs, label = 'max rate', color='lightgreen')
    plt.xlabel(xlabel)
    plt.ylabel('maximum rate')

    plt.legend()
    plt.show()

#####

# Masqurade
def read_masq_file(datafile, n=5000, pit=1 ):
    f = open(datafile, 'r')
    seq = []
    k = 0
    for l in f:
        l = l.strip()
        #print(l)
        if l != '':
            seq.append(l)
            k += 1
            if k >= n:
                break
    return seq

def read_masq_seqs(n=5000, pit=1, maxn=None , upto=None ):
    #mypath = '/Users/omadani/work/data/dir_unix_commands/' + \
    #    'dir_unix_data_saul_greenberg/computer-scientists/'
    mypath = '/Users/omadani/work/data/dir_unix_commands/' + \
            'dir_masqurade/dir_masquerade_data/'
    special =  'User'
    files = [f  for f in listdir(mypath) if \
             isfile(join(mypath, f)) and special in f ]
    seqs = []
    for f in files:
        f = join(mypath, f)
        seq = read_masq_file( f, n=n )
        if upto is not None:
            seq = seq[:upto]
        lsq = len(seq)
        seqs.append(seq)
    if maxn is not None:
        seqs = seqs[:maxn]
    if pit:
        print_seq_info(seqs)
        #if sort_it: # sort by increasing len
        #seqs.sort( key=lambda x: len(x)  )
    return seqs

#######


############# synthetic line generation


def make_n_digits(num):
    l = ''
    for i in range(num):
        l += str(random.randint(0, 9))
    return l

# def make_area_code(width=3, fixed=False, ob='a', cb='b'):
def make_area_code(width=3, fixed=False, ob='[', cb=']'):
    l = make_n_digits(width)
    if fixed: # always same fixed pattern?
        return  ob + l + cb # ob is open bracket, and cb is close bracket
    if random.uniform(0.0, 1.0) < 0.5:
        return l
    else:
        return ob + l + cb
    #else:
    #    l = l + ' '
    #return l

def make_prefix_phone(add_begin=0, add_end=0, num=4):
    assert num <= 4
    i = random.randint(0, num)
    s = ''
    e = ''
    if add_begin:
        s = 'begin '
    if add_end:
        e = ' end'
    #i = 0
    if i == 0:
        return s + 'tel: ' + e
    if i == 1:
        return s + 'cell: ' + e
    if i == 2:
        return s + 'mobile: ' + e
    if i == 3:
        return s + 'telephone: ' + e
    if i == 4:
        return s + 'phone: ' + e
    return s + 'phone: ' + e

def make_phone():
    return make_n_digits(3) + '-' + make_n_digits(4)

def simple_line():
    # return 'ab'
    return '99' + make_n_digits(1)
    #return make_n_digits(1)
    #return 'abcd' #  'ab' + line
    line = 'a'
    if random.uniform(0, 1.0) < 0.5:
        line = 'b'
    #line = 'abcd' # + line
    return line

def simple_grp1():
    if random.uniform(0, 1.0) < 0.35:
        return 'abc'
    else:
        return 'aed'
        #return 'ace'
        

# So every char is 'ambiguous', ie has multiple branches to its left
# and right (as long as we have begin and end buffer predictors
# too). Then before 'a', either begin-buffer occurs or 'c' occurs, and
# after it, either 'b' or 'c' or end-buffer occurs.. Similar for 'b',
# etc.
def simple_grp2():
    p = random.uniform(0, 1.0) 
    if p < 0.25:
        return 'abc'
    elif p < 0.5:
        return 'ace'
    elif p < 0.75:
        return 'bed'
    else:
        return 'bca'
        
def simple_grp3():
    if random.uniform(0, 1.0) < 0.3:
        return 'aaabbb'
    else:
        return 'bbbaaa'



def simple_compose():
    #return 'ab'
    # return 'abcdde'
    return 'aaaaab'
    # return 'abcde'
    #return 'aabbbaa'
    #return 'ababc'
    #return 'ababab'

gchars = '0123456789' + string.ascii_lowercase # this is 'abcdex...'

# to get many possible different single char symbols (~70)
many_chars = '0123456789' + string.ascii_lowercase + \
    string.ascii_uppercase # this is 'abcdex...'

def simple_grp(probs=[0.5]):
    global gchars
    sump = 0
    s = gchars[0]
    i = 0
    q = random.uniform(0, 1.0)
    #print probs
    for p in probs:
        sump += p
        if q < sump:
            break
        i += 1
        s = gchars[i]
    return s

# width of 2 means to possibilities, etc.
def make_grps(k=1, width=2):
    line = ''
    #probs = [0.5]
    #probs = [.3, .3]
    p = 1.0 / width
    probs = [ p for x in range(width) ]
    for i in range(k):
        line += simple_grp(probs)
    return line

def make_grp_of_composes(p=0.25):
    s = 'abc'
    if random.uniform(0, 1.0) < p:
        s = 'cde'
    return s

##

def make_full_phone(fixed=False):
    line = make_prefix_phone()
    line += make_area_code(fixed=fixed, ob='[', cb=']' )
    line += ' '
    line += make_phone()
    return line

###


# For now, phone numbers! >synthetic (July 2022, other varietries will
# be/is added!)
#
# ct is concept tree..
#
def synthesize_line(ct_to_gen=None):

    # Some CTree from which to generate..
    if ct_to_gen is not None:
        return CTree.generate(ct_to_gen)
    
    #return simple_compose()
    #return simple_grp1()
    #return simple_grp2()
    #return simple_grp3()
    
    # return make_prefix_phone(add_begin=0, add_end=0, num=4)

    #return make_area_code( width=3, fixed=True,  ob='[', cb=']' )
    
    # return make_area_code( width=5 , fixed=True,  ob='', cb='' )

    #return 'ef' + make_area_code(width=2, fixed=True) + 'ab'

    # return make_prefix_phone(num=4)

    # return make_prefix_phone(num=4) + make_area_code(width=1, fixed=True)
    #return make_prefix_phone(num=4) + make_area_code(width=3, fixed=True)

    return make_full_phone(fixed=True)
    
    #return make_grps(width=10, k=3)

    # return '(' + make_grps(3) + ')' # 4)
    # return '(' + make_grps(2) 
    # return '(' + make_grps(3) + ')'
    # return '(' + make_grps(4)
    
    return 'tel:' + make_area_code( width=3, fixed=True )
    # return make_area_code()
    # return 'phone:' 
    # return 'phone:' + make_area_code()
    #return make_grp_of_composes()

    # return make_full_phone()

    ct = None
    if 0:
        ct = make_prefix_phone_and_area_ct()

    if ct is not None:
        CTree.generate(ct)
    
    return make_prefix_phone()
    # return make_prefix_phone() + make_area_code()

    # return make_n_digits(1)

    # return line

#### end of synthetic gen






