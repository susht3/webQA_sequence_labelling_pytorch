#--*-- encoding:utf-8 --*--
import sys
from collections import Counter
from fuzzy_matching import FuzzyMatcher

def get_tagging_results(tokens, tags):
    '''
    @param tokens: a list of unicode strings
    @param tags: a list of B/I/O1/O2
    '''
    chunks = []
    start = -1
    for i, tok in enumerate(tokens):
        tag = tags[i]
        if tag == 'B':
            if start >= 0: chunks.append(' '.join(tokens[start:i]))
            start = i
        elif tag == 'I':
            if start < 0: start = i
        else:
            if start < 0: continue
            chunks.append(' '.join(tokens[start:i]))
            start = -1
    if start >= 0:
        chunks.append(' '.join(tokens[start:]))

    return chunks

__fuzzy_matcher = None
def is_right(upred_ans, ugolden_ans, fuzzy):
    '''
    @param upred_ans: predicted answer, unicode
    @param ugolden_ans: golden answer, unicode
    @param fuzzy: use fuzzy match if True
    '''
    global __fuzzy_matcher
    if upred_ans.lower() == u'no_result' and ugolden_ans == u'no_answer':
        return True
    if fuzzy:
        if not __fuzzy_matcher: __fuzzy_matcher = FuzzyMatcher()
        return __fuzzy_matcher.is_synonym(ugolden_ans, upred_ans)
    else:
        upred_ans = upred_ans.replace(' ', '').lower()
        ugolden_ans = ugolden_ans.replace(' ', '').lower()
        return upred_ans == ugolden_ans
