#--*-- encoding:utf-8 --*--
import re
import sys
from state_names import state_names
from names import ethnic_group_names
from re_util import build_greedy_regex_str
from collections import defaultdict

__all__ = ['StateMatcher']

# TODO: smaller state, city etc. names
# TODO: foreign state and city names
# TODO: names such as 新疆维吾尔自治区
# TODO: cases such as "云南 云南"

DEBUG = False

class StateMatcher(object):

    STATE_SUFFIXES = (u'省', u'市', u'县', u'旗', u'区', u'盟', u'州', 
                      u'自治省', u'自治市', u'自治县', u'自治旗', u'自治区', 
                      u'自治盟', u'自治州', u'特别行政区') 
    PATTERN_SPACE = re.compile(r'\s+', re.UNICODE)

    def __init__(self):
        regex = build_greedy_regex_str(ethnic_group_names)
        self.PATTERN_ETHNIC_GROUPS = re.compile(u'^(.+)(%s)$' % regex)

        self.state_ids = defaultdict(lambda: set())
        self.full_name_ids = {}
        for id_, name in state_names:
            full_name = name
            name = self.get_simple_state_name(name)
            if len(name) == 0: continue 
            self.state_ids[name].add(id_)
            self.full_name_ids[full_name] = id_
        self.state_ids.default_factory = None

        # match all simple state names
        regex = build_greedy_regex_str(self.state_ids.keys())
        self.PATTERN_STATE = re.compile(regex)

        names = self.state_ids.keys() + self.full_name_ids.keys()
        regex = build_greedy_regex_str(names)
        self.PATTERN_COMPLEX_STATE = re.compile(u'^(%s)+$' % regex)

    def get_simple_state_name(self, name):
        '''name: a single state name'''
        suffix = name[-1].strip()
        if suffix in u'省市县旗区盟州':
            name = name[0:-1].strip()
        if name.endswith(u'自治'):
            name = name[0:-2].strip()
        if name.endswith(u'特别行政'):
            name = name[0:-4].strip()
        m = self.PATTERN_ETHNIC_GROUPS.match(name)
        if m: name = m.group(1)
        suffixes = [u'维吾尔', u'哈萨克', u'蒙古', u'柯尔克孜', 
                    u'塔吉克', u'锡伯',]
        for suffix in suffixes:
            if name.endswith(suffix):
                name = name[0:-len(suffix)]
        return name 

    def get_all_state_names(self, text):
        return self.PATTERN_STATE.findall(self.__norm(text))

    def __key(self, name):
        name = self.__norm(name)
        if name in self.state_ids:
            return name
        return self.get_simple_state_name(name)

    def __norm(self, text):
        if not isinstance(text, unicode): text = text.decode('utf-8')
        return self.PATTERN_SPACE.sub('', text)

    def is_in(self, small_state, big_state):
        ids1 = self.state_ids.get(self.__key(small_state))
        ids2 = self.state_ids.get(self.__key(big_state))

        if not ids1 or not ids2:
            return False
        ids1 = [id_.rstrip('0') for id_ in ids1] 
        ids2 = [id_.rstrip('0') for id_ in ids2] 
        for id1 in ids1:
            for id2 in ids2:
                if id1 == id2: continue
                if id1.startswith(id2):
                    return True
        return False
    
    def is_same_state(self, state1, state2):
        state1, state2 = self.__norm(state1), self.__norm(state2)
        if len(state1) < len(state2):
            sstate, lstate = state1, state2
        else:
            sstate, lstate = state2, state1
        if not lstate.startswith(sstate):
            return False
        if sstate not in self.state_ids:
            return False
        if lstate not in self.full_name_ids:
            return False
        if self.__key(lstate) not in self.state_ids:
            return False
        return True

    def is_same_complex_state(self, std_text, other_text):
        if not self.maybe_complex_state_name(std_text): return False
        if not self.maybe_complex_state_name(other_text): return False

        std_text, other_text = self.__norm(std_text), self.__norm(other_text)
        # break into state names
        std_sub_names = self.get_all_state_names(std_text)
        std_text = ''.join(std_sub_names)

        other_sub_names = self.get_all_state_names(other_text)
        other_text = ''.join(other_sub_names)

        if DEBUG: print >> sys.stderr, 'std_text:', std_text.encode('utf-8')
        if DEBUG: print >> sys.stderr, 'other_text:', other_text.encode('utf-8')

        if not other_text.endswith(std_text) and \
                not std_text.endswith(other_text):
            return False

        # verify every part is a state name
        for sub_names in [std_sub_names, other_sub_names]:
            if DEBUG: print >> sys.stderr, 'check is state name: ',
            if DEBUG: print >> sys.stderr, ' '.join(sub_names).encode('utf-8')
            for sub_name in sub_names:
                if not self.is_state_name(sub_name):
                    return False
                if DEBUG: print >> sys.stderr, '\t', sub_name, '\tpass'
            if DEBUG: print >> sys.stderr, '\tevery part is a state name'

        if len(std_sub_names) < len(other_sub_names):
            sub_names = other_sub_names
        else:
            sub_names = std_sub_names

        # verify order
        if DEBUG: print >> sys.stderr, 'verify order:'
        if DEBUG: print >> sys.stderr, ' '.join(sub_names).encode('utf-8')
        for i in range(1, len(sub_names)):
            if not self.is_in(sub_names[i], sub_names[0]):
                return False
            if DEBUG: print >> sys.stderr, '\t', sub_names[i], '|', sub_names[0], '\tpass'
        if DEBUG: print >> sys.stderr, '\torder ok'

        return True

    def is_state_name(self, name):
        return self.__key(name) in self.state_ids
    
    def maybe_complex_state_name(self, name):
        return bool(self.PATTERN_COMPLEX_STATE.match(self.__norm(name)))
