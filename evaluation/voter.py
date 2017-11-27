import sys
import re
import json
from collections import defaultdict
from tagging_evaluation_util import get_tagging_results
from raw_result_parser import iter_results

__all__ = ['iter_voting_results', 'Voter']

class Voter(object):
    __PATTERN_SPACE = re.compile(ur'\s+', re.UNICODE)

    def __init__(self, q_tokens, golden_answers):
        self.first = None
        self.q_tokens = q_tokens
        self.results = defaultdict(int)
        self.golden_answers = golden_answers
        self.update_time = 0

    def norm(self, text):
        if not isinstance(text, unicode): text = text.decode('utf-8')
        text = self.__PATTERN_SPACE.sub('', text)
        return text

    def update(self, pred_answers, golden_answers, *others, **other_maps):
        if self.golden_answers is None:
            self.golden_answers = golden_answers
        elif len(self.golden_answers) == 1 \
                and self.golden_answers[0].lower() == u'no_answer':
            self.golden_answers = golden_answers
        else:
            assert self.golden_answers == golden_answers

        if not self.first:
            self.first = defaultdict(int)
            for pred in pred_answers:
                self.first[self.norm(pred)] += 1
        for pred in pred_answers:
            pred = self.norm(pred)
            self.results[pred] += 1
        self.update_time += 1

    def get_update_time(self):
        return self.update_time

    def transfer(self, d):
        pred_answers, freqs = [], []
        for pred_answer, freq in d.iteritems():
            pred_answers.append(pred_answer)
            freqs.append(freq)
        return self.q_tokens, self.golden_answers, pred_answers, freqs

    def vote(self):
        if len(self.results) == 0:
            return self.transfer({})

        max_freq = max(self.results.values())
        if max_freq == 1:
            return self.transfer(self.first)
        else:
            results = {}
            for key, value in self.results.iteritems():
                if value == max_freq:
                    results[key] = value
            return self.transfer(results)

def iter_voting_results(raw_prediction_file, test_file, schema):
    voter = None
    for q_tokens, e_tokens, tags, golden_answers in\
            iter_results(raw_prediction_file, test_file, schema):
        if not voter: voter = Voter(q_tokens, golden_answers) 
        if q_tokens is None:
            # one question has been processed
            yield voter.vote()
            voter = None
        else:
            pred_answers = get_tagging_results(e_tokens, tags)
            voter.update(pred_answers, golden_answers)

    if voter and voter.get_update_time():
        yield voter.vote()
