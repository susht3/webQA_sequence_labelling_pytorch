import re

def build_greedy_regex_str(words):
    words = sorted(words, key=lambda w: len(w), reverse=True)
    return '|'.join(words)
