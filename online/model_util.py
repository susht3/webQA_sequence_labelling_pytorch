import sys
sys.path.append('../code')

import h5py
import random
import torch
import numpy as np
from util import load_vocab
from online_util import get_inputs, get_answers, get_tuple_answers
from baiduSpider import get_evidences

STOP_TAG = "#OOV#" 

class Hyperparameters:
    vocab_path = '../char_data/vocabulary.txt'
    random_path = '../char_data/training.h5'
    #charQA_path = '../model/charQA_2017-08-11/f1-0.5583_0.34799_2'
    charQA_path = '../model/lossQA_2017-08-14/f1-0.5698_0.26918_5'

param = Hyperparameters()
word_set, word2idx, word_set_size = load_vocab(param.vocab_path)
idx2word = dict(zip(word2idx.values(), word2idx.keys()))


def random_sample():
    file = h5py.File(param.random_path)
    nb_samples = len(file['question'][:])
    
    index = random.randint(0, nb_samples-1)
    question = file['question'][index]
    question = ''.join([ idx2word[q] for q in question if q != 0 ])
    return question
    

class baselineQA(object):
    def __init__(self):
        self.word_set, self.word2idx, self.word_set_size = word_set, word2idx, word_set_size
        self.idx2word = idx2word
        self.model = torch.load(param.charQA_path)

    def pred(self, question, pages = 20):
        evidences = get_evidences(question, pages) 
        question, evidence, q_mask, e_mask, q_feat, e_feat = get_inputs(question, evidences, self.word2idx)
        #answer = get_answers(self.model, question, evidence, q_mask, e_mask, q_feat, e_feat, self.idx2word)
        answer, ans2evid = get_tuple_answers(self.model, question, evidence, q_mask, e_mask, q_feat, e_feat, self.idx2word)
        return answer, ans2evid


if __name__ == '__main__':
    question = '我的前半生中，靳东演的是谁'
    model = baselineQA()
    p = model.pred(question)
    
    