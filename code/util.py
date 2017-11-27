import json
import jieba
import pickle
import csv, h5py
import pandas as pd
import numpy as np
from tqdm import *
import torch
from torch import Tensor
from torch.autograd import Variable
import torch.utils.data as data

STOP_TAG = "#OOV#"  

class Hyperparameters:
    tagset_size = 4
    answer_size = 16
    question_size = 64
    evidence_size = 512

    webQA_train_path = '../data/training.json'
    webQA_test_ann_path = '../data/test.ann.json'
    webQA_test_ir_path = '../data/test.ir.json'
    webQA_val_ann_path = '../data/validation.ann.json'
    webQA_val_ir_path = '../data/validation.ir.json'
    
    train_path = '../data/training.h5'
    test_ann_path = '../data/test.ann.h5'
    test_ir_path = '../data/test.ir.h5'
    val_ann_path = '../data/validation.ann.h5'
    val_ir_path = '../data/validation.ir.h5'
    vocab_path = '../data/vocabulary.txt'

    
param =  Hyperparameters()   


def load_words(path, ret):
    with open(path) as f:
        for line in tqdm(f):
            txt = json.loads(line)
            question_tokens = ''.join(txt['question_tokens'])
            evidences = txt['evidences']
            for e in evidences:
                evidence_tokens = ''.join(e['evidence_tokens'])
                for i in evidence_tokens:
                    ret.add(i)   
                golden_answers = ''.join(e['golden_answers'][0])
                if golden_answers == 'no_answer':
                    ret.add(golden_answers)
                else:
                    for a in golden_answers:
                        ret.add(a)
            for q in question_tokens:
                ret.add(q)    
    return ret 


def get_vocab(paths):
    print('Getting vacabulary...')
    ret = set()
    for p in paths:
        ret = load_words(p, ret)
    
    ret = sorted(list(ret))
    input_set = [STOP_TAG]
    input_set.extend(list(ret))
    input_set_size = len(input_set)
    input2idx = dict(zip(input_set, range(input_set_size)))
    print('Vacabulary size:', input_set_size, '\n')
    return input_set, input2idx, input_set_size


def save_vocab(path, input2idx):
    print('Saving bocabulary...')
    f = open(path,'wb')
    pickle.dump(input2idx, f)
    f.close()


def load_vocab(path):
    print('Loading vocabulary...')
    f = open(path, 'rb')
    input2idx = pickle.load(f)
    input_set = list(input2idx.keys())
    input_set_size = len(input_set)
    f.close()
    print('Vacabulary size:', input_set_size, '\n')
    return input_set, input2idx, input_set_size
    
    
def load_webQA_vocab():
    print('Loading webQA vacabulary...')
    input_set = []
    with open(param.pre_vocab_path) as f:
        for word in tqdm(f):
            word = word.strip('\n')
            input_set.append(word)
    input_set_size = len(input_set)
    input2idx = dict(zip(input_set, range(input_set_size)))
    print('Vacabulary size:', input_set_size, '\n')
    return input_set, input2idx, input_set_size


def load_webQA_embedding():
    print('Loading webQA word embedding...')
    matrix = []
    i = 0
    with open(param.pre_embedding_path) as f:
        for embed in tqdm(f):
            embed = embed.strip('\n').split(',')
            embed = [ float(e) for e in embed ]
            embed = np.array(embed)
            matrix.append(embed)

    matrix = np.array(matrix)
    print('embedding size: ', matrix.shape)
    return matrix


# ------------------ save the file --------------------------- #    

def load_chars(seq, input2idx):
    seq = ''.join(seq)
    vector = [ input2idx[s] for s in seq if s in input2idx]
    return vector, len(vector)
    

def load_evidence_and_feats(evidence, q_feats, e_feats, input2idx):
    evidence_vector = []
    q_vector = []
    e_vector = []
    for evid, qf, ef in zip(evidence, q_feats, e_feats):
        for e in evid:
            if e in input2idx:
                evidence_vector.append(input2idx[e])
                q_vector.append(qf)   
                e_vector.append(ef)
    return evidence_vector, q_vector, e_vector
    
    
def load_tags(evidence, tags, word2idx, tag2idx):
    tags_vector = []
    for evid, t in zip(evidence, tags):
        first = True
        for e in evid:
            if e not in word2idx:
                continue
            
            if t == 'b' :
                if first == True:
                    first = False
                    tags_vector.append(tag2idx['b']) 
                else:
                    tags_vector.append(tag2idx['i']) 
            else:
                tags_vector.append(tag2idx[t])
                
    return tags_vector
    

def pad_sequence(seq, seq_size, input2idx):
    vector = []
    for i in range(seq_size):
        if i >= len(seq):
            vector.append(input2idx[STOP_TAG])
        else:
            vector.append(seq[i])
    mask = Tensor(vector).le(0).tolist()
    return vector, mask


def compare(labels, golden_labels):
    for l, g in zip(labels, golden_labels):
        if l != g:
            return False
    return True

def save_h5py_file(old_path, new_path, word2idx, idx2word, tag2idx):
    print('Saving (', new_path, ')...')
    file = h5py.File(new_path,'w')
    question_list = []
    evidence_list = []
    labels_list = []
    q_list = []
    e_list = []
    pos_list = []
    ner_list = []
    q_mask_list = []
    e_mask_list = []
    answer_list = []

    questions_size = 0
    evidences_size = 0
    active_evidences_size = 0
    negetive_evidences_size = 0
    has_labels = True
    with open(old_path) as f:
        for line in tqdm(f):
            txt = json.loads(line)
            question_tokens = txt['question_tokens']
            question, q_length = load_chars(question_tokens, word2idx)
            question, q_mask = pad_sequence(question, param.question_size, word2idx)
            
            no_ans = 0
            evidences = txt['evidences']
            for e in evidences:
                golden_answers = e['golden_answers'][0]
                evidence_tokens = e['evidence_tokens']
                q_feats = e['q-e.comm_features']
                e_feats = e['eecom_features_list'][0]['e-e.comm_features']

                # new ir
                ans_str = ''.join(golden_answers)
                evid_str = ''.join(evidence_tokens)
                if question_tokens != 'no_answer' and ans_str not in evid_str:
                    continue
               
                
                if len(evidence_tokens) > param.evidence_size:
                    continue
                    
                if  ''.join(golden_answers) == 'no_answer':
                    no_ans += 1
                    negetive_evidences_size += 1
                else :
                    active_evidences_size += 1
                
                if no_ans > 8:
                    continue   
                    
                
                evidence, q_feats , e_feats = load_evidence_and_feats(evidence_tokens, q_feats, e_feats, word2idx)
                evidence, e_mask = pad_sequence(evidence, param.evidence_size, word2idx)
                q_tags, _ = pad_sequence(q_feats, param.evidence_size, word2idx)
                e_tags, _ = pad_sequence(e_feats, param.evidence_size, word2idx)
                    
                golden_answers, _ = load_chars(golden_answers, word2idx)
                answer, _ = pad_sequence(golden_answers, param.answer_size, word2idx)
                
                if 'golden_labels' in e:
                    golden_labels = e['golden_labels']
                    labels = load_tags(evidence_tokens, golden_labels, word2idx, tag2idx)
                    labels, _ = pad_sequence(labels, param.evidence_size, tag2idx)
                    labels_list.append(labels)
                else:
                    has_labels = False
         
                
                question_list.append(question)
                evidence_list.append(evidence)
                q_list.append(q_tags)
                e_list.append(e_tags)
                q_mask_list.append(q_mask)
                e_mask_list.append(e_mask)
                answer_list.append(answer)
                
                evidences_size += 1
            questions_size += 1
            
    file.create_dataset('question', data = np.asarray(question_list))
    file.create_dataset('evidence', data = np.asarray(evidence_list))
    file.create_dataset('answer', data = np.asarray(answer_list))
    file.create_dataset('q_mask', data = np.asarray(q_mask_list))
    file.create_dataset('e_mask', data = np.asarray(e_mask_list))
    file.create_dataset('q_feat', data = np.asarray(q_list))
    file.create_dataset('e_feat', data = np.asarray(e_list))
    
    if has_labels == True:
        file.create_dataset('labels', data = np.asarray(labels_list))

    file.close()
    print('questions size: ', questions_size)
    print('evidences size: ', evidences_size, '\n')


def save_files(old_paths, new_paths, word2idx, idx2word, tag2idx):
    print('Saving h5py files...')
    for i in range(len(old_paths)):
        save_h5py_file(old_paths[i], new_paths[i], word2idx, idx2word, tag2idx)
    print('Files Saved')

    

if __name__ == '__main__':

    webQA_paths = [param.webQA_train_path, param.webQA_test_ann_path, 
                   param.webQA_test_ir_path, param.webQA_val_ann_path, param.webQA_val_ir_path]
    
    h5py_paths = [param.train_path, param.test_ann_path, param.test_ir_path, param.val_ann_path, param.val_ir_path]
    
    #h5py_paths = [param.pre_train_path, param.pre_test_ann_path, param.pre_test_ir_path, param.pre_val_ann_path, param.pre_val_ir_path]
    
    #word_set, word2idx, word_set_size = get_vocab(webQA_paths)
    #save_vocab(param.vocab_path, word2idx)
    
    word_set, word2idx, word_set_size = load_vocab(param.vocab_path)
    #word_set, word2idx, word_set_size = load_webQA_vocab()
    
    idx2word = dict(zip(word2idx.values(), word2idx.keys()))
    tag2idx = { "b": 0, "i": 1, "o1": 2, "o2":3, STOP_TAG: 4}
    
    save_files(webQA_paths, h5py_paths, word2idx, idx2word, tag2idx)

    
    #new_test_ir_path = '../char_data/new_test.ir.h5'
    #save_h5py_file(param.webQA_test_ir_path, new_test_ir_path, word2idx, idx2word, tag2idx)
    
    #save_test_file(param.webQA_val_ir_path, param.lonely_test_ann_path, word2idx, idx2word, tag2idx)
    
    
    
   
    
