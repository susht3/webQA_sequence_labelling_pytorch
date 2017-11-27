import torch
import torch.nn as nn
from torch.autograd import Variable
import os
import jieba
import time
from datetime import datetime
from collections import Counter

STOP_TAG = '#OOV#'

def fuzzy_match(preds, ans):
    if preds == 'no_answer':
        return 0
    if preds in ans or ans in preds:
        return 1
    return 0

def fuzzy_match_list(preds, ans):
    for p in preds:
        if p in ans or ans in p:
            return 1
    return 0

def exact_match(preds, ans):
    if preds == 'no_answer':
        return 0
    if preds == ans:
        return 1
    return 0

def clean_answer(text):
    std_text = text.replace(' ', '')
    std_text = std_text.lstrip(u'"').lstrip(u'“').lstrip(u"'")\
                       .lstrip(u"‘").lstrip(u'<').lstrip(u'《')\
                       .lstrip(u'【')
    std_text = std_text.rstrip(u'"').rstrip(u'”').rstrip(u"'")\
                       .rstrip(u"’").rstrip(u'>').rstrip(u'》')\
                       .rstrip(u"】")
    return std_text

def get_corrected_results(tokens, tags):
    char2word = dict()
    words = list(jieba.cut(''.join(tokens)))
    c = 0
    for i, w in enumerate(words):
        for ww in w:
            char2word[c] = i
            c += 1
            
    chunks =  []
    start = -1
    
    for i, tok in enumerate(tokens):
        tag = tags[i]
        if tag == 0:  # B
            if start >= 0: chunks.append([start, i])
            start = i
        elif tag == 1:  # I
            if start < 0: start = i
        else:
            if start < 0: continue
            chunks.append([start, i])
            start = -1
    if start >= 0:
        chunks.append([start, len(tokens)-1])
    
    answers = set()
    for c in chunks:
        ans = []
        for i in range(c[0], c[-1]):
            w = words[char2word[i]]
            if w not in ans: ans.append(w)
        ans = clean_answer(''.join(ans))
        if ans != '': answers.add(ans)    
    
    if len(answers) == 0:
        answers.add('no_answer')
    return list(answers)


def get_tagging_results(tokens, tags):
    chunks = set()
    start = -1
    for i, tok in enumerate(tokens):
        tag = tags[i]
        if tag == 0:  # B
            if start >= 0: chunks.add(''.join(tokens[start:i]))
            start = i
        elif tag == 1:  # I
            if start < 0: start = i
        else:
            if start < 0: continue
            chunks.add(''.join(tokens[start:i]))
            start = -1
    if start >= 0:
        chunks.add(''.join(tokens[start:]))
        
    if len(chunks) == 0:
        chunks.add('no_answer')
    return list(chunks)


def get_batch_scores(pred_scores, pred_tags, answer, question, evidence, idx2word):
    nb_pred = 0
    A, C, Q = 0, 0, 0
    question = question.data.cpu().numpy()
    evidence = evidence.data.cpu().numpy()
    for score, pred, ans , ques, evid in zip(pred_scores, pred_tags, answer, question, evidence):
        ques = [ idx2word[q] for q in ques if q != 0 ]
        evid = [ idx2word[e] for e in evid if e != 0 ]
        ans = clean_answer(''.join( [ idx2word[a] for a in ans if a != 0 ] ))
        
        if ans == '': 
            ans = 'no_answer'
            continue
        
        #pred_ans = get_tagging_results(evid, pred)
        answers, max_answer = get_corrected_results(evid, pred, score)
        
        
        print('Question: ', ''.join(ques), '\n')
        print('Evidence: ', ''.join(evid), '\n')
        #print('Tags: ', pred, '\n')
        print('Predict Answers: ', answers)
        if len(answers) > 1 :
            print('Select Answer: ', max_answer)
        print('Golden Answers: ', ans)
        print('\n ---------------------------- \n')
        
        
        if max_answer != ['no_answer']:
            nb_pred += 1
        
        max_answer = answers
        
        C += fuzzy_match_list(max_answer, ans)
        #if max_answer != ['no_answer']: 
        A += len(max_answer)
        #if ans != 'no_answer': Q += 1
        Q += 1
    
    if ( A == 0):
        pre = 0
    else:
        pre = C / A
    
    if ( Q == 0):
        rec = 0
    else:
        rec = C / Q
        
    if (pre + rec == 0):
        f1 = 0
    else:
        f1 = (2 * pre * rec) / (pre + rec)
        
    return pre, rec, f1, C

# Pre: 0.6158569522114896     Rec: 0.6158569522114896     F1: 0.6158569522114896 
# Pre: 0.5424550693527175     Rec: 0.7234684799186578     F1: 0.619153368856323
# select: Pre: 0.7644436058119162     Rec: 0.6160325050838841     F1: 0.6819662141911356
def test_by_evidences(model, loader, idx2word):
    print('Testing model...')
    nb_batch = 0
    epoch_pre, epoch_rec, epoch_f1, epoch_pred = 0, 0, 0, 0
    for batch_idx, (question, evidence, q_mask, e_mask, q_feat, e_feat, answer) in enumerate(loader):
        nb_batch += 1
        
        question = Variable(question.long()).cuda()
        evidence = Variable(evidence.long()).cuda()
        q_feat = Variable(q_feat.long()).cuda()
        e_feat = Variable(e_feat.long()).cuda()
        q_mask = Variable(q_mask.byte()).cuda()
        e_mask = Variable(e_mask.byte()).cuda()

        pred_scores, pred_tags = model.get_tags(question, evidence, q_mask, e_mask, q_feat, e_feat)
        pre, rec, f1 , nb_pred = get_batch_scores(pred_scores, pred_tags, answer, question, evidence, idx2word)
        print('batch:',batch_idx,'  nb_pred:', nb_pred, '   ||  pre: ', pre, '   rec: ', rec, '   f1  :', f1)
        
        epoch_pre += pre
        epoch_rec += rec
        epoch_f1 += f1
        epoch_pred += nb_pred
                                         
    epoch_pre = epoch_pre / nb_batch
    epoch_rec = epoch_rec / nb_batch
    epoch_f1 = epoch_f1 / nb_batch
    print('Pre:', epoch_pre, '    Rec:', epoch_rec,'    F1:', epoch_f1, '\n')
    return epoch_pre, epoch_rec, epoch_f1, epoch_pred


# ---------------------------------------------------
def get_corrected_results(tokens, tags, scores):
    char2word = dict()
    words = list(jieba.cut(''.join(tokens)))
    c = 0
    for i, w in enumerate(words):
        for ww in w:
            char2word[c] = i
            c += 1
            
    chunks =  []
    start = -1
    score = 0
    for i, tok in enumerate(tokens):
        if tags[i] == 0:  # B
            score += scores[i]
            if start >= 0: 
                chunks.append([start, i, score])
                score = 0 
            start = i
        elif tags[i] == 1:  # I
            score += scores[i]
            if start < 0: start = i
        else:
            if start < 0: continue
            chunks.append([start, i, score])
            start = -1
            score = 0
    if start >= 0:
        chunks.append([start, len(tokens)-1, score])
    
    answers = set()
    max_score = -1000
    max_answer = 'no_answer'
    for c in chunks:   
        ans = []
        for i in range(c[0], c[1]):
            w = words[char2word[i]]
            if w not in ans: ans.append(w)
        ans = clean_answer(''.join(ans))
        if ans != '': answers.add(ans)
    
        if (c[-1] > max_score and ans != ''):
            max_score = c[-1]
            max_answer = ans
            
    if len(answers) == 0:
        answers.add('no_answer')
    return list(answers), [max_answer]


def get_batch_ques2ans(pred_scores, pred_tags, answer, question, evidence, idx2word, ques2ans):
    question = question.data.cpu().numpy()
    evidence = evidence.data.cpu().numpy()
    for score, pred, ans , ques, evid in zip(pred_scores, pred_tags, answer, question, evidence):
        ques = ''.join( [idx2word[q] for q in ques if q != 0] )
        evid = [ idx2word[e] for e in evid if e != 0 ]
        ans = clean_answer(''.join( [ idx2word[a] for a in ans if a != 0 ] ))
        if ans == '': ans = 'no_answer'
        
        #pred_ans = get_tagging_results(evid, pred)
        answers, max_answer = get_corrected_results(evid, pred, score)
        
        #max_answer = answers
        
        ques2ans[ques] = (set([]), [])
        ques2ans[ques][0].add(ans)
        ques2ans[ques][1].extend(max_answer)
        
        print('Question: ', ''.join(ques), '\n')
        print('Evidence: ', ''.join(evid), '\n')
        #print('Tags: ', pred, '\n')
        print('Predict Answers: ', answers)
        if len(answers) > 1 :
            print('Select Answer: ', max_answer)
        print('Golden Answers: ', ans)
        print('\n ---------------------------- \n')
      
    return ques2ans


def get_epoch_scores(ques2ans):
    A, C, Q = 0, 0, 0
    for (question, answers) in ques2ans.items():
        ans = ''.join(list(answers[0]))
        pred = answers[1]
        pred = [ a for a in pred if a != 'no_answer' ]
        if pred == []:
            pred_ans = 'no_answer'
        else:
            (pred_ans, _) = Counter(pred).most_common(1)[0]
        
        print('Question: ', question)
        print('Predict Answers: ', pred_ans)
        print('Golden Answers: ', ans)
        print('\n ---------------------------- \n')
        
        C += fuzzy_match(pred_ans, ans)
        if pred_ans != 'no_answer': A += 1
        Q += 1
    
    if ( A == 0):
        pre = 0
    else:
        pre = C / A
    
    if ( Q == 0):
        rec = 0
    else:
        rec = C / Q
        
    if (pre + rec == 0):
        f1 = 0
    else:
        f1 = (2 * pre * rec) / (pre + rec)
        
    return pre, rec, f1

# new_ir
# all: Pre: 0.7     Rec: 0.5422701246210846     F1: 0.6111216549629911 
# score: Pre: 0.7524924143909839     Rec: 0.5847086561131695     F1: 0.6580742987111448

# ir 
# max score: Pre: 0.6128342245989304     Rec: 0.37896825396825395     F1: 0.46832856559051
# all: Pre: 0.5743207245604688     Rec: 0.35648148148148145     F1: 0.43991022240359107 
def test_by_questions(model, loader, idx2word):
    print('Testing model...')
    nb_batch = 0
    ques2ans = dict()
    epoch_pre, epoch_rec, epoch_f1, epoch_pred = 0, 0, 0, 0
    for batch_idx, (question, evidence, q_mask, e_mask, q_feat, e_feat, answer) in enumerate(loader):
        nb_batch += 1
        question = Variable(question.long()).cuda()
        evidence = Variable(evidence.long()).cuda()
        q_feat = Variable(q_feat.long()).cuda()
        e_feat = Variable(e_feat.long()).cuda()
        q_mask = Variable(q_mask.byte()).cuda()
        e_mask = Variable(e_mask.byte()).cuda()

        pred_scores, pred_tags = model.get_tags(question, evidence, q_mask, e_mask, q_feat, e_feat)
        ques2ans = get_batch_ques2ans(pred_scores, pred_tags, answer, question, evidence, idx2word, ques2ans)
        #print('batch:',batch_idx,'  nb_pred:', nb_pred, '   ||  pre: ', pre, '   rec: ', rec, '   f1  :', f1)
        
    pre, rec, f1 = get_epoch_scores(ques2ans)
    print('Question: ', len(ques2ans))
    print('Pre:', pre, '    Rec:', rec,'    F1:', f1, '\n')
    return pre, rec, f1
    
    

if __name__ == '__main__':
    
    print('Hey')
    
    

    
    
    
    
    
    
