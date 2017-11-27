import torch
from torch import Tensor
from torch.autograd import Variable
import random
import numpy as np
from test import get_batch_scores, clean_answer, get_corrected_results,get_tagging_results, get_batch_ques2ans, fuzzy_match
from util import load_vocab, pad_sequence
from collections import Counter
from baiduSpider import get_evidences
from loader import loadTrainDataset, loadTestDataset


STOP_TAG = "#OOV#"  

class Hyperparameters:
    tagset_size = 4
    answer_size = 16
    question_size = 64
    evidence_size = 512
    batch_size = 128
 
    train_path = '../char_data/training.h5'
    test_ann_path = '../char_data/test.ann.h5'
    test_ir_path = '../char_data/test.ir.h5'
    val_ann_path = '../char_data/validation.ann.h5'
    val_ir_path = '../char_data/validation.ir.h5'
    vocab_path = '../char_data/vocabulary.txt'

    featQA_path = '../model/featQA_2017-08-09/f1-0.5132_0.16255_2'
    #charQA_path = '../model/charQA_2017-08-11/f1-0.5583_0.34799_2'
    charQA_path = '../model/lossQA_2017-08-14/f1-0.5698_0.26918_5'

param =  Hyperparameters()   

def get_chars(seq, input2idx):
    vector = [ input2idx[s] for s in seq if s in input2idx]
    return vector, len(vector)

def get_feats(question, evidence):
    q_vector = []
    for e in evidence:
        if e in question:
            q_vector.append(1)
        else:
            q_vector.append(0) 
    return q_vector

def get_question():
    question = '谁与苗侨伟、黄日华、汤镇业、刘德华并称为"香港无线五虎将"?'
    question = '三生三世十里桃花女主角是谁'
    question = '我的前半生中，靳东演的是谁'
    question = '被英国媒体称为"东方之星"的中国斯诺克选手是谁?'
    return question

def get_inputs(question, evidences, word2idx):
    question_list = []
    evidence_list = []
    q_list = []
    e_list = []
    q_mask_list = []
    e_mask_list = []
        
    ques, q_len = get_chars(question, word2idx)
    question, q_mask = pad_sequence(ques, param.question_size, word2idx)
    
    nb_evid = len(evidences)
    for i, e in enumerate(evidences):
        e, e_len = get_chars(e, word2idx)
        if e_len == 0: continue
        
        other_id = random.randint(0, nb_evid-1)
        if nb_evid != 1:
            while other_id == i:
                other_id = random.randint(0, nb_evid-1)
        other_evidence = evidences[other_id]
        other_evidence, _ = get_chars(other_evidence , word2idx)
                
        q_feat = get_feats(ques, e)
        e_feat = get_feats(other_evidence, e)
    
        evidence, e_mask = pad_sequence(e, param.evidence_size, word2idx)
        q_tags, _ = pad_sequence(q_feat, param.evidence_size, word2idx)
        e_tags, _ = pad_sequence(e_feat, param.evidence_size, word2idx)

        question_list.append(question)
        evidence_list.append(evidence)
        q_list.append(q_tags)
        e_list.append(e_tags)
        q_mask_list.append(q_mask)
        e_mask_list.append(e_mask)
    
    question = Variable(torch.LongTensor(question_list)).cuda()
    evidence = Variable(torch.LongTensor(evidence_list)).cuda()
    e_feat = Variable(torch.LongTensor(e_list)).cuda()
    q_feat = Variable(torch.LongTensor(q_list)).cuda()
    q_mask = Variable(torch.ByteTensor(q_mask_list)).cuda()
    e_mask = Variable(torch.ByteTensor(e_mask_list)).cuda()
    
    return question, evidence, q_mask, e_mask, q_feat, e_feat 


def get_answers(model, question, evidence, q_mask, e_mask, q_feat, e_feat, idx2word):
    pred_scores, pred_tags = model.get_tags(question, evidence, q_mask, e_mask, q_feat, e_feat)
    
    ques = question.data.cpu().tolist()
    ques = ''.join([ idx2word[q] for q in ques[0] if q != 0 ])
    print('Question: ', ques, '\n')
    
    answers = []
    evidence = evidence.data.cpu().numpy()
    for score, pred, evid in zip(pred_scores, pred_tags, evidence):
        evid = [ idx2word[e] for e in evid if e != 0 ]
        print('Evidence: ', ''.join(evid), '\n')
        
        #pred_ans = get_tagging_results(evid, pred)
        answer, max_answer = get_corrected_results(evid, pred, score)
        if max_answer != ['no_answer']:
            answers.extend(max_answer)
        
        print('Predict Answers: ', max_answer, '\n')
        print('---------------\n')
     
    if answers == []:
        vote_answer = 'no_answer'
    else:
        (vote_answer, _) = Counter(answers).most_common(1)[0]
    #print('\nFinal Answer: ', vote_answer)
    return vote_answer

def get_tuple_answers(model, question, evidence, q_mask, e_mask, q_feat, e_feat, idx2word):
    pred_scores, pred_tags = model.get_tags(question, evidence, q_mask, e_mask, q_feat, e_feat)
    
    ques = question.data.cpu().tolist()
    ques = ''.join([ idx2word[q] for q in ques[0] if q != 0 ])
    print('Question: ', ques, '\n')
    
    ans2evid = {}
    answers = []
    evidence = evidence.data.cpu().numpy()
    for score, pred, evid in zip(pred_scores, pred_tags, evidence):
        evid = [ idx2word[e] for e in evid if e != 0 ]
        print('Evidence: ', ''.join(evid), '\n')
        
        #pred_ans = get_tagging_results(evid, pred)
        answer, max_answer = get_corrected_results(evid, pred, score)
        if max_answer != ['no_answer']:
            answers.extend(max_answer)
        
        if max_answer[0] not in ans2evid:
            ans2evid[max_answer[0]] = []
        ans2evid[max_answer[0]].append(''.join(evid))
        
        print('Predict Answers: ', max_answer, '\n')
        print('---------------\n')
     
    if answers == []:
        votes = [('no_answer', 0)]
    else:
        votes = Counter(answers).most_common(2)
        
    #print('\nFinal Answer: ', vote_answer)
    return votes, ans2evid

def get_batch_scores(C, A, Q):
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

#Nb_batch:  24 Pre: 0.5909090909090909     Rec: 0.5869565217391305     F1: 0.5889261744966443
def test_online(model, loader, idx2word, word2idx):
    print('Testing online ...')
    ques2ans = dict()
    A, C, Q = 0, 0, 0
    nb_batch, nb_epoch = 0, 0
    pre_epoch, rec_epoch, f1_epoch = 0, 0, 0
    for batch_idx, (question, evidence, q_mask, e_mask, q_feat, e_feat, answer) in enumerate(loader):
        nb_batch += 1
        #answer = answer.cpu().tolist()
        for ques, ans in zip(question, answer):
            nb_epoch += 1
            ans = ''.join([ idx2word[a] for a in ans if a != 0])
            ques = ''.join([ idx2word[q] for q in ques if q != 0])
            print(nb_epoch, '.Question: ', ques)
            
            evidences = get_evidences(ques, 20)
            if evidences == []:
                print('No Evidence\n')
                continue
                
            question, evidence, q_mask, e_mask, q_feat, e_feat = get_inputs(ques, evidences, word2idx)
            pred_ans = get_answers(model, question, evidence, q_mask, e_mask, q_feat, e_feat, idx2word)
            
            print('\n ##################### \n')
            print(nb_epoch, '.Question: ', ques)
            print('Seclecd Answer: ', pred_ans)
            print('Golden Answer: ', ans)
            print('\n ##################### \n')
            
            C += fuzzy_match(pred_ans, ans)
            if pred_ans != 'no_answer': A += 1
            Q += 1
            
        pre, rec, f1 = get_batch_scores(C, A, Q)
        pre_epoch += pre 
        rec_epoch += rec
        f1_epoch += f1
        print('Nb_batch: ', nb_batch ,'Pre:', pre, '    Rec:', rec,'    F1:', f1, '\n')
        
    pre_epoch = pre_epoch / nb_batch
    rec_epoch = rec_epoch / nb_batch
    f1_epoch = f1_epoch / nb_batch
    
    print('Pre:', pre_epoch, '    Rec:', rec_epoch, '    F1:', f1_epoch, '\n')
    return pre_epoch, rec_epoch, f1_epoch


if __name__ == '__main__':
    word_set, word2idx, word_set_size = load_vocab(param.vocab_path)
    idx2word = dict(zip(word2idx.values(), word2idx.keys()))
    model = torch.load(param.charQA_path)
    model.eval()
    
    test_dataset = loadTestDataset(param.test_ann_path)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = param.batch_size, num_workers = 1, shuffle = True)
    #test_online(model, test_loader, idx2word, word2idx)
    

    question = get_question()
    #evidences = get_evidences()
    evidences = get_evidences(question)  
    
    question, evidence, q_mask, e_mask, q_feat, e_feat = get_inputs(question, evidences, word2idx)
    #answers = get_answers(model, question, evidence, q_mask, e_mask, q_feat, e_feat, idx2word)
    answers = get_tuple_answers(model, question, evidence, q_mask, e_mask, q_feat, e_feat, idx2word)

   
    
   