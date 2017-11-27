import torch
import pycrfsuite
import h5py
import numpy as np
from tqdm import *
from torch.autograd import Variable
from loader import loadTrainDataset, loadTestDataset
from util import load_vocab

STOP_TAG = "#OOV#" 

class Hyperparameters:
    batch_size = 128
    e_hidden_size = 128
    evidence_size = 512
    
    train_path = '../char_data/training.h5'
    test_ann_path = '../char_data/test.ann.h5'
    test_ir_path = '../char_data/test.ir.h5'
    new_test_ir_path = '../char_data/new_test.ir.h5'
    val_ann_path = '../char_data/validation.ann.h5'
    val_ir_path = '../char_data/validation.ir.h5'
    vocab_path = '../char_data/vocabulary.txt'

    crf_path = '../model/test.crfsuite'
    crf_train_path = '../char_data/crf_training.h5'
    charQA_path = '../model/charQA_2017-08-11/f1-0.5583_0.34799_2'

param = Hyperparameters()    
    
    
def get_inputs(model, loader, idx2word):
    print('Getting inputs...')
    x_train = []
    y_train = []
    count = 0
    tag2idx = { "b": 0, "i": 1, "o1": 2, "o2":3, STOP_TAG: 4}
    idx2tag = dict(zip(tag2idx.values(), tag2idx.keys()))
    feats = [ str(i) for i in range(param.e_hidden_size)] 
    for batch_idx, (question, evidence, q_mask, e_mask, q_feat, e_feat, labels, answer) in tqdm(enumerate(loader)):
        
        #count += 1
        #if count == 2:
        #    break
        
        question = Variable(question.long()).cuda()
        evidence = Variable(evidence.long()).cuda()
        q_feat = Variable(q_feat.long()).cuda()
        e_feat = Variable(e_feat.long()).cuda()
        q_mask = Variable(q_mask.byte()).cuda()
        e_mask = Variable(e_mask.byte()).cuda()
        labels = Variable(labels.long(), requires_grad = False).cuda()
        
        batch_lstm = model.get_lstm(question, evidence, q_mask, e_mask, q_feat, e_feat)
        batch_lstm = batch_lstm.data.cpu().tolist()
        labels = labels.data.cpu().tolist()
        
        for ans, label, lstm in zip(answer, labels, batch_lstm):
            ans = [ idx2word[a] for a in ans if a != 0 ]
            if (''.join(ans) == 'no_answer'): continue
            
            y_train.append([idx2tag[l] for l in label])  
            
            seq = []
            for vec in lstm:
                seq.append(dict(zip(feats, vec)))
            x_train.append(seq)

    
        print('y train: (', len(y_train),' , ', len(y_train[0]), ')')
        print('x train: (', len(x_train),' , ', len(x_train[0]), ' , ', len(x_train[0][0]),')')
            
    return x_train, y_train


def save_inputs(model, loader, dataset_size, idx2word):
    print('Saving inputs...')
    file = h5py.File(param.crf_train_path,'w')
    tag2idx = { "b": 0, "i": 1, "o1": 2, "o2":3, STOP_TAG: 4}
    idx2tag = dict(zip(tag2idx.values(), tag2idx.keys()))
    feats = [ str(i) for i in range(param.e_hidden_size)] 
    
    file.create_dataset('y', (dataset_size, param.evidence_size), 'i')
    file.create_dataset('x', (dataset_size, param.evidence_size, param.e_hidden_size), 'i')
    start = 0
    idx = 0
    for batch_idx, (question, evidence, q_mask, e_mask, q_feat, e_feat, labels, answer) in tqdm(enumerate(loader)):
        
        #count += 1
        #if count == 2:
        #    break
        
        batch_size = len(question)
        question = Variable(question.long()).cuda()
        evidence = Variable(evidence.long()).cuda()
        q_feat = Variable(q_feat.long()).cuda()
        e_feat = Variable(e_feat.long()).cuda()
        q_mask = Variable(q_mask.byte()).cuda()
        e_mask = Variable(e_mask.byte()).cuda()
        labels = Variable(labels.long(), requires_grad = False).cuda()

        batch_lstm = model.get_lstm(question, evidence, q_mask, e_mask, q_feat, e_feat)
        batch_lstm = batch_lstm.data.cpu().tolist()
        labels = labels.data.cpu().tolist()
        
        i = 0
        for ans, label, lstm in zip(answer, labels, batch_lstm):
            ans = [ idx2word[a] for a in ans if a != 0 ]
            if (''.join(ans) == 'no_answer'): 
                if i == 10: continue
                else: i += 1
            
            file['y'][idx] = label
            file['x'][idx] = lstm
            idx += 1
            
        print(idx)
        '''
        end = (batch_idx+1)*batch_size
        print('(', start, ',',  end, ')')
        file['x'][start: end] = batch_lstm
        file['y'][start: end] = labels  
        start = end
        '''
        
    file.close()
    return idx


def load_inputs(idx):
    print('Loading inputs...')
    file = h5py.File(param.crf_train_path)
    x = file['x'][:idx]
    y = file['y'][:idx]
    file.close()
    
    x_train = []
    y_train = []
    tag2idx = { "b": 0, "i": 1, "o1": 2, "o2":3, STOP_TAG: 4}
    idx2tag = dict(zip(tag2idx.values(), tag2idx.keys()))
    feats = [ str(i) for i in range(param.e_hidden_size)] 
    
    for label, lstm in tqdm(zip(y, x)):
        y_train.append([idx2tag[l] for l in label])  

        seq = []
        for vec in lstm:
            seq.append(dict(zip(feats, vec)))
        x_train.append(seq)
        
        print(' y train: (', len(y_train),',', len(y_train[0]), ')')
    
    return x_train, y_train


def train_crf(x_train, y_train):
    print('Training...')
    trainer = pycrfsuite.Trainer(verbose=False)
    for xseq, yseq in zip(x_train, y_train):
        trainer.append(xseq, yseq)
    
    trainer.set_params({
        'c1': 1.0,   # coefficient for L1 penalty
        'c2': 1e-3,  # coefficient for L2 penalty
        'max_iterations': 500,  # stop earlier

        # include transitions that are possible, but not observed
        'feature.possible_transitions': True
    })

    trainer.train(param.crf_path)



def get_tags(xs, ys):
    print('Testing...')
    tagger = pycrfsuite.Tagger()
    tagger.open(param.crf_path)

    for x , y in zip(xs, ys):
        pred_tag = tagger.tag(x)
        pred_tag = [ t for t in pred_tag if t != STOP_TAG]
        y = [ t for t in y if t != STOP_TAG]
        print("Predicted:", ' '.join(pred_tag))
        print("Correct:  ", ' '.join(y), '\n')
    
    #return pred_tag


def test():
    X_train = [[{'foo': 1, 'bar': 0, 's':0, 'p': 4, 'd':True, 'a':0.7, 'b': 0.5, 'c': 9}, 
            {'foo': 0, 'baz': 1, 's':0, 'p': 0, 'd': False, 'a':8.7, 'b': 7.5, 'c': 1}]]
    X_train = [[['foo=1', 'bar=0', 'c=9', 's=0', 'sd=12', 'cd=2', 'ca=3', 'd=True', 'cc=89'], 
            ['foo=4', 'bar=7', 'c=3', 's=1', 'sd=8', 'cd=9', 'ca=1','d=False', 'cc=18']]]
    y_train = [['0', '1']]
    #print('x train: ', y_train[0])


    trainer = pycrfsuite.Trainer(verbose=False)

    for xseq, yseq in zip(X_train, y_train):
        print('x: ', xseq)
        print('y: ', yseq)
        trainer.append(xseq, yseq)

    trainer.set_params({
        'c1': 1.0,   # coefficient for L1 penalty
        'c2': 1e-3,  # coefficient for L2 penalty
        'max_iterations': 500,  # stop earlier

        # include transitions that are possible, but not observed
        'feature.possible_transitions': True
    })



    trainer.train('conll2002-esp.crfsuite')
    #print (len(trainer.logparser.iterations), trainer.logparser.iterations[-1])


    tagger = pycrfsuite.Tagger()
    tagger.open('conll2002-esp.crfsuite')

    print("Predicted:", ' '.join(tagger.tag(X_train[0])))
    print("Correct:  ", ' '.join(y_train[0]))

    
if __name__ == '__main__':
    #test()
    
    train_dataset = loadTrainDataset(param.train_path)
    val_dataset = loadTestDataset(param.val_ann_path)
    test_dataset = loadTestDataset(param.new_test_ir_path)
     
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = param.batch_size, num_workers = 1, shuffle = False)  
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = param.batch_size, num_workers = 1, shuffle = False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = param.batch_size, num_workers = 1, shuffle = False)
    
    model = torch.load(param.charQA_path)
    
    
    train_dataset_size = train_dataset.__len__()
    word_set, word2idx, vocab_size = load_vocab(param.vocab_path)
    idx2word = dict(zip(word2idx.values(), word2idx.keys()))
    
    #idx = save_inputs(model, train_loader, train_dataset_size, idx2word)
    #print('idx: ', idx)
    
    idx = 159753
    
    x_train, y_train = load_inputs(idx)
    
    #x_train, y_train = get_inputs(model, train_loader, idx2word)
    
    train_crf(x_train, y_train)
    
    get_tags(x_train[0:20], y_train[0:20])





