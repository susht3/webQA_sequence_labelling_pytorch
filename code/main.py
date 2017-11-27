import torch
import os
import time
from datetime import datetime
from baselineQA import baselineQA
from loader import loadTrainDataset, loadTestDataset
from util import load_vocab, load_webQA_vocab, load_webQA_embedding
from train_loss import train
from test import test_by_evidences, test_by_questions


class Hyperparameters:
    nb_epoch = 1000
    batch_size = 128
    tagset_size = 4
    question_size = 64
    evidence_size = 512

    qe_embedding_size = 2
    embedding_size = 64
    
    q_hidden_size = 64
    e_hidden_size = 128
    t_hidden_size = 64
    num_layers = 1
        
    pre_embeds = False
    pre_lstm = False
    clip = 5.0
    learning_rate = 0.001
    model_dir = ''


class Paths:
    train_path = '../data/training.h5'
    test_ann_path = '../data/test.ann.h5'
    test_ir_path = '../data/test.ir.h5'
    new_test_ir_path = '../data/new_test.ir.h5'
    val_ann_path = '../data/validation.ann.h5'
    val_ir_path = '../data/validation.ir.h5'
    vocab_path = '../data/vocabulary.txt'
    
    attQA_path = '../model/lstm_2017-07-26/weights.0-0.1475'
    maskQA_path = '../model/weightQA_2017-08-04/f1-0.4113_0.00173_59'
    #featQA_path = '../model/feat2QA_2017-08-08/f1-0.5119_0.04968_5'
    featQA_path = '../model/featQA_2017-08-09/f1-0.5132_0.16255_2'
    charQA_path = '../model/charQA_2017-08-11/f1-0.5583_0.34799_2'
    
path = Paths()

    
def train_featQA(train_loader, val_loader, param): # 2 is not weight
    param.model_dir = '../model/baselineQA_' + str(datetime.now()).split('.')[0].split()[0] + '/'
    if os.path.exists(param.model_dir) == False:
        os.mkdir(param.model_dir)
        
    word_set, word2idx, vocab_size = load_vocab(path.vocab_path)
    idx2word = dict(zip(word2idx.values(), word2idx.keys()))
    
    model = baselineQA(vocab_size, param, 0).cuda()
    train(model, train_loader, val_loader, param, idx2word)
    
    
def test_model(loader, model_path):
    word_set, word2idx, vocab_size = load_vocab(path.vocab_path)
    idx2word = dict(zip(word2idx.values(), word2idx.keys()))
    
    model = torch.load(model_path)
    #test_by_evidences(model, loader, idx2word) # 0.58 0.62
    test_by_questions(model, loader, idx2word) # 0.61 0.66
    


if __name__ == '__main__':
    param = Hyperparameters() 
    
    train_dataset = loadTrainDataset(path.train_path)
    val_dataset = loadTestDataset(path.val_ann_path)
    test_dataset = loadTestDataset(path.val_ann_path)
     
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = param.batch_size, num_workers = 1, shuffle = True)  
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = param.batch_size, num_workers = 1, shuffle = True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = param.batch_size, num_workers = 1, shuffle = True)
    
    print('Biu ~ ~  ~ ~ ~ Give you buffs ~ \n')
    
    
    #train_maskQA(train_loader, val_loader, param)  
    #train_attQA(train_loader, val_loader, param)
    #train_featQA(train_loader, val_loader, param)

    test_model(test_loader, path.charQA_path)
    print('test dataset: ', test_dataset.__len__())

    #print(test_dataset.nb_samples)
    
    #--------------------  record  --------------------------
    
    # '../model/feat2QA_2017-08-08/f1-0.5119_0.04968_5'
    # test_ann: Pre: 0.46563813811991994     Rec: 0.5778645833333333     F1: 0.5154754567060206
    # test_ir: Pre: 0.3044053618532579     Rec: 0.35484035326086955     F1: 0.3275912294056864
    
    # model/featQA_2017-08-09/f1-0.5132_0.16255_2
    # test_ann: Pre: 0.45854179487277863     Rec: 0.5921223958333334     F1: 0.5165802991715742
    # test_ir: Pre: 0.3061375799953651     Rec: 0.3651154891304348     F1: 0.3328905391650594
    # new_test_ir: Pre: 0.47119347316799043     Rec: 0.59765625     F1: 0.5266610723362033
    # ann_can_pred: Pre: 0.6123457108622822     Rec: 0.8700601263366222     F1: 0.7181746023030019
    
    
    
    
