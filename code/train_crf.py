import torch
import torch.nn as nn
from torch.autograd import Variable
import os
import time
from datetime import datetime
from test import get_batch_scores

def save_train_model(model, epoch, f1, loss, model_dir):
    model_path = model_dir + 'loss-' + str(round(loss, 6)) + '_' + str(round(f1, 3)) + '_' + str(epoch)
    with open(model_path, 'wb') as f:
        torch.save(model, f)

def save_vaild_model(model, epoch, f1, loss, model_dir):
    model_path = model_dir + 'f1-' + str(round(f1, 4)) + '_' + str(round(loss, 5)) + '_' + str(epoch)
    with open(model_path, 'wb') as f:
        torch.save(model, f)
            
def train(model, train_loader, valid_loader, param, idx2word):
    print('Training model...')
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters, lr = param.learning_rate)  
    
    vaild_loss = 0
    best_loss = 1000
    best_f1 = 0
    for epoch in range(param.nb_epoch):
        epoch = epoch 
        train_loss = train_epoch(model, epoch, train_loader, optimizer)
        pre, rec, f1 = eval_epoch(model, epoch, valid_loader, idx2word)

        if(f1 > best_f1):
            best_f1 = f1
            save_vaild_model(model, epoch, f1, train_loss, param.model_dir) 
            
        elif(train_loss < best_loss):
            best_loss = train_loss
            save_train_model(model, epoch, f1, train_loss, param.model_dir) 

    print('Train End.\n')
    

def train_epoch(model, epoch, loader, optimizer):
    print('Train epoch :', epoch)
    model.train()
                                         
    epoch_loss = 0.0
    nb_batch = 0
    for batch_idx, (question, evidence, q_mask, e_mask, q_feat, e_feat, labels) in enumerate(loader):
        nb_batch += 1

        question = Variable(question.long()).cuda()
        evidence = Variable(evidence.long()).cuda()
        q_feat = Variable(q_feat.long()).cuda()
        e_feat = Variable(e_feat.long()).cuda()
        q_mask = Variable(q_mask.byte()).cuda()
        e_mask = Variable(e_mask.byte()).cuda()
        labels = Variable(labels.long(), requires_grad = False).cuda()
        
        lstm = get_lstm(self, question, evidence, q_mask, e_mask, q_feat, e_feat)
        batch_loss = model.get_lstm(question, evidence, q_mask, e_mask, q_feat, e_feat, labels) 
        
        optimizer.zero_grad()
        batch_loss.backward()  
        #nn.utils.clip_grad_norm(model.parameters(), max_norm = 5.0)
        optimizer.step()
            
        epoch_loss += sum(batch_loss.data.cpu().numpy())
        print('-----epoch:', epoch, ' batch:',batch_idx,' train_loss:', batch_loss.data[0])
        
    epoch_loss = epoch_loss / nb_batch
    print('\nEpoch: ', epoch, ', Train Loss: ', epoch_loss, '\n')
    return epoch_loss


def eval_epoch(model, epoch, loader, idx2word):
    print('Eval epoch :', epoch)
    model.eval()
    
    nb_batch = 0
    epoch_pre, epoch_rec, epoch_f1, epoch_pred = 0, 0, 0, 0
    for batch_idx, (question, evidence, q_mask, e_mask, q_feat, e_feat, answer) in enumerate(loader):
        nb_batch += 1
        question = Variable(question.long()).cuda()
        evidence = Variable(evidence.long()).cuda()
        e_feat = Variable(e_feat.long()).cuda()
        q_feat = Variable(q_feat.long()).cuda()
        q_mask = Variable(q_mask.byte()).cuda()
        e_mask = Variable(e_mask.byte()).cuda()

        pred_scores, pred_tags = model.get_tags(question, evidence, q_mask, e_mask, q_feat, e_feat)
        
        question = question.data.cpu().numpy()
        evidence = evidence.data.cpu().numpy()
        pre, rec, f1 , nb_pred = get_batch_scores(pred_tags, answer, question, evidence, idx2word)
        print('-----epoch:', epoch, ' batch:',batch_idx,'  can_pred:', nb_pred, '   ||  pre: ', pre, '   rec: ', rec, '   f1  :', f1)
        
        epoch_pre += pre
        epoch_rec += rec
        epoch_f1 += f1
        epoch_pred += nb_pred
                                         
    epoch_pre = epoch_pre / nb_batch
    epoch_rec = epoch_rec / nb_batch
    epoch_f1 = epoch_f1 / nb_batch
    print('\nEpoch: ', epoch, '  Pred: ', epoch_pred, '  || Pre:', epoch_pre, '    Rec:', epoch_rec,'    F1:', epoch_f1, '\n')
    return epoch_pre, epoch_rec, epoch_f1




if __name__ == '__main__':

    print('Hey')
    
    

    
    
    
    
    
    
