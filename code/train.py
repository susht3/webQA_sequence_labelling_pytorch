import torch
from torch.autograd import Variable
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np
import csv

def save_model(model, epoch, loss, acc,  model_dir):
    model_path = model_dir + 'loss' + str(round(loss, 4)) + '_acc' + str(round(acc, 4)) + '_' + str(epoch)
    with open(model_path, 'wb') as f:
        torch.save(model, f)

def train(model, train_loader, param):
    print('Training model...')
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters, lr = param.learning_rate)  
    
    vaild_loss = 0
    best_acc = 0
    for epoch in range(param.nb_epoch):
        train_loss, acc = train_epoch(model, epoch, train_loader, optimizer)

        if(acc >= best_acc):
            best_acc = acc
            save_model(model, epoch, train_loss, acc,  param.model_dir) 

    print('Train End.\n')


def train_epoch(model, epoch, loader, optimizer):
    print('Train epoch :', epoch)
    model.train()

    label_list = []
    pred_list = []
    prob_list = []
    
    epoch_loss = 0.0
    nb_batch = 0
    for batch_idx, (attribute, label) in enumerate(loader):
        nb_batch += 1

        attribute = Variable(attribute.float())#.cuda()
        label = Variable(label.long(), requires_grad = False)

        if torch.cuda.is_available():
            attribute = attribute.cuda()
            label = label.cuda()

        batch_loss = model.get_loss(attribute, label) 
        pred = model.get_tags(attribute)


        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        epoch_loss += sum(batch_loss.data.cpu().numpy())
        print('-----epoch:', epoch, ' batch:',batch_idx,' train_loss:', batch_loss.data[0])

        label_list.extend(label.data.cpu().tolist())
        pred_list.extend(pred)
        

    epoch_loss = epoch_loss / nb_batch

    acc = accuracy_score(np.asarray(label_list), np.asarray(pred_list))

    print('\nEpoch: ', epoch, ', Train Loss: ', epoch_loss, '\n', 'Accuracy: ', acc)

    return epoch_loss, acc

def test(model, loader, param):
    print('Getting test csv file ....')
    pred_list = []
    headers = ['ImageId','Label']
    f = open(param.res_path, 'w')
    f_csv = csv.writer(f)
    f_csv.writerow(headers)

    i = 1
    for batch_idx, (attribute) in enumerate(loader):
        attribute = Variable(attribute.float())#.cuda()
        if torch.cuda.is_available():
            attribute = attribute.cuda()
        
        pred = model.get_tags(attribute)
        for p in pred:
            f_csv.writerow([i, p])
            i = i+1

    f.close()
    print('Test over')


if __name__ == '__main__':
    print('Hey')