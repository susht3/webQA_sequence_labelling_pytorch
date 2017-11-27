import h5py
import math
import torch
import torch.utils.data as data

class loadTrainDataset(data.Dataset):
    def __init__(self, path):
        self.file = h5py.File(path)
        self.nb_samples = len(self.file['question'][:])

    def __getitem__(self, index):
        question = self.file['question'][index]
        evidence = self.file['evidence'][index]
        q_mask = self.file['q_mask'][index]
        e_mask = self.file['e_mask'][index]
        q_feat = self.file['q_feat'][index]
        e_feat = self.file['e_feat'][index]
        tags = self.file['labels'][index]
        answer = self.file['answer'][index]
        #return question, evidence, q_mask, e_mask, q_feat, e_feat, tags
        return question, evidence, q_mask, e_mask, q_feat, e_feat, tags, answer

    def __len__(self):
        return self.nb_samples
    
    
    
class loadTestDataset(data.Dataset):
    def __init__(self, path):
        self.file = h5py.File(path)
        self.nb_samples = len(self.file['question'][:])

    def __getitem__(self, index):
        question = self.file['question'][index]
        evidence = self.file['evidence'][index]
        q_mask = self.file['q_mask'][index]
        e_mask = self.file['e_mask'][index]
        q_feat = self.file['q_feat'][index]
        e_feat = self.file['e_feat'][index]
        answer = self.file['answer'][index]
        return question, evidence, q_mask, e_mask, q_feat, e_feat, answer

    def __len__(self):
        return self.nb_samples
        