import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class baselineQA(nn.Module):
    
    def __init__(self, vocab_size, param, embeds):
        super(baselineQA, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = param.embedding_size
        self.qe_embedding_size = param.qe_embedding_size
        
        self.tagset_size = param.tagset_size
        self.evidence_size = param.evidence_size
        self.q_hidden_size = 48 #param.q_hidden_size
        self.e_hidden_size = 96 #param.e_hidden_size
        self.t_hidden_size = param.t_hidden_size
        self.num_layers = 1
        
        self.lookup = nn.Embedding(self.vocab_size, self.embedding_size)
        self.q_lookup = nn.Embedding(2, self.qe_embedding_size)
        self.e_lookup = nn.Embedding(2, self.qe_embedding_size)
        
        if param.pre_embeds == True :
            self.lookup.weight.data.copy_(torch.from_numpy(embeds))
            for param in self.lookup.parameters():
                param.requires_grad = False
        
        self.q_size = self.embedding_size + self.q_hidden_size + self.qe_embedding_size * 2
        self.q_lstm = nn.LSTM(self.embedding_size, self.q_hidden_size, self.num_layers, dropout = 0.1)
        self.e_lstm = nn.LSTM(self.q_size, self.e_hidden_size // 2, self.num_layers, dropout = 0.2, bidirectional = True)
        self.t_lstm = nn.LSTM(self.e_hidden_size, self.t_hidden_size, self.num_layers)
        
        self.att_linear = nn.Linear(self.q_hidden_size, 1)
        self.hidden2tag_linear = nn.Linear(self.e_hidden_size, self.tagset_size + 1)
        self.norm = nn.BatchNorm1d(self.evidence_size, self.e_hidden_size)

        #self.weight = torch.FloatTensor([2.0, 2.0, 0.6, 0.6, 0]).cuda() # char 08-12 / 13
        #self.weight = torch.FloatTensor([2.2, 2.0, 0.5, 0.5, 0]).cuda() #char 08-11
        #self.weight = torch.FloatTensor([2.0, 2.0, 0.8, 0.8, 0]).cuda() #score 08-12
        self.weight = torch.FloatTensor([2.0, 2.0, 0.2, 0.2, 0]).cuda() # char 14
        self.loss_func = nn.NLLLoss(weight = self.weight)
        
        #self.loss_func = nn.NLLLoss()
    
    
    def init_hidden(self, num_layers, batch_size, hidden_size):
        h0 = Variable(torch.zeros(num_layers, batch_size, hidden_size)).cuda()
        c0 = Variable(torch.zeros(num_layers, batch_size, hidden_size)).cuda()
        return (h0, c0)
    
    
    # x = (batch, seq_len, hsize)
    # return (batch, hidden_size)
    def attention(self, x, x_mask):
        x_flat = x.view(-1, x.size(-1))
        scores = self.att_linear(x_flat).view(x.size(0), x.size(1))
        scores.data.masked_fill_(x_mask.data, -float('inf'))
        weights = F.softmax(scores)
        out = weights.unsqueeze(1).bmm(x).squeeze(1)
        return out
    
    
    # return pack rnn inputs
    def get_pack_rnn_inputs(self, x, x_mask):
        lengths = x_mask.data.eq(0).long().sum(1).squeeze()
        _, idx_sort = torch.sort(lengths, dim = 0, descending = True)
        _, idx_unsort = torch.sort(idx_sort, dim = 0)

        lengths = list(lengths[idx_sort])

        # sort x
        x = x.index_select(0, Variable(idx_sort))
        x = x.transpose(0, 1).contiguous()
        rnn_input = nn.utils.rnn.pack_padded_sequence(x, lengths)
        
        return rnn_input, Variable(idx_unsort)

    
    def get_pad_rnn_outputs(self, output, x_mask, idx_unsort):
        output = nn.utils.rnn.pad_packed_sequence(output)[0]
        
        # transpose and unsort
        output = output.transpose(0, 1).contiguous()
        output = output.index_select(0, idx_unsort)

        # pad up to original batch sequence length
        if output.size(1) != x_mask.size(1):
            padding = torch.zeros(output.size(0),
                                  x_mask.size(1) - output.size(1),
                                  output.size(2)).type(output.data.type())
            output = torch.cat([output, Variable(padding)], 1)
        
        return output
    
    
    # embeds = (batch, seq_len, embedding_size)
    # return (batch, q_size)
    def question_lstm(self, question, q_mask):
        batch_size = question.size()[0]
        embeds = self.lookup(question)
        inputs, idx_unsort = self.get_pack_rnn_inputs(embeds, q_mask)
        
        init_hidden = self.init_hidden(self.num_layers, batch_size, self.q_hidden_size)
        lstm_out, _ = self.q_lstm(inputs, init_hidden)
        lstm_out = self.get_pad_rnn_outputs(lstm_out, q_mask, idx_unsort)

        lstm_vector = self.attention(lstm_out, q_mask)
        return lstm_vector
    
    
    # return (batch, seq_len, e_size)
    def evidence_lstm(self, evidence, q_vector, q_tag, e_tag, e_mask):
        batch_size = evidence.size()[0]
        embeds = self.lookup(evidence)
        q_feat, e_feat = self.q_lookup(q_tag), self.e_lookup(e_tag)
        
        q_vector = q_vector.expand(self.evidence_size, *q_vector.size()) 
        q_vector = q_vector.transpose(0,1).contiguous()

        inputs = torch.cat([embeds, q_vector, q_feat, e_feat], -1)
        inputs, idx_unsort = self.get_pack_rnn_inputs(inputs, e_mask)
        
        init_hidden = self.init_hidden(self.num_layers * 2, batch_size, self.e_hidden_size // 2)
        lstm_out, _ = self.e_lstm(inputs, init_hidden)
        
        lstm_out = self.get_pad_rnn_outputs(lstm_out, e_mask, idx_unsort)
        return lstm_out
    
    
    # return (batch, seq_len, t_size)
    def tagger_lstm(self, inputs, e_mask, batch_size, idx_unsort):
        init_hidden = self.init_hidden(self.num_layers, batch_size, self.t_hidden_size)
        lstm_out, _ = self.t_lstm(inputs, init_hidden)
        lstm_out = self.get_pad_rnn_outputs(lstm_out, e_mask, idx_unsort)
        return lstm_out
    
    
    # return (batch, seq_len, tsize)
    def get_lstm(self, question, evidence, q_mask, e_mask, q_feat, e_feat):
        q_lstm = self.question_lstm(question, q_mask)
        e_lstm = self.evidence_lstm(evidence, q_lstm, q_feat, e_feat, e_mask)
        #t_lstm = self.tagger_lstm(e_lstm, e_mask, batch_size, idx_unsort)  
        lstm = self.norm(e_lstm)
        return e_lstm
   

    # return (batch, seq_len, tag_size)
    def forward(self, question, evidence, q_mask, e_mask, q_feat, e_feat):
        lstm = self.get_lstm(question, evidence, q_mask, e_mask, q_feat, e_feat)
        score_list = []
        for t in lstm: 
            tag_space = self.hidden2tag_linear(t)
            tag_scores = F.log_softmax(tag_space)
            score_list.append(tag_scores) 
        scores = torch.cat(score_list, 0).view(len(score_list), *score_list[0].size())
        return scores    

    
    # return (batch, seq_len)
    def get_tags(self, question, evidence, q_mask, e_mask, q_feat, e_feat):
        scores = self.forward(question, evidence, q_mask, e_mask, q_feat, e_feat)
        score, tags = torch.max(scores, dim = -1)
        return score.data.cpu().tolist(), tags.data.cpu().tolist()
  
    '''
    # return (batch, seq_len)
    def get_tags(self, question, evidence, q_mask, e_mask, q_feat, e_feat):
        scores = self.forward(question, evidence, q_mask, e_mask, q_feat, e_feat)
        score, tags = torch.max(scores, dim = -1)
        return tags.data.cpu().tolist()
    '''
    
    # return one value
    def get_loss(self, question, evidence, q_mask, e_mask, q_feat, e_feat, labels):
        scores = self.forward(question, evidence, q_mask, e_mask, q_feat, e_feat)
        loss_list = []
        for tag_scores, tag in zip(scores, labels):
            loss = self.loss_func(tag_scores, tag)
            loss_list.append(loss)
        batch_loss = torch.mean(torch.cat(loss_list, -1))
        return batch_loss
    
    
    
    


    
