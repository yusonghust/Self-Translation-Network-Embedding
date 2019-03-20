# encoding: utf-8
import time
import torch
import numpy as np
import torch.nn as nn
import torchvision
from config import config
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from utils import *
import os
os.environ["CUDA_VISIBLE_DEVICES"]='1'

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SeqDataIter(Dataset):
    def __init__(self,seqs):
        self.data = seqs

    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        seq = self.data[idx]
        return seq


class STNE(nn.Module):
    def __init__(self,cfg):
        super(STNE,self).__init__()
        self.cfg = cfg
        self.encoder = nn.LSTM(self.cfg.fea_dim,self.cfg.h_dim,self.cfg.dpt,batch_first=True,bidirectional=True,dropout=self.cfg.dropout if self.cfg.dpt>1 else 0)
        self.decoder = nn.LSTM(2*self.cfg.h_dim,2*self.cfg.h_dim,batch_first=True,bidirectional=False)
        self.fc_h = nn.Linear(2*self.cfg.dpt*self.cfg.h_dim,2*self.cfg.h_dim)
        self.fc_c = nn.Linear(2*self.cfg.dpt*self.cfg.h_dim,2*self.cfg.h_dim)
        self.drop = nn.Dropout(self.cfg.dropout)
        self.fc = nn.Linear(2*self.cfg.h_dim,self.cfg.node_num)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self,embedding_W,seqs):
        input_seqs = torch.LongTensor(seqs).to(device)
        input_seq_embed = embedding_W(input_seqs)
        encoder_out,(encoder_h,encoder_c) = self.encoder(input_seq_embed)

        encoder_h = self.drop(encoder_h)
        encoder_h = encoder_h.permute(1,0,2).reshape(input_seqs.size()[0],-1)
        encoder_h = self.fc_h(encoder_h).reshape(1,input_seqs.size()[0],2*self.cfg.h_dim)
        encoder_c = self.drop(encoder_c)
        encoder_c = encoder_c.permute(1,0,2).reshape(input_seqs.size()[0],-1)
        encoder_c = self.fc_c(encoder_c).reshape(1,input_seqs.size()[0],2*self.cfg.h_dim)

        decoder_out,_ = self.decoder(encoder_out,(encoder_h,encoder_c)) # out: tensor of shape (batch_size, seq_length, hidden_size*2)
        out = self.fc(decoder_out)
        out = out.reshape(input_seqs.size()[0],self.cfg.node_num,-1)
        loss = self.criterion(out,input_seqs)
        return decoder_out,loss

def main():
    cfg = config()

    start = time.time()
    X, Y = read_node_label(cfg.node_label_file)
    node_fea = read_node_features(cfg.node_fea_file)
    node_seq = read_node_sequences(cfg.node_seq_file)

    cfg.update_config('node_list',X)
    cfg.update_config('node_label',Y)
    cfg.update_config('node_fea',node_fea)
    cfg.update_config('node_seq',node_seq)
    cfg.update_config('node_num',node_fea.shape[0])
    cfg.update_config('fea_dim',node_fea.shape[1])

    if cfg.node_fea is not None:
        assert cfg.node_num == cfg.node_fea.shape[0] and cfg.fea_dim == cfg.node_fea.shape[1]
        embedding_W = nn.Embedding.from_pretrained(torch.FloatTensor(cfg.node_fea),freeze=cfg.freeze).to(device)
    else:
        embedding_W = nn.Embedding(cfg.node_num,cfg.fea_dim).to(device)

    dataset = SeqDataIter(cfg.node_seq)
    dataloader = DataLoader(dataset,batch_size=cfg.batchsize,shuffle=True)

    model = STNE(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr = cfg.lr)

    for epoch in range(cfg.epc):
        pbar = tqdm(dataloader)
        pbar.set_description("[Epoch {}]".format(epoch))

        for seqs in pbar:
            decoder_out,loss = model.forward(embedding_W,seqs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=loss.item())

    for ratio in cfg.clf_ratio:
        micro,macro = node_classification(model,embedding_W,2*cfg.h_dim,cfg.node_seq,cfg.s_len,cfg.node_num,cfg.node_list,cfg.node_label,ratio)
        print('clf_ratio = ',ratio,' micro_f1 score = ',micro,' macro_f1 score = ',macro)


if __name__ == '__main__':
    main()



