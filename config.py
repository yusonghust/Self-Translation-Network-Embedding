#-*-coding:utf-8-*-
import numpy as np
import torch
import os

class config():
    def __init__(self):
        self._configs = {}

        self._configs['dpt'] = 1 # Depth of both the encoder and the decoder layers (MultiCell RNN)
        self._configs['h_dim'] = 500 # Hidden dimension of encoder LSTMs
        self._configs['s_len'] = 10 # Length of input node sequence
        self._configs['epc'] = 5 # Number of training epochs
        self._configs['freeze'] = True #  Node features freeze or not
        self._configs['dropout'] = 0.5 # Dropout ration
        self._configs['clf_ratio'] = [0.1,0.2,0.3,0.4,0.5]
        self._configs['batchsize'] = 128
        self._configs['lr'] = 0.001

        self._configs['node_label_file'] = './data/cora/labels.txt'
        self._configs['node_fea_file'] = './data/cora/cora.features'
        self._configs['node_seq_file'] = './data/cora/node_sequences_10_10.txt'

        self._configs['node_num'] = 0
        self._configs['fea_dim'] = 0
        self._configs['node_fea'] = None
        self._configs['node_seq'] = None
        self._configs['node_list'] = None
        self._configs['node_label'] = None


    @property
    def dpt(self):
        return self._configs['dpt']

    @property
    def h_dim(self):
        return self._configs['h_dim']

    @property
    def s_len(self):
        return self._configs['s_len']

    @property
    def epc(self):
        return self._configs['epc']

    @property
    def freeze(self):
        return self._configs['freeze']

    @property
    def dropout(self):
        return self._configs['dropout']

    @property
    def clf_ratio(self):
        return self._configs['clf_ratio']

    @property
    def batchsize(self):
        return self._configs['batchsize']

    @property
    def lr(self):
        return self._configs['lr']

    @property
    def node_num(self):
        return self._configs['node_num']

    @property
    def fea_dim(self):
        return self._configs['fea_dim']

    @property
    def node_fea(self):
        return self._configs['node_fea']

    @property
    def node_seq(self):
        return self._configs['node_seq']

    @property
    def node_label_file(self):
        return self._configs['node_label_file']

    @property
    def node_fea_file(self):
        return self._configs['node_fea_file']

    @property
    def node_seq_file(self):
        return self._configs['node_seq_file']

    @property
    def node_list(self):
        return self._configs['node_list']

    @property
    def node_label(self):
        return self._configs['node_label']

    def update_config(self,key,value):
        if key in self._configs.keys():
            self._configs[key] = value
        else:
            raise RuntimeError('Update_Config_Error')