# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel,AutoModel,RoFormerModel


class BERTBinding_TP_cnn(nn.Module):
    def __init__(self, beta_dir, alpha_dir, antigen_dir, emb_dim=256):
        super().__init__()
        self.BetaModel = AutoModel.from_pretrained(beta_dir, output_hidden_states=True, return_dict=True)
        self.AlphaModel = AutoModel.from_pretrained(alpha_dir, output_hidden_states=True, return_dict=True)
        self.AntigenModel = AutoModel.from_pretrained(antigen_dir, output_hidden_states=True, return_dict=True,cache_dir = "../cache")

        self.cnn1 = MF_CNN(in_channel= 120)
        self.cnn2 = MF_CNN(in_channel = 120)
        self.cnn3 = MF_CNN(in_channel = 12,hidden_size=76)#56)

        self.binding_predict = nn.Sequential(
            nn.Linear(in_features=128 * 3, out_features=emb_dim),
            nn.ReLU(),
            nn.Linear(in_features=emb_dim, out_features=emb_dim),
            nn.ReLU(),
            nn.Linear(in_features=emb_dim, out_features=1)
        )

    def forward(self, beta, alpha, antigen):
        beta_encoded = self.BetaModel(**beta).last_hidden_state
        alpha_encoded = self.AlphaModel(**alpha).last_hidden_state
        antigen_encoded = self.AntigenModel(**antigen).last_hidden_state

        beta_cls = self.cnn1(beta_encoded)
        alpha_cls = self.cnn2(alpha_encoded)
        antigen_cls = self.cnn3(antigen_encoded)
        
        concated_encoded = torch.concat((beta_cls,alpha_cls,antigen_cls) , dim = 1)


        output = self.binding_predict(concated_encoded)

        return output


class BERTBinding_TP_cnn_4_input(nn.Module):
    def __init__(self, beta_dir, alpha_dir, antigen_dir, hla_dir, emb_dim=256):
        super().__init__()
        self.BetaModel = AutoModel.from_pretrained(beta_dir, output_hidden_states=True, return_dict=True)
        self.AlphaModel = AutoModel.from_pretrained(alpha_dir, output_hidden_states=True, return_dict=True)
        self.AntigenModel = AutoModel.from_pretrained(antigen_dir, output_hidden_states=True, return_dict=True,
                                                      cache_dir="../cache")
        self.HLAModel = AutoModel.from_pretrained(hla_dir, output_hidden_states=True, return_dict=True,
                                                  cache_dir="../cache")

        self.cnn1 = MF_CNN(in_channel=120)
        self.cnn2 = MF_CNN(in_channel=120)
        self.cnn3 = MF_CNN(in_channel=12, hidden_size=76)  # 56)
        self.cnn4 = MF_CNN(in_channel=34, hidden_size=76)

        self.binding_predict = nn.Sequential(
            nn.Linear(in_features=128 * 4, out_features=emb_dim),
            nn.ReLU(),
            nn.Linear(in_features=emb_dim, out_features=emb_dim),
            nn.ReLU(),
            nn.Linear(in_features=emb_dim, out_features=1)
        )

    def forward(self, beta, alpha, antigen, hla):
        beta_encoded = self.BetaModel(**beta).last_hidden_state
        alpha_encoded = self.AlphaModel(**alpha).last_hidden_state
        antigen_encoded = self.AntigenModel(**antigen).last_hidden_state
        hla_encoded = self.HLAModel(**hla).last_hidden_state

        beta_cls = self.cnn1(beta_encoded)
        alpha_cls = self.cnn2(alpha_encoded)
        antigen_cls = self.cnn3(antigen_encoded)
        hla_cls = self.cnn4(hla_encoded)

        concated_encoded = torch.concat((beta_cls, alpha_cls, hla_cls, antigen_cls), dim=1)

        output = self.binding_predict(concated_encoded)

        return output


class MF_CNN(nn.Module):
    def __init__(self, in_channel=118,emb_size = 20,hidden_size = 92):#189):
        super(MF_CNN, self).__init__()
        
        # self.emb = nn.Embedding(emb_size,128)  # 20*128
        self.conv1 = cnn_block(in_channel = in_channel,hidden_channel = 64)   # 118*64
        self.conv2 = cnn_block(in_channel = 64,hidden_channel = 32) # 64*32

        self.conv3 = cnn_block(in_channel = 32,hidden_channel = 32)

        self.fc1 = nn.Linear(32*hidden_size , 128) # 32*29*512
        self.fc2 = nn.Linear(128 , 128)

        self.fc3 = nn.Linear(128 , 128)

    def forward(self, x):
        #x = x
        # x = self.emb(x)
        
        x = self.conv1(x)
        
        x = self.conv2(x)

        x = self.conv3(x)
        
        x = x.view(x.shape[0] ,-1)
        
        x = nn.ReLU()(self.fc1(x))
        sk = x
        x = self.fc2(x)

        x = self.fc3(x)
        return x +sk




class cnn_block(nn.Module):
    def __init__(self, in_channel=2, hidden_channel=2, out_channel=2):
        super(cnn_block, self).__init__()
        
        self.cnn = nn.Conv1d(in_channel , hidden_channel , kernel_size = 5 , stride = 1) # bs * 64*60
        self.max_pool = nn.MaxPool1d(kernel_size = 2 , stride=2)# bs * 32*30
                               
        self.relu = nn.ReLU()
    
    def forward(self, x):

        x = self.cnn(x)
        x = self.max_pool(x)
        x = self.relu(x)
        return x