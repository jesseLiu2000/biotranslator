import itertools
import numpy as np
from torch import nn
from torch.nn import init
from transformers import AutoTokenizer, AutoModel
import torch
import collections


# BioTranslator Model
class ProteinEncoder(nn.Module):

    def __init__(self,
                 hidden_dim=1000,
                 seq_input_nc=21,
                 seq_in_nc=512,
                 seq_max_kernels=129,
                 seq_length=2000):
        """

        :param seq_input_nc:
        :param seq_in_nc:
        :param seq_max_kernels:
        :param dense_num:
        :param seq_length:
        """
        super(ProteinEncoder, self).__init__()

        self.para_conv, self.para_pooling = [], []
        kernels = range(8, seq_max_kernels, 8)
        self.kernel_num = len(kernels)
        for i in range(len(kernels)):
            exec("self.conv1d_{} = nn.Conv1d(in_channels=seq_input_nc, out_channels=seq_in_nc, kernel_size=kernels[i], padding=0, stride=1)".format(i))
            exec("self.pool1d_{} = nn.MaxPool1d(kernel_size=seq_length - kernels[i] + 1, stride=1)".format(i))
        self.fc_seq =[nn.Linear(len(kernels)*seq_in_nc, hidden_dim), nn.LeakyReLU(inplace=True)]
        self.fc_seq = nn.Sequential(*self.fc_seq)

    def forward(self, x=None):
        x_list = []
        for i in range(self.kernel_num):
            exec("x_i = self.conv1d_{}(x)".format(i))
            exec("x_i = self.pool1d_{}(x_i)".format(i))
            exec("x_list.append(torch.squeeze(x_i).reshape([x.size(0), -1]))")
        x_enc = self.fc_seq(torch.cat(tuple(x_list), dim=1))

        return x_enc