# -*- coding: utf-8 -*-

import torch.nn as nn


#%% activation functions
def relu():
    #here 'inplace=True' is used to save GPU memory
    return nn.ReLU(inplace=True)

def leakyrelu():
    return nn.LeakyReLU(inplace=True)

def prelu():
    return nn.PReLU()

def rrelu():
    return nn.RReLU(inplace=True)

def elu():
    return nn.ELU(inplace=True)

def activation(active_name='relu'):
    return eval('%s()'%active_name)
