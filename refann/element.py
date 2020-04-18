# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 01:17:28 2018

@author: Guojian Wang
"""

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

