# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 01:44:24 2018

@author: Guojian Wang
"""
from . import element
import torch.nn as nn

class SeqName(object):
    def __init__(self, module_name):
        """ The name of sequence, to be used by class LinearSeq """
        self.moduleName = module_name
    
    def seq_name(self):
        self.moduleName = str(eval(self.moduleName)+1)
        return self.moduleName

class BatchNorm(object):
    """ Batch Normalization, to be used by class LinearSeq """
    def _batchnorm1d(self, name, n_output):
        self.seq.add_module(name, nn.BatchNorm1d(n_output, eps=self.eps, momentum=self.momentum))

class Activation(object):
    """ Activation functions, to be used by class LinearSeq """
    def _activation(self, module_name, active_name):
        self.seq.add_module(module_name, element.activation(active_name=active_name))

class Pooling(object):
    """ Pooling, to be used by class LinearSeq """
    def _pooling(self, module_name, pool_name):
        self.seq.add_module(module_name, element.pooling(pool_name=pool_name))

class Dropout(object):
    """ Dropout, to be used by class LinearSeq """
    def _dropout(self, module_name, dropout_name):
        self.seq.add_module(module_name, element.get_dropout(dropout_name))


class LinearSeq(SeqName,BatchNorm,Activation,Dropout):
    """ sequence of Linear """
    def __init__(self, nodes, mainBN=True, finalBN=False, mainActive='relu',
                 finalActive='None', mainDropout='None', finalDropout='None'):
        SeqName.__init__(self, '-1') #or super(LinearSeq, self).__init__('-1')
        self.nodes = nodes
        self.layers = len(nodes) - 1
        self.mainBN = mainBN
        self.finalBN = finalBN
        self.mainActive = mainActive
        self.finalActive = finalActive
        self.mainDropout = mainDropout
        self.finalDropout = finalDropout
        self.eps = 1e-05
        self.momentum = 0.1
        self.seq = nn.Sequential()

    def __linear(self, name, n_input, n_output):
        self.seq.add_module(name, nn.Linear(n_input, n_output))
    
    def get_seq(self):
        for i in range(self.layers-1):
            self.__linear(self.seq_name(), self.nodes[i], self.nodes[i+1])
            if self.mainBN:
                self._batchnorm1d(self.seq_name(), self.nodes[i+1])
            if self.mainActive!='None':
                self._activation(self.seq_name(), self.mainActive)
            if self.mainDropout!='None':
                self._dropout(self.seq_name(), self.mainDropout)
        
        self.__linear(self.seq_name(), self.nodes[-2], self.nodes[-1])
        if self.finalBN:
            self._batchnorm1d(self.seq_name(), self.nodes[-1])
        if self.finalActive!='None':
            self._activation(self.seq_name(), self.finalActive)
        if self.finalDropout!='None':
            self._dropout(self.seq_name(), self.finalDropout)
        return self.seq

