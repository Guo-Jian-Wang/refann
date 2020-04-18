# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 02:04:07 2018

@author: Guojian Wang
"""

import numpy as np
import torch


def numpy2torch(data):
    dtype = torch.FloatTensor
    data = torch.from_numpy(data).type(dtype)
    return data

def numpy2cuda(data, device=None):
    if device is None:
        dtype = torch.cuda.FloatTensor
        data = torch.from_numpy(data).type(dtype)
    else:
        data = numpy2torch(data)
        data = torch2cuda(data, device=device)
    return data

def torch2cuda(data, device=None):
    return data.cuda(device=device)

def torch2numpy(data):
    return data.numpy()

def cuda2torch(data):
    return data.cpu()

def cuda2numpy(data):
    return data.cpu().numpy()


class Normalize(object):
    """ Normalize data """
    def __init__(self, x, statistic={}, norm_type='z_score'):
        self.x = x
        self.stati = statistic
        self.norm_type = norm_type
    
    def minmax(self):
        """min-max normalization
        
        Rescaling the range of features to scale the range in [0, 1] or [a,b]
        https://en.wikipedia.org/wiki/Feature_scaling
        """
        return (self.x-self.stati['min'])/(self.stati['max']-self.stati['min'])
    
    def mean(self):
        """ mean normalization """
        return (self.x-self.stati['mean'])/(self.stati['max']-self.stati['min'])
    
    def z_score(self):
        """ standardization/z-score/zero-mean normalization """
        return (self.x-self.stati['mean'])/self.stati['std']
    
    def norm(self):
        return eval('self.%s()'%self.norm_type)

class InverseNormalize(object):
    """ Inverse transformation of class Normalize """
    def __init__(self, x1, statistic={}, norm_type='z_score'):
        self.x = x1
        self.stati = statistic
        self.norm_type = norm_type
    
    def minmax(self):
        return self.x * (self.stati['max']-self.stati['min']) + self.stati['min']
    
    def mean(self):
        return self.x * (self.stati['max']-self.stati['min']) + self.stati['mean']
    
    def z_score(self):
        return self.x * self.stati['std'] + self.stati['mean']
    
    def inverseNorm(self):
        return eval('self.%s()'%self.norm_type)


class Statistic(object):
    """ Statistics of an array """
    def __init__(self, x):
        self.x = x
    
    @property
    def mean(self):
        return np.mean(self.x)
    
    @property
    def xmin(self):
        return np.min(self.x)
    
    @property
    def xmax(self):
        return np.max(self.x)
    
    @property
    def std(self):
        return np.std(self.x)
    
    def statistic(self):
        st = {'min' : float(self.xmin),
              'max' : float(self.xmax),
              'mean': float(self.mean),
              'std' : float(self.std),
              }
        return st
