# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 11:51:30 2018

@author: Guojian Wang
"""

import math


class LrDecay:
    """ Let the learning rate decay with iteration """
    def __init__(self, iter_mid, iteration=10000, lr=0.1, lr_min=1e-6):
        self.lr = lr
        self.lr_min = lr_min
        self.iter_mid = iter_mid
        self.iteration = iteration

    def exp(self, gamma=0.999, auto_params=True):
        """exponential decay
        
        Parameters
        ----------
        auto_params : bool, if True, gamma is set automatically
        
        Return
        ------
        lr * gamma^iteration
        """
        if auto_params:
            gamma = (self.lr_min/self.lr)**(1./self.iteration)
        lr_new = self.lr * gamma**self.iter_mid
        return lr_new
    
    def step(self, stepsize=1000, gamma=0.3, auto_params=True):
        """let the learning rate decays step by step, similar to 'exp'
        
        auto_params : bool, if True, gamma is set automatically
        """
        if auto_params:
            gamma = (self.lr_min/self.lr)**(1./(self.iteration*1.0/stepsize))
        lr_new = self.lr * gamma**(math.floor(self.iter_mid*1.0/stepsize))
        return lr_new
    
    def poly(self, decay_step=500, power=0.999, cycle=True):
        """polynomial decay, 
        
        Parameters
        ----------
        
        Return
        ------
        (lr-lr_min) * (1 - iteration/decay_steps)^power +lr_min
        """
        if cycle:
            decay_steps = decay_step * math.ceil(self.iter_mid*1.0/decay_step)
        else:
            decay_steps = self.iteration
        lr_new = (self.lr-self.lr_min) * (1 - self.iter_mid*1.0/decay_steps)**power + self.lr_min
        return lr_new

