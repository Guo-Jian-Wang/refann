# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 14:00:35 2019

@author: Guojian Wang
"""
from . import optimize
from . import data_process as dp
import torch
from torch.autograd import Variable
import numpy as np


def loss_funcs(name='L1'):
    if name=='L1':
        lf = torch.nn.L1Loss()
    elif name=='MSE':
        lf = torch.nn.MSELoss()
    elif name=='SmoothL1':
        lf = torch.nn.SmoothL1Loss()
    return lf

class Train(object):
    def __init__(self,net,loss_func='L1',iteration=10000,optimizer='Adam'):
        self.net = net
        self.loss_func = loss_funcs(name=loss_func)
        self.iteration = iteration
        self.lr = 1e-1
        self.lr_min = 1e-6
        self.batch_size = 128
        self.optimizer = self._optimizer(name=optimizer)
    
    def _prints(self, items, prints=True):
        if prints:
            print(items)
    
    def call_GPU(self, prints=True):
        if torch.cuda.is_available():
            self.use_GPU = True
            gpu_num = torch.cuda.device_count()
            if gpu_num > 1:
                self.use_multiGPU = True
                self._prints('\nTraining the network using {} GPUs'.format(gpu_num), prints=prints)
            else:
                self.use_multiGPU = False
                self._prints('\nTraining the network using 1 GPU', prints=prints)
        else:
            self.use_GPU = False
            self._prints('\nTraining the network using CPU', prints=prints)
    
    def transfer_net(self, use_DDP=False, device_ids=None, prints=True):
        if device_ids is None:
            device = None
        else:
            device = device_ids[0]
        self.call_GPU(prints=prints)
        if self.use_GPU:
            self.net = self.net.cuda(device=device)
            if self.use_multiGPU:
                if use_DDP:
                    self.net = torch.nn.parallel.DistributedDataParallel(self.net, device_ids=device_ids)
                else:
                    self.net = torch.nn.DataParallel(self.net, device_ids=device_ids)
    
    def transfer_data(self, device=None):
        if self.use_GPU:
            self.inputs = dp.numpy2cuda(self.inputs, device=device)
            self.target = dp.numpy2cuda(self.target, device=device)
        else:
            self.inputs = dp.numpy2torch(self.inputs)
            self.target = dp.numpy2torch(self.target)
    
    def _optimizer(self, name='Adam'):
        if name=='Adam':
            _optim = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        return _optim
    
    def train_0(self, xx, yy, iter_mid, repeat_n=3, lr_decay=True):
        xx = Variable(xx)
        yy = Variable(yy, requires_grad=False)
        for t in range(repeat_n):
            _predicted = self.net(xx)
            _loss = self.loss_func(_predicted, yy)
            
            self.optimizer.zero_grad()
            _loss.backward()
            self.optimizer.step()
        
        if lr_decay:
            #reduce the learning rate
            lrdc = optimize.LrDecay(iter_mid,iteration=self.iteration,lr=self.lr,lr_min=self.lr_min)
            self.optimizer.param_groups[0]['lr'] = lrdc.exp()
        return _loss.item(), _predicted.data

    def train_1(self, inputs, target, repeat_n=1, set_seed=False, lr_decay=True,
                print_info=True, showIter_n=200):
        if self.batch_size > len(inputs):
            raise ValueError('The batch size should be smaller than the number of the training set')
            
        if set_seed:
            np.random.seed(1000)#
        loss_all = []
        for iter_mid in range(1, self.iteration+1):
            batch_index = np.random.choice(len(inputs), self.batch_size, replace=False)#Note: replace=False
            xx = inputs[batch_index]
            yy = target[batch_index]
            
            _loss, _ = self.train_0(xx, yy, iter_mid, repeat_n=repeat_n, lr_decay=lr_decay)
            loss_all.append(_loss)
            
            if print_info:
                if iter_mid%showIter_n==0:
                    print('(iteration:%s/%s; loss:%.5f; lr:%.8f)'%(iter_mid, self.iteration, _loss, self.optimizer.param_groups[0]['lr']))
        return self.net, loss_all
