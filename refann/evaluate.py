# -*- coding: utf-8 -*-

from . import data_process as dp
from torch.autograd import Variable
import numpy  as np
import matplotlib.pyplot as plt


def predict(net, inputs, use_GPU=False, in_type='numpy'):
    """Make predictions using a well-trained network.
    
    Parameters
    ----------
    inputs : numpy array or torch tensor
        The inputs of the network.
    use_GPU : bool
            If True, calculate using GPU, otherwise, calculate using CPU.
    in_type : str
        The data type of the inputs, it can be 'numpy' or 'torch'.
    """
    if use_GPU:
        net = net.cuda()
        if in_type=='numpy':
            inputs = dp.numpy2cuda(inputs)
        elif in_type=='torch':
            inputs = dp.torch2cuda(inputs)
    else:
        if in_type=='numpy':
            inputs = dp.numpy2torch(inputs)
    net = net.eval() #this works for the batch normalization layers
    pred = net(Variable(inputs))
    if use_GPU:
        pred = dp.cuda2numpy(pred.data)
    else:
        pred = dp.torch2numpy(pred.data)
    return pred

def plot_loss(loss):
#    print ('The last 5 losses: ', np.array(loss[-5:]))
    print ('The average of last 100 losses: %.8f\n'%(np.mean(loss[-100:])))
    plt.figure(figsize=(6*2., 4.5*1.))
    plt.subplot(1,2,1)
    plt.semilogx(loss)
    plt.xlabel('Iteration', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    
    plt.subplot(1,2,2)
    plt.loglog(loss)
    plt.xlabel('Iteration', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
