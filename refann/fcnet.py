# -*- coding: utf-8 -*-

from . import sequence as seq
from . import nodeframe, hpmodel
import torch


class FcNet(torch.nn.Module):
    def __init__(self, nodes, mainActive='relu', finalActive='None', mainBN=False,
                 finalBN=False, mainDropout='None', finalDropout='None'):
        super(FcNet, self).__init__()
        self.fc = seq.LinearSeq(nodes,mainActive=mainActive,finalActive=finalActive,mainBN=mainBN,
                                finalBN=finalBN,mainDropout=mainDropout,finalDropout=finalDropout).get_seq()
        
    def forward(self, x):
        x = self.fc(x)
        return x

def get_FcNet(node_in=2000, node_out=6, hidden_layer=3, nodes=None, hparams={}):
    """Get a fully connected network.
    
    Parameters
    ----------
    node_in : int
        The number of the input nodes.
    node_out : int
        The number of the output nodes.
    hidden_layer : int
        The number of the hidden layers.
    nodes : None or list
            If list, it should be a collection of nodes of the network, e.g. [node_in, node_hidden1, node_hidden2, ..., node_out]
    hparams : dict
        A dictionary of hyperparameters (or hidden parameters, such as the activation function,
        the batch normalization, and the dropout) of the netwowrk. See :func:`~.hpmodel.models`.
    """
    if nodes is None:
        nodes = nodeframe.decreasingNode(node_in=node_in, node_out=node_out, hidden_layer=hidden_layer, get_allNode=True)
    
    if hparams:
        hp = hparams
    else:
        hp = hpmodel.models('default')
        print ('Using the default hyperparameters, you can also select another set of hyperparameters \n')
    
    hp = hpmodel.nuisance_hp(hp)
    net = FcNet(nodes,mainActive=hp['active'],finalActive=hp['finalActive'],mainBN=hp['BN'],
                finalBN=hp['finalBN'],mainDropout=hp['dropout'],finalDropout=hp['finalDropout'])
#    print ('Nodes: %s'%nodes)
#    print ('Network: %s'%net)
    return net
