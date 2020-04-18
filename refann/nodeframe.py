# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 01:26:19 2018

@author: Guojian Wang
"""

def decreasingNode(node_in=1970, node_out=5, hidden_layer=3, get_allNode=True):
    """A network structure that the number of neurons in each hidden layer is decreased proportionally
    
    Parameters
    ----------
    node_in : int, the number of nodes in the input layer
    node_out : int, the number of nodes in the output layer
    hidden_layer : int, the number of the hidden layers
    get_allNode : bool, if True, return the number of all nodes, otherwise, only return the number of nodes of hidden layers
    
    Return
    ------
    the number of nodes in each layer
    """
    decreasing_factor = (node_in*1.0/node_out)**( 1.0/(hidden_layer+1) )
    nodes = []
    for i in range(hidden_layer):
        nodes.append(int(round(node_in/decreasing_factor**(i+1))))
    nodes = tuple(nodes)
    if get_allNode:
        nodes = tuple([node_in])+nodes+tuple([node_out])
    return list(nodes)

def triangleNode(hidden1_node=8, hidden_layer=5):
    """A neural network structure that the number of neurons in each hidden layer is increase first and then decrease,
    the number of nodes in the hidden layers is symmetrical.
    
    Parameters
    ----------
    hidden1_node: int, the number of nodes in the first hidden layer
    hiddel_layer: int, the number of hidden layers

    Return
    ------
    the number of nodes in each layer
    """
    nodes_1 = []
    nodes_2 = []
    if hidden_layer<2:
        nodes_1.append(hidden1_node)
        nodes = nodes_1
    else:
        midLayer = (hidden_layer+1)/2.
        midLayer_f = (hidden_layer+1)//2
        for i in range(midLayer_f):
            if i<=midLayer:
                nodes_1.append(2**i * hidden1_node)
        if midLayer==midLayer_f:
            nodes_2 = sorted(nodes_1, reverse=True)[1:]
        else:
            nodes_2 = sorted(nodes_1, reverse=True)
        nodes = nodes_1 + nodes_2
    return nodes

def triangleNode_1(node_in=1, node_mid=1024, node_out=2, hidden_layer=5):
    """The same as triangleNode, but the number of nodes in the hidden layers is
    increased proportionally and then decreased proportionally
    
    Parameters
    ----------
    node_in: the number of nodes in the input layer
    node_out: the number of nodes in the output layer
    node_mid: the number of nodes in the middle (hidden) layer
    hiddel_layer: the number of hidden layers

    Return
    ------
    the number of nodes in each layer
    """
    if hidden_layer<2:
        nodes = [node_in, node_mid, node_out]
    else:
        midLayer = (hidden_layer+1)/2.
        midLayer_f = (hidden_layer+1)//2
        nodes_1 = decreasingNode(node_in=node_in,node_out=node_mid,hidden_layer=midLayer_f-1, get_allNode=True)
        if midLayer==midLayer_f:
            nodes_2 = sorted(nodes_1, reverse=True)[1:-1]
        else:
            nodes_2 = sorted(nodes_1, reverse=True)[:-1]
        nodes = nodes_1 + nodes_2 + [node_out]
    return nodes

