# -*- coding: utf-8 -*-

def decreasingNode(node_in=1970, node_out=5, hidden_layer=3, get_allNode=True):
    """A network structure that the number of neurons in each hidden layer is decreased proportionally.
    
    Parameters
    ----------
    node_in : int
        The number of nodes in the input layer.
    node_out : int
        The number of nodes in the output layer.
    hidden_layer : int
        The number of the hidden layers.
    get_allNode : bool
        If True, return the number of all nodes, otherwise, only return the number of nodes of hidden layers. Default: True
    
    Returns
    -------
    list
        A list that contains the number of nodes in each layer.
    """
    decreasing_factor = (node_in*1.0/node_out)**( 1.0/(hidden_layer+1) )
    nodes = []
    for i in range(hidden_layer):
        nodes.append(int(round(node_in/decreasing_factor**(i+1))))
    nodes = tuple(nodes)
    if get_allNode:
        nodes = tuple([node_in])+nodes+tuple([node_out])
    return list(nodes)

def triangleNode_1(node_in=1, node_mid=1024, node_out=2, hidden_layer=5):
    """A neural network structure that the number of neurons in each hidden layer is increased proportionally 
    and then decreased proportionally, the number of nodes in the hidden layers is symmetrical.
    
    Parameters
    ----------
    node_in : int
        The number of nodes in the input layer.
    node_out : int
        The number of nodes in the output layer.
    node_mid : int
        The number of nodes in the middle (hidden) layer.
    hidden_layer : int
        The number of the hidden layers.
    
    Returns
    -------
    list
        A list that contains the number of nodes in each layer.
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
