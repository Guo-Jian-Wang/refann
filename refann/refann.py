# -*- coding: utf-8 -*-

from . import data_process as dp
from . import train, evaluate, fcnet, hpmodel, nodeframe, save
import torch
import numpy as np
import matplotlib.pyplot as plt


class ANN(train.Train):
    """Reconstruct functions with Artificial Neural Network.
    
    Parameters
    ----------
    data : array-like
        An array with shape of (N, 3), where N is the number of data points, each column represents :math:`X, Y, \sigma_Y`.
    hidden_layer : int, optional
        The number of hidden layers. Default: 1
    mid_node : int, optional
        The number of nodes (or neurons) of the middle layer. Default: 4096
    hp_model : int, optional
        The hyperparameter models, 'rec_1' (no batch normalization) or 'rec_2' (with batch normalization). Default: 'rec_1'
    loss_func : str, optional
        The loss function, there are three loss functions in this code, L1Loss ('L1'), 
        MSELoss ('MSE'), and SmoothL1Loss ('SmoothL1'). Default: 'L1'
    
    Attributes
    ----------
    lr : float, optional
        The learning rate. Default: 1e-1
    lr_min : float, optional
        The minimum of the learning rate. Default: 1e-8
    iteration : int, optional
        The number of iterations. Default: 30000
    batch_size_max : int, optional
        The maximum of the batch size. Defalt: 300
    scale_inputs : bool, optional
        If True, the inputs data will be normalized, otherwise, do nothing. Defalt: True
    scale_target : bool, optional
        If True, the outputs (or target) data will be normalized, otherwise, do nothing. Defalt: True
    scale_type : str, optional
        The normalization method, 'minmax', 'mean', or 'z_score'. Default: 'z_score'
    fix_initialize : bool, optional
        If True, the network will be initialized from a specific seed. Default: True
    print_info : bool, optional
        If True, some information about the training process will be printed. Default: True
    
    Note
    ----
    Hyperparameters of the ANN, such as the number of hidden layers (``hidden_layer``), 
    the number of neurons (``mid_node``), hyperparameter model (``hp_model``), should be optimized 
    before reconstructing functions from data. See https://doi.org/10.3847/1538-4365/ab620b for details.
    """        
    def __init__(self,data,hidden_layer=1,mid_node=4096,hp_model='rec_1',loss_func='L1'):
        self.data = data
        self.inputs = np.reshape(data[:,0],(-1,1))
        self.target = data[:,1:]
        self.mid_node = mid_node
        self.hidden_layer = hidden_layer
        self.hp_model = hp_model
        self.lr = 1e-1
        self.lr_min = 1e-8
        self.iteration = 30000
        self.batch_size_max = 300
        self.batch_size = self._batch_size()
        self.loss_func = train.loss_funcs(name=loss_func)
        self.scale_inputs = True
        self.scale_target = True
        self.scale_type = 'z_score'
        self.fix_initialize = True
        self.print_info = True
    
    def _nodes(self):
        return nodeframe.triangleNode_1(node_in=len(self.inputs[0]),node_mid=self.mid_node,node_out=len(self.target[0]),hidden_layer=self.hidden_layer)
    
    def _hparams(self):
        return hpmodel.models(self.hp_model)
    
    def _net(self):
        if self.fix_initialize:
            torch.manual_seed(1000) # Fixed parameter initialization
        self.nodes = self._nodes()
        self.hparams = self._hparams()
        self.net = fcnet.get_FcNet(nodes=self.nodes, hparams=self.hparams)
        if self.print_info:
            print(self.net)

    def _batch_size(self):
        num_train = len(self.data)
        if num_train//2 < self.batch_size_max:
            bs = num_train//2
        else:
            bs = self.batch_size_max
        return bs
    
    def statistic(self):
        self.inputs_statistic = dp.Statistic(self.inputs).statistic()
        self.target_statistic = dp.Statistic(self.target).statistic()

    def train(self):
        self._net()
        self.transfer_net(prints=self.print_info)
        
        self.optimizer = self._optimizer(name='Adam')
        
        self.statistic()
        self.transfer_data()
        if self.scale_inputs:
            self.inputs = dp.Normalize(self.inputs, self.inputs_statistic, norm_type=self.scale_type).norm()
        if self.scale_target:
            self.target = dp.Normalize(self.target, self.target_statistic, norm_type=self.scale_type).norm()
        
        self.net, self.loss = self.train_1(inputs=self.inputs,target=self.target,repeat_n=1,set_seed=True,lr_decay=True,print_info=self.print_info,showIter_n=2000)
        self.net = self.net.cpu()
        return self.net, self.loss

    def predict(self, xpoint=None, xspace=None):
        """Prediction
        
        Parameters
        ----------
        xpoint : array-like or None, optional
            An array of :math:`X` points. Default: None
        xspace : tuple, list, or None, optional
            If not None, xpoint will be ignored, and it should be (xmin, xmax, npoint) or [xmin, xmax, npoint]. Default: None

        Returns
        -------
        array-like
            The reconstructed function.

        """
        if xpoint is None:
            if xspace is None:
                xpoint = np.linspace(min(self.data[:,0]), max(self.data[:,0]), 100)
            else:
                xpoint = np.linspace(xspace[0], xspace[1], xspace[2])
        x = np.copy(xpoint)
        if self.scale_inputs:
            xpoint = dp.Normalize(xpoint, self.inputs_statistic, norm_type=self.scale_type).norm()
        y = evaluate.predict(self.net, np.reshape(xpoint, (-1,1)), use_GPU=self.use_GPU)##
        if self.scale_target:
            y = dp.InverseNormalize(y, self.target_statistic, norm_type=self.scale_type).inverseNorm()
        self.func = np.c_[x, y]
        return self.func
    
    def save_net(self, path='func', obsName='Hz'):
        path = path + '/nn'
        save.mkdir(path)
        full_path = path + '/ANN-%s_nodes%s.pt'%(obsName, str(self.nodes))
        torch.save(self.net, full_path)
    
    def _save_loss(self, path='func', obsName='Hz'):
        path = path + '/nn'
        fileName = 'ANN_loss-%s_nodes%s'%(obsName, str(self.nodes))
        save.savenpy(path, fileName, self.loss)
    
    def save_func(self, path='func', obsName='Hz', file_type='npy'):
        fileName = 'ANN_%s_nodes%s'%(obsName, str(self.nodes))
        if file_type=='txt':
            save.savetxt(path, fileName, self.func)
        elif file_type=='npy':
            save.savenpy(path, fileName, self.func)
    
    def plot_loss(self):
        evaluate.plot_loss(self.loss)
    
    def plot_func(self):
        plt.figure(figsize=(8, 6))
        plt.errorbar(self.data[:,0], self.data[:,1], yerr=self.data[:,2], fmt='ro', alpha=1, label='Observational data')
        plt.plot(self.func[:,0], self.func[:,1], 'k', label=r'$\rm Predicted\ f(x)$', lw=2)
        plt.fill_between(self.func[:,0], self.func[:,1]-self.func[:,2], self.func[:,1]+self.func[:,2],
                         label=r'$\rm\sigma_{f(x)}$', color='g', alpha=0.5)
        plt.xlabel('x', fontsize=16)
        plt.ylabel('f(x)', fontsize=16)
        plt.legend(fontsize=16)

class OptimizeANN(object):
    def __init__(self,truth,hidden_layers=[1,2,3],mid_nodes=[128,256]):
        self.truth = truth
        self.mid_nodes = mid_nodes
        self.hidden_layers = hidden_layers
    
    def _file_name(self, obsName, hidden_layer, mid_node):
        nodes = nodeframe.triangleNode_1(node_in=1,node_mid=mid_node,node_out=2,hidden_layer=hidden_layer)
        return 'ANN_%s_nodes%s'%(obsName, nodes)
    
    def rss(self, predict):
        return sum((self.truth - predict)**2)

    def risk(self, predict, error):
        return round(self.rss(predict) + sum(error**2), 3)
    
    def get_risk(self,path='',obsName='Hz',file_type='npy'):
        self.risks = {}
        self.risk_mean = {}
        for hidden_layer in self.hidden_layers:
            rsk_1 = []
            for mid_node in self.mid_nodes:
                file_name = self._file_name(obsName,hidden_layer,mid_node)
                if file_type=='txt':
                    f = np.loadtxt('%s/%s.txt'%(path,file_name))
                elif file_type=='npy':
                    f = np.load('%s/%s.npy'%(path,file_name))
                rsk_1.append(self.risk(f[:,1], f[:,2]))
            self.risks[str(hidden_layer)] = rsk_1
            self.risk_mean[str(hidden_layer)] = round(np.mean(rsk_1), 3)
    
    def get_optimal(self):
        self.optimal_layer = min(self.risk_mean, key=self.risk_mean.get)
        minRisk_index = self.risks[self.optimal_layer].index(min(self.risks[self.optimal_layer]))
        self.optimal_node = self.mid_nodes[minRisk_index]
        self.optimal_risk = self.risks[self.optimal_layer][minRisk_index]
    
    def plot_risk(self):
        self.get_optimal()
        plt.figure(figsize=(6*1.2*2,4.5*1.2))
        plt.subplot(1,2,1)
        for hidden_layer in self.hidden_layers:
            plt.semilogx(self.mid_nodes, self.risks[str(hidden_layer)], '-o', label=r'$\rm Hidden\ layer: %s$'%hidden_layer, lw=2)
            plt.title(r'$\rm Optimal\ layer: %s$'%self.optimal_layer, fontsize=16)
            plt.legend(fontsize=16)
        plt.xlabel('Number of neurons', fontsize=16)
        plt.ylabel('Risk', fontsize=16)
        plt.subplot(1,2,2)
        plt.semilogx(self.mid_nodes, self.risks[self.optimal_layer], '-o', label=r'$\rm Hidden\ layer: %s$'%self.optimal_layer, lw=2)
        plt.title(r'$\rm Optimal\ node: %s$'%self.optimal_node, fontsize=16)
        plt.legend(fontsize=16)
        plt.xlabel('Number of neurons', fontsize=16)
        plt.ylabel('Risk', fontsize=16)

class RePredictANN(ANN):
    """Reconstruct function using the saved well-trained networks
    
    Parameters
    ----------
    data : array-like
        An array with shape of (N, 3), where N is the number of data points, each column represents :math:`X, Y, \sigma_Y`.
    hidden_layer : int, optional
        The number of hidden layers. Default: 1
    mid_node : int, optional
        The number of nodes (or neurons) of the middle layer. Default: 4096
    
    Attributes
    ----------
    scale_inputs : bool, optional
        If True, the inputs data will be normalized, otherwise, do nothing. Defalt: True
    scale_target : bool, optional
        If True, the outputs (or target) data will be normalized, otherwise, do nothing. Defalt: True
    scale_type : str, optional
        The normalization method, 'minmax', 'mean', or 'z_score'. Default: 'z_score'
    """
    def __init__(self,data,hidden_layer=1,mid_node=4096):
        self.data = data
        self.inputs = np.reshape(data[:,0],(-1,1))
        self.target = data[:,1:]
        self.mid_node = mid_node
        self.hidden_layer = hidden_layer
        self.nodes = self._nodes()
        self.scale_inputs = True
        self.scale_target = True
        self.scale_type = 'z_score'
    
    def load_net(self, path='func', obsName='Hz'):
        full_path = path + '/nn/ANN-%s_nodes%s.pt'%(obsName, str(self.nodes))
        self.net = torch.load(full_path)
        self.statistic()
