# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 14:07:37 2019

@author: Guojian Wang
"""

# the models of hyperparameters


def models(key):
    return eval(key)()

def nuisance_hp(hparams):
    """ the hyperparameters that could be set to deterministic values """
    nhp = {'finalActive' : 'None',
           'finalBN' : False,
           'finalDropout' : 'None'}
    for key in nhp.keys():
        if key not in hparams.keys():
            hparams[key] = nhp[key]
    return hparams

#%% this is used for reconstructing functions from data
def rec_1():
    return {'active' : 'elu',
            'BN' : False,
            'dropout' : 'None'
            }

def rec_2():
    return {'active' : 'elu',
            'BN' : True,
            'dropout' : 'None'
            }


#%% default
def default():
    return {'active' : 'relu',
            'BN' : True,
            'dropout' : 'None'
            }

