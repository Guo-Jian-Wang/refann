# -*- coding: utf-8 -*-

# the models of hyperparameters

def models(key):
    """Hyperparameter models.
    
    Parameters
    ----------
    key : str
        Hyperparameter model that contains hyperparameters (such as activation function, batch normalization, dropout, etc.) used in the network.
        It can be 'rec_1' (no batch normalization) or 'rec_2' (with batch normalization).
    
    Returns
    -------
    object
        Hyperparameter model.
    
    """
    return eval(key)()

def nuisance_hp(hparams):
    """ The hyperparameters that could be set to deterministic values. """
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
