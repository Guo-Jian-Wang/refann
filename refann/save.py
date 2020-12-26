# -*- coding: utf-8 -*-

import numpy as np
import os


def mkdir(path):
    """Make a directory in a particular location if it is not exists, otherwise, do nothing.
    
    Parameters
    ----------
    path : str
        The path of a file.
    
    Examples
    --------
    >>> mkdir('/home/UserName/test')
    >>> mkdir('test/one')
    >>> mkdir('../test/one')
    """
    #remove the blank space in the before and after strings
    #path.strip() is used to remove the characters in the beginning and the end of the character string
#    path = path.strip()
    #remove all blank space in the strings, there is no need to use path.strip() when using this command
    path = path.replace(' ', '')
    #path.rstrip() is used to remove the characters in the right of the characters strings
    
    if path=='':
        raise ValueError('The path cannot be an empty string')
    path = path.rstrip("/")
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        print('The directory "%s" is successfully created !'%path)
        return True
    else:
#        print('The directory "%s" is already exists!'%path)
#        return False
        pass
 
def savetxt(path, FileName, File):
    """Save the .txt files using :func:`numpy.savetxt()` funtion.

    Parameters
    ----------
    path : str
        The path of the file to be saved.
    FileName : str
        The name of the file to be saved.
    File : object
        The file to be saved.
    """
    mkdir(path)
    np.savetxt(path + '/' + FileName + '.txt', File)

def savenpy(path, FileName, File, dtype=np.float32):
    """Save an array to a binary file in .npy format using :func:`numpy.save()` function.
    
    Parameters
    ----------
    path : str
        The path of the file to be saved.
    FileName : str
        The name of the file to be saved.
    File : object
        The file to be saved.
    dtype : str or object
        The type of the data to be saved. Default: ``numpy.float32``.
    """
    mkdir(path)
    File = File.astype(dtype)
    np.save(path + '/' + FileName + '.npy', File)
