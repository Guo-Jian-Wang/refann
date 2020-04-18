# -*- coding: utf-8 -*-
"""
Created on Sun Aug 13 10:48:37 2017
2017 - 2017

@author: Guojian Wang
"""

import numpy as np
import os


def mkdir(path):
    """Make a directory in a particular location if it is not exists, otherwise, do nothing.
        
    Usage : mkdir('/home/jian/test'), mkdir('test/one') or mkdir('../test/one') 
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
    """Save the .txt files using np.savetxt() funtion
    
    path : the path of the file to be saved
    FileName : the name of the file to be saved
    File : the file to be saved
    """
    mkdir(path)
    np.savetxt(path + '/' + FileName + '.txt', File)

def savedat(path, FileName, File):
    """Save the .dat files using np.savetxt() funtion
    
    path : the path of the file to be saved
    FileName : the name of the file to be saved
    File : the file to be saved
    """
    mkdir(path)
    np.savetxt(path + '/' + FileName + '.dat', File)

def savenpy(path, FileName, File, dtype=np.float32):
    """Save an array to a binary file in NumPy .npy format using np.save() function
    
    path : the path of the file to be saved
    FileName : the name of the file to be saved
    File : the file to be saved
    """
    mkdir(path)
    File = File.astype(dtype)
    np.save(path + '/' + FileName + '.npy', File)

