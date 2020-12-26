# -*- coding: utf-8 -*-

import sys
sys.path.append('..')
import refann as rf
import numpy as np
import matplotlib.pyplot as plt
import time
start_time = time.time()


def func(x):
    return x * np.sin(x)

x = np.linspace(0, 10, 31)
y = func(x)
np.random.seed(1)
err_y = np.abs(np.random.randn(len(y)))

plt.figure(figsize=(8,6))
plt.plot(x, y, '-', label='$f(x)=x\ sin(x)$')
plt.errorbar(x, y, yerr=err_y, fmt='ro')
plt.xlabel('x', fontsize=16)
plt.ylabel('f(x)', fontsize=16)
plt.legend(fontsize=16)

#%%
data = np.c_[x, y, err_y]

reconstructor = rf.ANN(data, mid_node=4096, hidden_layer=1, hp_model='rec_1')
reconstructor.iteration = 30000
reconstructor.train()
func = reconstructor.predict(xpoint=np.linspace(0, 10, 101))
# func = reconstructor.predict(xspace=(0, 10, 101)) #or use this
reconstructor.save_func(path='rec_1', obsName='xsin') #save the reconstructed function

# reconstructor.plot_loss()
reconstructor.plot_func()


#%%
print ("Time elapsed: %.3f mins" %((time.time()-start_time)/60))
plt.show()
