ReFANN
======

**ReFANN (Reconstruct Functions with Artificial Neural Network)**

ReFANN is a nonlinear interpolating tool based on ANN, without assuming 
a model or parameterization. It can reconstruct functions from data with 
no assumption to the data, and is a completely data-driven approach.

It is proposed by `Guo-Jian Wang, Xiao-Jiao Ma, Si-Yao Li, Jun-Qing Xia (2020) 
<https://doi.org/10.3847/1538-4365/ab620b>`_.

.. The code is open source and has been used in several projects in the Astrophysics literature. #to be updated



Attribution
-----------

If you use this code in your research, please cite `Guo-Jian Wang, Xiao-Jiao Ma, 
Si-Yao Li, Jun-Qing Xia, ApJS, 246, 13 (2020) <https://doi.org/10.3847/1538-4365/ab620b>`_.



Requirements
------------

PyTorch

CUDA (optional)



Installation
------------

You can install refann by using pip::

    $ sudo pip install refann

or from source::

    $ git clone https://github.com/Guo-Jian-Wang/refann.git    
    $ cd refann
    $ sudo python setup.py install


License
-------

Copyright 2020-2020 Guojian Wang

refann is free software made available under the MIT License. For details see the LICENSE file.

