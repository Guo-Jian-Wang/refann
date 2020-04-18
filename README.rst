ReFANN
=====

**ReFANN (Reconstruct Functions with Artificial Neural Network)**

ReFANN is a nonlinear interpolating tool based on ANN, without assuming 
a model or parameterization. It can reconstruct functions from data with 
no assumption to the data, and is a completely data-driven approach.

It is proposed by `Guo-Jian Wang, Xiao-Jiao Ma, Si-Yao Li, Jun-Qing Xia (2020) 
<https://doi.org/10.3847/1538-4365/ab620b>`_.



Attribution
-----------

If you use this code in your research, please cite `Guo-Jian Wang, Xiao-Jiao Ma, 
Si-Yao Li, Jun-Qing Xia, ApJS, 246, 13 (2020) <https://doi.org/10.3847/1538-4365/ab620b>`_.
The BibTeX entry for the paper is::

@article{Wang:2019vxv,
    author = "Wang, Guo-Jian and Ma, Xiao-Jiao and Li, Si-Yao and Xia, Jun-Qing",
    archivePrefix = "arXiv",
    doi = "10.3847/1538-4365/ab620b",
    eprint = "1910.03636",
    journal = "Astrophys. J. Suppl.",
    number = "1",
    pages = "13",
    primaryClass = "astro-ph.CO",
    title = "{Reconstructing Functions and Estimating Parameters with Artificial Neural Networks: A Test with a Hubble Parameter and SNe Ia}",
    volume = "246",
    year = "2020"
}


Requirements
------------

PyTorch

CUDA (optional)



Installation
------------

$ sudo pip install refann-master.zip

or

$ sudo python setup.py install



License
-------

Copyright 2020-2020 Guojian Wang

refann is free software made available under the MIT License. For details see the LICENSE file.
