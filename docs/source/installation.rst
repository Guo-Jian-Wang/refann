.. _installation:

Installation
************

Since refann is a pure python module, it is easy to install.


Dependencies
============

The main dependencies of refann are:

* `PyTorch <https://pytorch.org/>`_
* `CUDA <https://developer.nvidia.com/cuda-downloads>`_ (optional, but suggested)


Package managers
================

You can install refann by using pip::

    $ sudo pip install refann

or from source::

    $ git clone https://github.com/Guo-Jian-Wang/refann.git    
    $ cd refann
    $ sudo python setup.py install


.. how to use conda?


Test the installation
=====================

To test the correctness of the installation, you just need to download the `examples <https://github.com/Guo-Jian-Wang/refann/tree/master/examples>`_ and execute it in the examples directory by using the following command::

    $ python rec_xsin.py

