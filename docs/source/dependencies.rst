Required software
-----------------

To setup your computing environment you will need a working installation of
Python 3 and :code:`pip`. A convenient way to obtain both of them is through
`Anaconda <https://docs.conda.io/projects/conda/en/latest/user-guide/install>`_.

Typhon
^^^^^^

The :code:`typhon` package provides an implementation
of QRNNs. Since the work we are doing here will likely
depend on new features added to typhon, you have to
install the development version.

.. code-block:: none

   git clone https://github.com/simonpf/typhon -b qrnn_refactoring
   cd typhon
   pip install -e .


Pytorch
^^^^^^^

We will use the :code:`pytorch` deep-learning package as backend
for neural networks. To install it, simply follow the instructions
at `pytorch.org <https://pytorch.org/>`_.

Jupyter
^^^^^^^

We will use jupyter notebooks for data analysis. To install :code:`jupyter`
follow the instruction at the `jupyter homepage <https://jupyter.org/install>`_.

regn
^^^^

This repository is itself a Python package which we will use to collect
code that we will write. To install it navigate to the top-level directory
and install it using

.. code-block:: none

   pip install -e .

