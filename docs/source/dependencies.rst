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

.. code-block:: bash

   git clone https://github.com/atmtools/typhon
   cd typhon
   pip install -e .


Pytorch
^^^^^^^

We will use the :code:`pytorch` deep-learning package as backend
for neural networks. To install it, simply follow the instruction
at `pytorch.org <https://pytorch.org/>`_.

