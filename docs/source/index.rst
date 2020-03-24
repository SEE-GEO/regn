REGN
====

Robust Estimation of Global Precipitation using Neural Networks (REGN) is a
research project investigating how quantile regression neural networks (`QRNNs
<https://www.atmos-meas-tech.net/11/4627/2018/>`_) can be used to yield more
reliable and potentially more accurate precipitation estimates from the
satellite observations of the Global Precipitation Measurement (`GPM
<https://www.nasa.gov/mission_pages/GPM/main/index.html>`_).

.. toctree::
   :maxdepth: 1
   :caption: Getting started

   dependencies
   dendrite
   jupyter_notebook
   documentation

.. toctree::
   :maxdepth: 1
   :caption: REGN

   overview
   data

.. toctree::
   :maxdepth: 1
   :caption: Examples
   notebooks/examples/train_qrnn
   notebooks/examples/train_unet
   notebooks/examples/qrnn_classification

.. toctree::
   :maxdepth: 1
   :caption: Results
   notebooks/classification/test_classification

.. toctree::
   :maxdepth: 1
   :caption: API Documentation

   api.data
