Overview
========

The REGN project aims to produce more robust and potentially more accurate
retrievals of precipitation from satellite imagery through the use of machine
learning. The project comprises methodological development as well as case-specific
applications which investigate the benefits of the developed methods for
specific retrieval problems.

Quantile regression
-------------------

An important property of precipitation retrievals setting them apart from other
machine learning problems is the inherent (*aleatoric*) uncertainties which
typically dominates the model (*epistemic*) uncertainty. In othe words, given
a sufficiently flexible model, there is usually sufficient data available to
accurately determine model parameters, but nonetheless the retrieval will be
affected by significant uncertainties caused by the ill-posedness of the
retrieval problem.

In a `previous study <https://www.atmos-meas-tech.net/11/4627/2018/>`_, we have
shown that using quantile regression neural networks can be used to produce
well-calibrated estimates of aleatoric uncertainty. Moreover, we showed that they
are compatible with the Bayesian framework that is traditionally applied for
precipitation retrievals, making them a powerful alternative which may overcome
some of the limitation of traditional retrieval methods.

GProf
-----

We are currently investigating the use of QRNNs in the processing of the Goddard
Profiling (GProf) Algorithm. The GProf algorithm is used to produce the global
precipitation product from the Global Precipitation Measurement (`GPM
<https://www.nasa.gov/mission_pages/GPM/main/index.html>`_) mission.

Geostationary Imagery
---------------------

Furthermore, we are investigating using QRNNs to produce precipitation estimates
from geostationary observations. Here we are particularly interested in leveraging
information contained in spatial structure to improve the retrievals.


