# Robust Estimation of Global precipitation using Neural networks (REGN)

The acronym REGN stands for *Robust Estimation of Global precipitation using
Neural networks*. At the same time, *regn* ([rɛŋn]) is the swedish word for
rain. The aim of the REGN project is to develop a neural-network based
implementation of the GPROF algorithm.

This repository is used to collect all of the code and and results from
this project.

## AGU Presentation

Intermediate results from the REGN project have been presented at AGU 2020
in the presentation

> H206-07 - Using Neural Networks for Bayesian Precipitation Retrievals from GPM Passive Microwave Observations

as part of the session *H206 - Space-Based Precipitation Observations and Estimation: Innovations for Science and Applications I*.

Slides from the presentation can be found [here](https://raw.githubusercontent.com/SEE-MOF/regn/master/presentation/agu_presentation.pdf).

## Running the code

The code required to reproduce the presented results consists of two parts:

- The ``regn`` Python package, which implements the QRNN-based GPROF retrieval
- The Jupyter notebooks contained in the [notebooks/gmi](notebooks/gmi) and
  [notebook/mhs](notebooks/mhs) folders, which contain the Python code which
  performs the numerical analyses.
  
### Python dependencies

> **Note**: Before installing any of these dependencies it is probably a good
> idea to create a new environment using Python venv or conda.

Our work builds on and requires a range publicly available packages, which
are collected in the ``requirements.txt``. After cloning this repository, you
can install these packages using:

````
$ python3 -m p install -r requirements.txt
````

### Installing the ``regn`` package

To run any of the notebooks, the ``regn`` package must be in your ``PYTHONPATH``.
The easiest way to achieve this is probably to just install the package using
``pip``:

````
$ python3 -m p install -e .
````
## The QRNN implementation

We have recently migrated our implementation of QRNNs from the
[typhon](https://github.com/atmtools/typhon/) package to a new, separate package
called [quantnn](https://github.com/simonpf/quantnn). This is still
relatively new and lacks extensive documentation but is what has been used within
this study.

## References

For background information on quantile regression neural networks (QRNNs), refer
to the following article:

- Pfreundschuh, S., Eriksson, P., Duncan, D., Rydberg, B., Håkansson, N., and Thoss, A.: A neural network approach to estimating a posteriori distributions of Bayesian retrieval problems, Atmos. Meas. Tech., 11, 4627–4643, [https://doi.org/10.5194/amt-11-4627-2018](https://doi.org/10.5194/amt-11-4627-2018), 2018.

For more information on the current GPROF algorithm, please refer to the following
publications:

- Kummerow, C. D., Randel, D. L., Kulie, M., Wang, N. Y., Ferraro, R., Joseph Munchak, S., & Petkovic, V. (2015). The evolution of the Goddard profiling algorithm to a fully parametric scheme. Journal of Atmospheric and Oceanic Technology, 32(12), 2265-2280.

- [The GPROF version 5 ATBD](http://rain.atmos.colostate.edu/ATBD/ATBD_GPM_June1_2017.pdf)


