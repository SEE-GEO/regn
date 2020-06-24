Rain/No-rain classification
---------------------------

The aim of this study is to investigate suitable ways to classify pixels of passive-microwave
observations as either raining or non-raining.

Approaches
==========

1. Neural network classifier: The common deep-learning approach to binary classification is
   to train a (fully-connected) neural network with Sigmoid loss and binary-cross entropy
   loss function.
2. Use a QRNN to estimate the posterior probability of the observed rain exceeding a given
   threshold.
3. Use a loss function that yields well-calibrated probabilities (how?).



Discussion
==========

 Approach 1 can be formally motivated by predicting the parameter :math:`p` of a
 Bernoulli distribution conditioned on the network input. Although this should
 allow interpreting the squashed network output as probability, the predicted
 probabilities are not well calibrated (citation?).


Other questions
===============

- What rain threshold should be used? Gustav used :math:`0.1\ \text{mm\ h}^{-1}`
