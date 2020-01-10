# Binacox: automatic cut-point detection in high-dimensional Cox model with applications in genetics

We introduce the binacox, a prognostic method to deal with the problem of detecting multiple cut-points per features in a multivariate setting where a large number of continuous features are available.
The method is based on the Cox model and combines one-hot encoding with the binarsity penalty, which uses total-variation regularization together with an extra linear constraint, and enables feature selection. Nonasymptotic oracle inequalities for prediction and estimation with a fast rate of convergence are established.
The statistical performance of the method is examined in an extensive Monte Carlo simulation study, and then illustrated on three publicly available genetic cancer datasets.
On these high-dimensional datasets, our proposed method significantly outperforms state-of-the-art survival models regarding risk prediction in terms of the C-index, with a computing time orders of magnitude faster. In addition, it provides powerful interpretability from a clinical perspective by automatically pinpointing significant cut-points in relevant variables.

See preprint [here](http://simonbussy.fr/papers/binacox.pdf).
