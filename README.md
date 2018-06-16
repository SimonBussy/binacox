# Binacox: automatic cut-points detection in high-dimensional Cox model, with applications to genetic data

Determining significant prognostic biomarkers is of increasing importance in many
areas of medicine. In order to translate a continuous biomarker into a clinical decision,
it is often necessary to determine cut-points. There is so far no standard
method to help evaluate how many cut-points points are optimal for a given feature
in a survival analysis setting. Moreover, most existing methods are univariate, hence
not well suited for high-dimensional frameworks. 

We introduces a prognostic method called Binacox to deal with the problem of detecting multiple cut-points per
features in a multivariate setting where a large number of continuous features are
available. It is based on the Cox model and combines one-hot encodings with the
binarsity penalty. This penalty uses total-variation regularization together with an
extra linear constraint to avoid collinearity between the one-hot encodings and enable
feature selection. 

See preprint [here](http://simonbussy.fr/papers/binacox.pdf).
