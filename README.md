# Binacox
_binacox_ is a high-dimensional survival model that automatically detects multiple cut-points

## Quick description
We introduce the _binacox_, a prognostic method to deal with the problem of detecting multiple cut-points per features in a multivariate setting where a large number of continuous features are available.
The method is based on the Cox model and combines one-hot encoding with the binarsity penalty, which uses total-variation regularization together with an extra linear constraint, and enables feature selection. Nonasymptotic oracle inequalities for prediction and estimation with a fast rate of convergence are established.
The statistical performance of the method is examined in an extensive Monte Carlo simulation study, and then illustrated on three publicly available genetic cancer datasets.
On these high-dimensional datasets, our proposed method significantly outperforms state-of-the-art survival models regarding risk prediction in terms of the C-index, with a computing time orders of magnitude faster. In addition, it provides powerful interpretability from a clinical perspective by automatically pinpointing significant cut-points in relevant variables.

See preprint [here](http://simonbussy.fr/papers/binacox.pdf).

## Installation
Clone the repository, then inside the folder, use a `virtualenv` to install the requirements
```shell script
git clone git@github.com:Califrais/binacox.git
cd binacox

# If your default interpreter is Python3:
virtualenv .env
# If your default interpreter is Python2, you can explicitly target Python3 with:
virtualenv -p python3 .env

source .env/bin/activate
```
Then, to download all required modules and initialize the project run the following commands:
```shell script
pip install -r requirements.txt
```

To use the package outside the build directory, the build path should be added to the `PYTHONPATH` environment variable, as such (replace `$PWD` with the full path to the build directory if necessary):

    export PYTHONPATH=$PYTHONPATH:$PWD

For a permanent installation, this should be put in your shell setup script. To do so, you can run this from the _binacox_ directory:

    echo 'export PYTHONPATH=$PYTHONPATH:'$PWD >> ~/.bashrc

Replace `.bashrc` with the variant for your shell (e.g. `.tcshrc`, `.zshrc`, `.cshrc` etc.).

## Other files

The Jupyter notebook "tutorial" gives useful example of how to use the model based on simulated data.
It will be very simple then to adapt it to your own data.