# Binacox
_binacox_ is a high-dimensional survival model that automatically detects multiple cut-points

## Quick description
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