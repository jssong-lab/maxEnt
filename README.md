# maxEnt
A method to extract features used by feed-forward artifical neural networks for genomic sequence classification

## Dependencies
The maxEnt code has been tested for artifical neural networks implemented as sequential models in Keras 1.2.0 using the Theano 0.8.2 backend.

Install Theano 0.8.2 with the conda package manager using
```
conda install theano==0.8.2
```
or without conda using
```
pip install theano==0.8.2
```
Install Keras 1.2.0 using
```
pip install keras==1.2.0
```
Then check that your $HOME/.keras/keras.json file contains the lines:
> "image_dim_ordering": "th",  
> "backend": "theano",

## Using the code
See examples/CTCF_convNet_example.ipynb. Note that some visualization in this example requires a local installation of R with the package RWebLogo.
