## Simple Neural Network

A simple sigmoidal neural network with backpropagation implemented
from scratch in Python 3.

### Requirements

* A recent version on numpy (provides basic array for computation)
* A recent version of sklearn (provides metrics for performance evaluation)

### How to download and run

```
$ git clone https://github.com/ccurro/snn
$ python nn.py
```

At startup the program will prompt the user, asking if they intend to
train a new network, or test a previously trained network. If training
a new network the program will ask for the location of a init file
describing the network architecture, the location of a training file,
and an output file to save the trained weights of the network. If
testing a previously trained network, the program will prompt the user
for the location of the file containing the trained weights, the
location of the test set file, and an output file where the program
will report the performance metrics.

### Datasets

#### Provided class datasets

* WDBC dataset:
* Grades dataset:

#### Extra manufactured dataset

* Binary multiplication dataset:
