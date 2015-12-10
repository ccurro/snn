## Simple Neural Network

A simple sigmoidal neural network with backpropagation implemented from
scratch in Python 3. In partial satsifaciton of ECE 469: Artificial
Intelligence.

### Requirements

* A recent version on numpy (provides basic array for computation)
* A recent version of sklearn (provides metrics for performance evaluation)
* A recent version of scipy (provides Rayleigh and Rician distributions for 
extra dataset described below)

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

Init files, training sets, test sets, and results files are provided for each
of the following datasets, in the appropriate directories.

#### Provided course datasets

* WDBC dataset: A preprocessed version of the Wisconsin diagnosis breast
cancer dataset. The data consists of various measurements of tumors, and
labels them as either benign or malignant.

* Grades dataset: A dataset of student test scores. Each example corresponds
to a student and five of their assignment grades, each student is labeled with
a their final grade for the term.

#### Extra manufactured dataset

* Rayleigh vs Rician: A dataset comprised of 500 examples each of length 10
Rayleigh and Rician based auto-regressive first-order stochastic processes.
Each example is labeled according to the underlying distribution of the
process. Extra training examples can be generated with
[extraSetGen2.py][./extraSetGen2.py]. The init file for this set was
generating with [genInitWeights.py][./genInitWeights.py]. The weights were
drawn from a uniform distribution over the range [0,0.1].
