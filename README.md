# Symbolic Regression via Gradient Descent (SRvGD)

This code constructs a dataset, constructs a neural network, trains that neural network and evaluates the neural network.

## Getting started
I am running this code on Python 3.7.4 with the following dependencies.

numpy==1.17.1, pandas==0.25.1, tensorflow==2.3.1, sympy==1.6.2, scipy==1.4.1 TODO: add version numbers

Also using EQLearner to help generated the dataset. TODO: put link and maybe more explanation.

### Install
```
cd SRvGD
pip install .
```

### Generate dataset
```
cd SRvGD/src/srvgd/data_gathering
python gen_dataset.py
```
This creates SRvGD/datasets/dataset.npy

### Train the neural network
```
cd SRvGD/src/srvgd/utils
python train.py
```
The model will be save in cd SRvGD/models/

### Evaluate the neural network
```
python eval.py
```
This will create a list of the root mean squared errors on the test dataset.

## TODOS
- [ ] switching models
- [ ] switching datasets
- [ ] specifying options
- [ ] more detail about what each step does
- [ ] include results
- [ ] include link to paper when exists
- [ ] confirm problem with fixed support
- [ ] fix problem with fixed support
