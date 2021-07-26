# The perils and pitfalls of symbolic regression

The ever-growing accumulation of data makes automated distillation of understandable models from that data ever-more desirable. Deriving equations directly from data using symbolic regression, as performed by genetic programming, continues its appeal due to its algorithmic simplicity and lack of assumptions about equation form. However, few models besides a sequence-to-sequence approach to symbolic regression, introduced in 2020 [1] that we call y2eq, have been shown capable of transfer learning: the ability to rapidly distill equations successfully on new data from a previously unseen domain, due to experience performing this distillation on other domains. In order to improve this model, it is necessary to understand the key challenges associated with it. We have identified three important challenges: corpus, coefficient, and cost. The challenge of devising a training corpus stems from the hierarchical nature of the data since the corpus should not be considered as a collection of equations but rather as a collection of functional forms and instances of those functional forms. The challenge of choosing appropriate coefficients for functional forms compounds the corpus challenge and presents further challenges during evaluation of trained models due to the potential for similarity between instances of different functional forms. The challenge with cost functions (used to train the model) is mainly the choice between numeric cost (compares y-values) and symbolic cost (compares written functional forms). This code repository re-implements the neural network from [1]. The repository can be used to provide evidence for the existence of the corpus, coefficient, and cost challenges. We hope that this repository can be used to initiate improvements to this already promising symbolic regression model.

With this code, all expimental results from here (TODO: like to thesis) can be recreated.


[1] Biggio, Luca, Tommaso Bendinelli, Aurelien Lucchi, and Giambattista Parascandolo. "A seq2seq approach to symbolic regression." In Learning Meets Combinatorial Algorithms at NeurIPS2020. 2020.


## Getting started
I am running this code on Python 3.7.4 with the following dependencies.

numpy==1.17.1, pandas==0.25.1, tensorflow==2.3.1, sympy==1.6.2, scipy==1.4.1

The [EQLearner respository](https://github.com/SymposiumOrganization/EQLearner) is also used to help generated the datasets. Note that you will need to add some empty `__init__.py` files to install EQLearner.

### Install
```
mkdir SRvGD
clone https://github.com/rgrindle/y2eq.git
cd SRvGD
pip install .
```

## Examples
The following examples can be used to recreate results from (TODO: link to thesis here)

### Example 1: Generate dataset
In this example, a training dataset of 50,000 observations is generated using 1,000 functional forms. Each functional form is involved in roughly 50 observations. Each observation is (y, f) where y is a set of y-values corresponding to the functional form f.
```
cd SRvGD/src/srvgd/data_gathering/
python generate_dataset_from_functional_forms.py
```
This creates SRvGD/datasets/dataset_train_ff1000.pt

### Example 2: Train the neural network
To train the y2eq model on the dataset created by Example 1, 
```
cd SRvGD/src/srvgd/architecture/transformer/
python y2eq_transformer.py
```
The model will be save as SRvGD/models/y2eq_transformer.pt (model on final epoch) and SRvGD/models/BEST_y2eq_transformer.pt (best model according to validation loss).

### Example 3: Evaluate trained the neural network
Once y2eq has been trained (see Example 2), y2eq can be evaluated by
```
cd SRvGD/src/srvgd/eval_y2eq-transformer-fixed-fixed/
python 00_eval_y2eq-transformer-fixed-fixed.py
python 01_eval_y2eq-transformer-fixed-fixed.py
for i in {0..999}
do
  python 02_eval_y2eq-transformer-fixed-fixed_vacc.py
done
python 03_eval_y2eq-transformer-fixed-fixed_vacc.py
```
First, 00 creates data to evaluate y2eq-transformer, then 01 records the output functional forms, then 02 computes root mean squared errors (RMSE) of these functional forms, and finally 03 groups all RMSEs into one file: 02_rmse_150.csv.

