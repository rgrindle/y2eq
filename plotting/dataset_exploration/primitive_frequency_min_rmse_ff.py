"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Apr 21, 2021

PURPOSE: Create histograms exploring the occurance
         of various primitives in the dataset.

NOTES:

TODO:
"""
from srvgd.updated_eqlearner.tokenization_rg import get_eq_string
from plot_bar_zordered import plot_bar_zordered

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def count_tokens(eq, token_list=None):
    if token_list is None:
        token_list = get_token_list()
    count = {}
    for token in token_list:
        count[token] = eq.count(token)
    count['*'] -= 2*count['**']
    count['x'] -= count['exp']
    return count


def get_token_list():
    return ('**', 'sin', 'exp', 'log', '*', '+', '(', ')', 'x',
            '1', '2', '3', '4', '5', '6')


def get_blank_count():
    keys = get_token_list()
    return {key: 0 for key in keys}


def get_complexity(ff):
    complexity = 0
    counts = count_tokens(ff)
    for token in counts:
        if token in ('(', ')'):
            continue
        else:
            complexity += counts[token]
    return complexity


def test_get_complexity():
    assert get_complexity('x**2') == 3
    assert get_complexity('exp(exp(x))') == 3
    assert get_complexity('sin(x)**3+sin(x)**2+exp(x)') == 12


def test_count_tokens():
    count = get_blank_count()
    count['x'] = 1
    assert count_tokens('x') == count

    count = get_blank_count()
    count['exp'] = 1
    count['x'] = 2
    count['('] = 1
    count[')'] = 1
    count['*'] = 1
    count['**'] = 1
    count['2'] = 1
    assert count_tokens('exp(x)*x**2') == count


def sum_dict(*dict_list):
    for d in dict_list:
        assert d.keys() == dict_list[0].keys()

    summed_dict = {}
    for key in dict_list[0]:
        summed_dict[key] = 0
        for d in dict_list:
            summed_dict[key] += d[key]
    return summed_dict


def test_sum_dict():
    assert sum_dict({'a': 4}, {'a': 5}) == {'a': 9}

    assert sum_dict({'a': 4}, {'a': 5}, {'a': 1}) == {'a': 10}

    assert sum_dict({'a': 1, 'b': 2, 'c': 3},
                    {'a': 4, 'b': 5, 'c': 6}) == {'a': 5, 'b': 7, 'c': 9}


test_count_tokens()
test_sum_dict()
test_get_complexity()

dataset_name = 'min_rmse_ff'

min_rmse_ff_data = pd.read_csv('../../src/srvgd/data_gathering/min_rmse_ff_list.csv', header=None).values.flatten().tolist()

dataset = torch.load('../../datasets/dataset_test_ff1000.pt'.format(dataset_name),
                     map_location=torch.device('cpu'))
# y2eq_outputs = pd.read_csv('../../eval_y2eq-fixed-fixed/01_predicted_ff.csv', header=None).values.flatten().tolist()
# xy2eq_outputs = pd.read_csv('../../eval_xy2eq-fixed-fixed/01_predicted_ff.csv', header=None).values.flatten().tolist()


ff_list = {'dataset_min_rmse_ff': min_rmse_ff_data,
           'dataset_random_ff': [get_eq_string(d[1].tolist())[5:-3] for d in dataset]}
# 'y2eq': y2eq_outputs,
# 'xy2eq': xy2eq_outputs}

# Get the unique functional forms.
# Count the occurances of each functional from
# in the dataset/output.
# And, get the sorted indices based on counts.
unique_ff_list = {}
ff_counts_list = {}
sorted_indices = {}
token_counts_per_ff = {}
token_counts_total = {}
token_counts_total_normalied = {}
complexity_of_ff = {}
for key in ff_list:
    unique_ff_list[key] = np.unique(ff_list[key])
    ff_counts_list[key] = []
    for ff in unique_ff_list[key]:
        ff_counts_list[key].append(ff_list[key].count(ff))
    ff_counts_list[key] = np.array(ff_counts_list[key])
    sorted_indices[key] = np.argsort(ff_counts_list[key])

    # Then, get the occurances of each token in
    # each functional form.
    token_counts_per_ff[key] = [count_tokens(ff) for ff in unique_ff_list[key]]
    token_counts_total[key] = sum_dict(*token_counts_per_ff[key])
    print(key, len(unique_ff_list[key]))
    token_counts_total_normalied[key] = {token: token_counts_total[key][token]/len(unique_ff_list[key]) for token in token_counts_total[key]}

    # Get complexity of functional forms
    complexity_of_ff[key] = [get_complexity(ff) for ff in unique_ff_list[key]]


key = 'dataset_min_rmse_ff'
fig, axes = plt.subplots(nrows=1+len(get_token_list()),
                         figsize=(3, 9))
plt.sca(axes[0])
plt.bar(range(len(unique_ff_list[key])), ff_counts_list[key][sorted_indices[key]],
        width=1.0, color='C0', edgecolor='C0')
plt.ylabel('Counts\nof ff', fontsize=8)
plt.xticks([])
plt.yticks(fontsize=4)

for i, token in enumerate(get_token_list()):
    plt.sca(axes[1+i])
    token_counts = [count[token] for count in token_counts_per_ff[key]]
    plt.plot(np.array(token_counts)[sorted_indices[key]],
             linewidth=0.1)
    if i < 18:
        plt.xticks([])
    plt.yticks(fontsize=4)
    plt.ylabel(token, fontsize=8)
    plt.ylim([0, 10])

plt.xlabel('Functional form (sorted by frequency)'
           '\n[each integer corresponds to a'
           '\nunique functional form]',
           fontsize=8)
plt.subplots_adjust(top=0.99, bottom=0.05,
                    left=0.21, right=0.99,
                    hspace=0.1)
plt.savefig('primitive_frequency_sorted_{}.pdf'.format(dataset_name))

plt.figure()
plot_bar_zordered(token_counts_total_normalied)
plt.ylabel('Average token counts per unique functional form')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())
plt.savefig('primitive_frequency_{}.pdf'.format(dataset_name))

plt.figure()
plt.boxplot(complexity_of_ff.values(), labels=complexity_of_ff.keys())
plt.ylabel('Functional form complexity\n(Number of tokens excluding parenthesis)')
plt.savefig('complexity_boxplot_{}.pdf'.format(dataset_name))

plt.figure()
for key in complexity_of_ff:
    complexity_of_ff[key] = {i: v for i, v in enumerate(complexity_of_ff[key])}
plot_bar_zordered(complexity_of_ff)
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.xlabel('Functional forms')
plt.ylabel('Functional form complexity\n(Number of tokens excluding parenthesis)')
plt.legend(by_label.values(), by_label.keys())
plt.savefig('complexity_{}.pdf'.format(dataset_name))
