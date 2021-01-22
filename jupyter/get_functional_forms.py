"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Jan 19, 2021

PURPOSE: Update existing dataset to have triple the number of points
         We will see if training is easier on this dataset.

NOTES: Modified from SeqSeqModel.ipynb

TODO:
"""
from eqlearner.dataset.processing.tokenization import get_string
from tensor_dataset import TensorDatasetCPU as TensorDataset  # noqa: F401

import torch
import numpy as np

SEED = 1234
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


def get_functional_forms(end_name):
    dataset = torch.load('dataset{}.pt'.format(end_name),
                         map_location=torch.device('cpu'))
    dataset_output_list = [d[1] for d in dataset]
    return [get_string(d.tolist()) for d in dataset_output_list]


if __name__ == '__main__':

    func_forms = get_functional_forms(end_name='_train')
    unique_func_forms = np.unique(func_forms)
    print(len(unique_func_forms))
    ff_counts = np.zeros_like(unique_func_forms, dtype=int)
    for i, ff1 in enumerate(unique_func_forms):
        for ff2 in func_forms:
            if ff1 == ff2:
                ff_counts[i] += 1
    print(ff_counts)
    assert np.all(ff_counts > 0)

    indices = np.argsort(ff_counts)

    for i in indices[::-1]:
        print(ff_counts[i], unique_func_forms[i])
    exit()

    import matplotlib.pyplot as plt

    def get_cdf(X):
        """Get the probability that x_i > X after X has
        been sorted (that is x_i is >= i+1 other x's).
        Parameters
        ----------
        X : list
            A sample from the distribution for
            which to compute the CDF
        Returns
        -------
        p : list
            A list of the same length as X that
            give the probability that x_i > X
            where X is a randomly selected value
            and i is the index.
        """

        X_sorted = sorted(X)
        n = len(X)
        p = [i/n for i, x in enumerate(X)]
        return p, X_sorted


    def plot_cdf(X, labels=True, **kwargs):
        """Use get_cdf to plot the CDF.
        Parameters
        ----------
        X : list
            A sample from the distribution for
            which to compute the CDF
        labels : bool (default=True)
            If true, label x-axis x and y-axis Pr(X < x)
        label : str (default=None)
            The legend label.
        color : str (default=None)
            Color to used in plot. If none, it will not
            be pasted to plt.step.
        """

        p, X = get_cdf(X)

        plt.fill_between(X, len(X)*np.array(p),
                         **kwargs)

        if labels:
            plt.ylabel('$Pr(X < x)$')
            plt.xlabel('$x$')


    plt.plot(sorted(ff_counts))
    plt.ylabel('# of instances of\nfunctional form in dataset')
    plt.xlabel('Each integer corresponds to a functional form\nsorted by y-axis')
    plt.tight_layout()
    plt.show()
    plt.figure()
    plt.cdf(ff_counts)
    plt.xlabel('# of instances of\nfunctional forms in dataset')
    plt.ylabel('Cummulative')
    plt.show()
