"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Dec 10, 2020

PURPOSE: Update DatasetCreator.generate_set to
         reject duplicate equations.

NOTES:

TODO:
"""
from srvgd.utils.normalize import normalize, get_normalization_params

from eqlearner.dataset.univariate.datasetcreator import DatasetCreator
from eqlearner.dataset.processing import tokenization, tensordataset

from sympy import lambdify
import numpy as np
import torch


class DatasetCreatorRG(DatasetCreator):

    def evaluate_function(self, X, sym_function, X_noise=0, Y_noise=0):
        x = self.symbol
        try:
            function = lambdify(x, sym_function)
            # All the warnings are related to function not correctly evaluated. So we catch them and set a nan.
            y = self.handling_nan_evaluation(X, function, X_noise=X_noise, Y_noise=Y_noise)
            y = np.array(y)
            return y
        except KeyError:    # ComplexInfinity
            return np.array([np.nan])

    def generate_batch(self,X, number_to_generate, X_noise=0, Y_noise=0, return_real_dict=False):
        """Generate batch ready for being deployed in the network
        Output is a np.array with X and y concatenated"""
        input_network = []
        output_network = []
        real_dict = []
        functions = []
        i = 0
        while i<number_to_generate:
            fun, dictionary, dictionary_cleaned = self.generate_fun()
            functions.append(fun)
            i += 1
            input_network.append(np.array([X, self.evaluate_function(X,fun, X_noise=X_noise, Y_noise=Y_noise)]))
            output_network.append(dictionary_cleaned)
            real_dict.append(dictionary)
        if return_real_dict:
            return input_network, output_network, functions, real_dict
        else:
            return input_network, output_network, functions

    def generate_set(self, support, num_equations,
                     isTraining=True, threshold=2000,
                     multiple_scaling=False,
                     consistent_scaling=False):
        assert not (multiple_scaling and consistent_scaling)
        if isTraining:
            self.x_data = {}
            self.y_data = {}
            dataset_type = 'train'
        else:
            dataset_type = 'test'
        self.x_data[dataset_type] = []
        self.y_data[dataset_type] = []
        eq_with_consts = []
        cnt = 0
        skipped = 0
        cond = True
        while cond:
            numerical, dictionary, function, real_dict = self.generate_batch(support, 1, return_real_dict=True)
            n = list(numerical[0][1])
            # print(numerical[0])
            if np.max(numerical[0][1]) < threshold and np.min(numerical[0][1]) > -threshold:
                sub_condition = n not in self.x_data['train']
                if not isTraining:
                    sub_condition = sub_condition and n not in self.x_data['test']
                if sub_condition:
                    self.x_data[dataset_type].append(n)
                    self.y_data[dataset_type].append(torch.Tensor((tokenization.pipeline(dictionary)[0])))
                    eq_with_consts.append(function[0])
                    cnt += 1
                    print('.', end='', flush=True)
                else:
                    skipped += 1
            else:
                skipped += 1
            if cnt == num_equations:
                cond = False
        if multiple_scaling:
            if isTraining:
                self.scaler.fit(self.x_data['train'])
            self.x_data[dataset_type] = self.scaler.transform(self.x_data[dataset_type])
        elif consistent_scaling:
            if isTraining:
                self.min_, self.scale_ = get_normalization_params(self.x_data['train'])
            self.x_data[dataset_type] = normalize(self.x_data[dataset_type], self.min_, self.scale_)
        l = [len(y) for y in self.y_data[dataset_type]]
        q = np.max(l)
        y_p = torch.zeros(len(self.y_data[dataset_type]), q)
        for i, y in enumerate(self.y_data[dataset_type]):
            y_p[i, :] = torch.cat([y, torch.zeros(q-y.shape[0])])
        dataset = tensordataset.TensorDataset(torch.Tensor(self.x_data[dataset_type]), y_p.long())

        info = self.get_info()
        info['isTraining'] = isTraining
        info['num_equations'] = num_equations
        info['threshold'] = threshold
        info['Support'] = support
        if multiple_scaling:
            info['min_'] = self.scaler.min_
            info['scale_'] = self.scaler.scale_
        elif consistent_scaling:
            info['min_'] = self.min_
            info['scale_'] = self.scale_
        else:
            info['min_'] = None
            info['scale_'] = None
        return dataset, info, eq_with_consts
