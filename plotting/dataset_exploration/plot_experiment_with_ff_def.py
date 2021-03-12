"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Mar 10, 2021

PURPOSE: I don't know how a functional for should be defined.
         Here I want to play with different ideas to help me
         fine tune my defition of a functional form. For example,
         should functional forms always used positive coefficients?

NOTES:

TODO:
"""
from plot_examples_of_functional_forms import plot_examples
from equation.EquationInfix import EquationInfix

import numpy as np

np.random.seed(123456)

ff = 'exp(sin(x)**3)'
eq = EquationInfix(ff)
ff = eq.eq_str
eq.eq_str = eq.eq_str.replace('np.', '')
print(eq.eq_str)


def get_examples(a, b, fixed_c=(None, None, None)):
    examples = []
    for _ in range(10):
        c = np.random.uniform(a, b, size=3)
        c = [ci if fci is None else fci for ci, fci in zip(c, fixed_c)]
        eq_ = eq.eq_str
        for i, ci in enumerate(c):
            eq_ = eq_.replace('c[{}]'.format(i), str(ci))
        examples.append(eq_)
    return examples


x = np.arange(0.1, 3.1, 0.1)

plot_examples(examples=get_examples(a=-3, b=3),
              x=x,
              ff=ff,
              save_loc='plot_experiment_with_ff_def/regular.pdf')


plot_examples(examples=get_examples(a=0, b=3),
              x=x,
              ff=ff,
              save_loc='plot_experiment_with_ff_def/positive_only.pdf')


plot_examples(examples=get_examples(a=-3, b=0),
              x=x,
              ff=ff,
              save_loc='plot_experiment_with_ff_def/negative_only.pdf')

for i in range(3):
    fixed_c = [1.]*3
    fixed_c[i] = None
    plot_examples(examples=get_examples(a=-3, b=3, fixed_c=fixed_c),
                  x=x,
                  ff=ff,
                  save_loc='plot_experiment_with_ff_def/only_c{}.pdf'.format(i))

