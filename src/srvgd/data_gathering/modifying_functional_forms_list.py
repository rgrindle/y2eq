"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Jan 19, 2021

PURPOSE: Elimiate confusing functional forms. This include
         functional forms of the form exp(x), exp(2x), exp(3x)
         which are all really the same.

NOTES: ff = functional form

TODO:
"""
import pandas as pd


def has_coeffs(ff):
    """Find coefficients that have slipped into the functional
    from. I expect these to be found as exp(#*x), but there
    may be other forms. Note that we need to return false
    for exponents. Example: x**3

    Parameters
    ----------
    ff : str
        The functional from to check.

    Returns
    -------
     : bool
        Returns true if there are coeffs in ff. Otherwise,
        returns false.
    """
    for i, char in enumerate(ff):
        if char.isdigit():
            if i != len(ff)-1:
                if ff[i+1:i+3] == '*x':
                    return True
    return False


def fix_ff(ff):
    """Fix ff if it has instances of 3*x, 2*x, ...
    Do this by using the lowest numbers possible.

    Parameters
    ----------
    ff : str
        The functional form

    Returns
    -------
    ff_fixed : str
        The fixed functional form

    Example
    -------
    >>> fix_ff('exp(3*x)+exp(2*x)')
    'exp(2*x)+exp(x)'
    """
    problem_indices = []
    for i, char in enumerate(ff):
        if char.isdigit() and ff[i+1:i+3] == '*x':
            problem_indices.append(i)

    if 'exp' not in ff == 1:
        return ff.replace(ff[problem_indices[0]]+'*', '')
    else:
        corrected = {i: None for i in problem_indices}
        for i in reversed(problem_indices):
            if i+11 >= len(ff):
                corrected[i] = 'x'
            elif ff[i+4:i+9] == '+exp(':
                if i+9 in corrected and corrected[i+9] is not None:
                    char_of_interest = corrected[i+9]
                else:
                    char_of_interest = ff[i+9]

                if char_of_interest == 'x':
                    corrected[i] = '2'
                elif char_of_interest.isdigit():
                    corrected[i] = str(int(char_of_interest)+1)
                else:
                    raise Exception('This is a situation not concidered', ff)
            else:
                corrected[i] = 'x'

        ff_fixed_list = []
        prev = 0
        for i in corrected:
            ff_fixed_list.append(ff[prev:i])
            ff_fixed_list.append(corrected[i])
            prev = i + 1
            if corrected[i] == 'x':
                prev += 2
        ff_fixed_list.append(ff[prev:])
        return ''.join(ff_fixed_list)


if __name__ == '__main__':
    import numpy as np

    ff_list = pd.read_csv('functional_forms00.csv', header=None).iloc[:, 1].values

    # Find functional forms with 3*x, 2*x, (are there other ints?)
    # I expect these to only occur for exp. Is that accurate?
    has_coeffs_list = []
    no_coeffs_list = []
    for ff in ff_list:
        if has_coeffs(ff):
            has_coeffs_list.append(ff)
        else:
            no_coeffs_list.append(ff)

    print(len(has_coeffs_list), len(no_coeffs_list))
    corrected_list = []
    for ff in sorted(has_coeffs_list):
        corrected_list.append(fix_ff(ff))

    fixed_ff_list = no_coeffs_list + corrected_list
    print(len(np.unique(corrected_list)))
    unique_ff_list = np.unique(fixed_ff_list)
    print(len(fixed_ff_list), len(unique_ff_list))

    for i, u_ff in enumerate(unique_ff_list):
        if 'E*' in u_ff:
            print(unique_ff_list[i])
            unique_ff_list[i] = u_ff.replace('E*', '')
            print(unique_ff_list[i])

    # unique_ff_list = np.unique(unique_ff_list)
    pd.DataFrame(unique_ff_list).to_csv('unique_ff_list.csv',
                                        index=False, header=None)    
