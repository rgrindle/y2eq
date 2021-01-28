from eqlearner.dataset.univariate.equationtracker import EquationTracker

import numpy as np

from collections import defaultdict


class EquationTrackerRG(EquationTracker):

    def get_equation(self, drop: int = 0):
        """replace random with np.random
        Note that random.randint = np.random.random_integers"""
        path = []
        curr = self.list_of_eq

        # curr = self.list_of_eq[k]
        while type(curr) == defaultdict:
            res = np.random.choice(list(curr.keys()))
            path.append(res)
            curr = curr[res]
        if len(curr) > 1:
            tmp = np.random.random_integers(0, len(curr)-1)
            path.append(tmp)
            res = curr[tmp]
        else:
            res = curr[0]
        self._drop_element(self.list_of_eq, path, drop)
        return res
