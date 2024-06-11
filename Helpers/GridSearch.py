import copy
import os
from cmath import inf
import numpy as np

from mealpy import FloatVar, SBO


def find_best_index(arr) -> int:
    idx = 0
    best = inf
    for i in range(len(arr)):
        if abs(arr[i]) < best:
            best = abs(arr[i])
            idx = i
    return idx


class MyGridSearch:
    def __init__(self, problem, params, seed=42, epoch=100):
        self.models = None
        self.model = None
        self.best_params = None
        self.best_model = None
        self.problem = problem
        self.params = params
        self.seed = seed
        self.epoch = epoch

    def fit(self) -> SBO.OriginalSBO:
        scores = []
        params_list = []
        self.models = []
        idx = 0
        for pop_size in self.params["pop_size"]:
            for alpha in self.params["alpha"]:
                for p_m in self.params["p_m"]:
                    for psw in self.params["psw"]:
                        model = SBO.OriginalSBO(epoch=self.epoch,
                                                pop_size=pop_size,
                                                alpha=alpha,
                                                p_m=p_m,
                                                psw=psw)
                        model.solve(self.problem, seed=self.seed)
                        self.models.append(copy.deepcopy(model))
                        scores.append(model.g_best.solution)
                        params_list.append({'pop_size': pop_size, 'alpha': alpha, 'p_m': p_m, 'psw': psw})
                        #os.system('clear') #mac
                        os.system('cls') #windows
                        if idx % 2 == 0:
                            print('======== \\MyGridSearch/ ========')
                        else:
                            print('======== /MyGridSearch\\ ========')
                        idx += 1
        idx = find_best_index(scores)
        self.best_params = params_list[idx]
        self.best_model = copy.deepcopy(self.models[idx])
        return self.models[idx]
