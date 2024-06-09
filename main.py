import time

from mealpy import FloatVar, SBO
import numpy as np
import benchmark_functions as bf
from Consts.enums import FunctionsOptions, MinMax
from Helpers.AlgorytmyGenetyczne.main_bin import get_time_for_bin
from Helpers.AlgorytmyGenetyczne.main_dec import get_time_for_dec
from Helpers.GridSearch import MyGridSearch

gen_bin_stats = get_time_for_bin()
gen_dec_stats = get_time_for_dec()

num_genes = 1  # Liczba wymiarów
func_enum = FunctionsOptions.RASTRIGIN  # Tutaj wybieramy funkcje do optymalizacji
func_min_max = MinMax.MIN  # Tutaj wybieramy czy liczymy maximum czy minimim

func = bf.Rastrigin(n_dimensions=num_genes) \
    if func_enum == FunctionsOptions.RASTRIGIN \
    else bf.Schwefel(n_dimensions=num_genes)

decode_start = func.suggested_bounds()[0][0]  # zakres początkowy w szukanej funkcji
decode_end = func.suggested_bounds()[1][0]  # zakres końcowy w szukanej funkcji

problem = {
    "obj_func": func,
    "bounds": FloatVar(lb=(decode_start,) * num_genes, ub=(decode_end,) * num_genes),
    "minmax": "min",
    "log_to": None,
}

param_grid = {'pop_size': [100],
              'alpha': [0.7, 0.75, 0.8, 0.85, 0.9, 0.95],  # 'alpha' is a float and value should
              # be in range: [0.5, 3.0].
              'p_m': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
              'psw': [0.01, 0.015, 0.02, 0.1, 0.2, 0.3, 0.4]}
search = MyGridSearch(problem, param_grid)
model = search.fit()

print(f"Best Params: {search.best_params}")
print(f"Best agent: {model.g_best}")
print(f"Best solution: {model.g_best.solution}")
print(f"Best accuracy: {model.g_best.target.fitness}")
print(f"Best parameters: {model.problem.decode_solution(model.g_best.solution)}")

start_time = time.time()
model = SBO.OriginalSBO(epoch=1000,
                        pop_size=search.best_params['pop_size'],
                        alpha=search.best_params['alpha'],
                        p_m=search.best_params['p_m'],
                        psw=search.best_params['psw'])
model.solve(problem, seed=42)
end_time = time.time()
SBO_duration = end_time - start_time
print(f"Czas optymalizacji dla najleprzych parametrów: {SBO_duration}s")
print(f"Czas optymalizacji binarnej reprezentacji algorytmów genetycznych: {gen_bin_stats['time']}s")
print(f"Best solution dla binarnej reprezentacji algorytmów genetycznych: {gen_bin_stats['sol']}")
print(f"Czas optymalizacji decymalnel reprezentacji algorytmów genetycznych: {gen_dec_stats['time']}s")
print(f"Best solution dla decymalnel reprezentacji algorytmów genetycznych: {gen_dec_stats['sol']}")

