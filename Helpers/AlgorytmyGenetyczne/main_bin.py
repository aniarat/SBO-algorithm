import logging
import time

import pygad
import numpy
import matplotlib.pyplot as plt
import benchmark_functions as bf
import os

from Helpers.AlgorytmyGenetyczne.Consts.enums import MinMax, CrossingMethodsBin, FunctionsOptions
from Helpers.AlgorytmyGenetyczne.Helpers.crossingMethodsBin import TestCrossover, SinglePointCrossover, \
    TwoPointCrossover, ThreePointCrossover, GrainCrossover, UniformCrossover, ScanningCrossover, PartialCopyCrossover, \
    MultivariateCrossover
from Helpers.AlgorytmyGenetyczne.Helpers.decimalBinaryMath import get_actual_values, split


############ BINARNA ############

print("\n---Reprezentacja binarna---\n")
#Funkcje do pobierania wartości od użytkownika
def choose_option(prompt, options):
    print(prompt)
    for i, option in enumerate(options, 1):
        print(f"{i}. {option}")
    choice = input("Wybierz opcję: ")
    if not choice.isdigit() or int(choice) < 1 or int(choice) > len(options):
        print("Nieprawidłowy wybór. Wybierz numer opcji z listy.")
    else:
        return options[int(choice) - 1]

def get_selection_type():
    options = ["rws", "random", "tournament"]
    return choose_option("Wybierz metodę selekcji:", options)

def get_mutation_type():
    options = ["random", "None", "swap", "inversion", "adaptive"]
    return choose_option("Wybierz rodzaj mutacji:", options)

def get_crossover_type():
    options = [
        CrossingMethodsBin.BUILD_IN_SINGLE_POINT,
        CrossingMethodsBin.BUILD_IN_DOUBLE_POINT,
        CrossingMethodsBin.BUILD_IN_UNIFORM,
        CrossingMethodsBin.SINGLE_POINT,
        CrossingMethodsBin.DOUBLE_POINT,
        CrossingMethodsBin.TRIPLE_POINT,
        CrossingMethodsBin.UNIFORM,
        CrossingMethodsBin.GRAIN,
        CrossingMethodsBin.SCANNING,
        CrossingMethodsBin.PARTIAL,
        CrossingMethodsBin.MULTIVARIATE
    ]
    return choose_option("Wybierz typ krzyżowania:", options)

def get_function_enum():
    options = [FunctionsOptions.RASTRIGIN, FunctionsOptions.SCHWEFEL]
    return choose_option("Wybierz funkcję optymalizacji:", options)

def get_min_max():
    options = [MinMax.MIN, MinMax.MAX]
    return choose_option("Wybierz minimalizację lub maksymalizację:", options)

def get_user_input(prompt, input_type=int):
    while True:
        try:
            value = input_type(input(f"{prompt}: "))
            if input_type == int and value <= 0:
                print("Wartość musi być liczbą całkowitą większą od zera.")
                continue
            elif input_type == str and not value:
                print("Wartość nie może być pusta.")
                continue
            else:
                return value
        except ValueError:
            print("Wprowadzona wartość jest nieprawidłowa.")

    # return input_type(input(f"{prompt}: "))

def get_input_with_check(prompt, condition):
    value = get_user_input(prompt)
    while not condition(value):
        print(f"Nieprawidłowa wartość: {value}. Proszę, spróbuj ponownie.")
        value = get_user_input(prompt)
    return value

#Parametry podawane przez użytkownika
num_genes = get_user_input("Podaj liczbę genów chromosomu (długość osobnika)")
num_of_dimensions = get_user_input("Podaj liczbę wymiarów")
num_generations = get_user_input("Podaj liczbę generacji (epok)")
sol_per_pop = get_user_input("Podaj liczbę rozwiązań (chromosomów) w populacji")
num_parents_mating = get_input_with_check(
    "Podaj liczbę chromosomów, które zostaną rodzicami",
    lambda x: x <= sol_per_pop
)

#Metody wybierane przez użytkownika
func_enum = get_function_enum()
func_min_max = get_min_max()
parent_selection_type = get_selection_type() #The parent selection type. Supported types are sss (for steady-state selection), rws (for roulette wheel selection), sus (for stochastic universal selection), rank (for rank selection), random (for random selection), and tournament (for tournament selection).
mutation_type = get_mutation_type()
crossover_type = get_crossover_type()

#q dla MULTIVARIATE
if crossover_type == CrossingMethodsBin.MULTIVARIATE:
    q = get_user_input("Podaj liczbę q dla krzyżowania wielowymiarowego")
else:
    q = None


#Długość osobnika * Liczba wymiarów
# num_genes = 48  #Number of genes in the solution/chromosome
# num_of_dimensions = 2  #Liczba wymiarów

## FLAGI
# func_enum = FunctionsOptions.RASTRIGIN  #Tutaj wybieramy funkcje do optymalizacji
# func_min_max = MinMax.MIN  #Tutaj wybieramy czy liczymy maximum czy minimim
# selected_crossover = CrossingMethodsBin.MULTIVARIATE  #Tutaj wybieramy funkcje crossover
# q = 3 #q dla MULTIVARIATE
# parent_selection_type = "tournament"  #(rws)(random)
# mutation_type = "swap"  #(random)(None)(swap)(inversion)(adaptive)


#Przypisanie parametrów
# num_generations = 80
# sol_per_pop = 80 #Number of solutions (i.e. chromosomes) within the population
# num_parents_mating = 50 #Number of solutions to be selected as parents

# q = 3 if crossover_type == CrossingMethodsBin.MULTIVARIATE else None  # q dla MULTIVARIATE

#When mutation_type='adaptive', then list/tuple/numpy.ndarray is expected
if mutation_type == "adaptive":
    mutation_num_genes = numpy.ones(num_genes, dtype=int)  # Lista o długości równej liczbie genów
else:
    mutation_num_genes = 1  # Pojedyncza wartość całkowita

func = bf.Rastrigin(n_dimensions=num_of_dimensions) \
    if func_enum == FunctionsOptions.RASTRIGIN \
    else bf.Schwefel(n_dimensions=num_of_dimensions)

decode_start = func.suggested_bounds()[0][0]  #zakres początkowy w szukanej funkcji
decode_end = func.suggested_bounds()[1][0]  ##zakres końcowy w szukanej funkcji

# crossover_type = "uniform"  #(single_point)(two_points)(uniform)
match (crossover_type):  #przypisanie własnych funkcji crossover
    case CrossingMethodsBin.BUILD_IN_SINGLE_POINT:
        crossover_type = CrossingMethodsBin.BUILD_IN_SINGLE_POINT.value
        crossover_name = CrossingMethodsBin.BUILD_IN_SINGLE_POINT.value
    case CrossingMethodsBin.BUILD_IN_DOUBLE_POINT:
        crossover_type = CrossingMethodsBin.BUILD_IN_DOUBLE_POINT.value
        crossover_name = CrossingMethodsBin.BUILD_IN_DOUBLE_POINT.value
    case CrossingMethodsBin.BUILD_IN_UNIFORM:
        crossover_type = CrossingMethodsBin.BUILD_IN_UNIFORM.value
        crossover_name = CrossingMethodsBin.BUILD_IN_UNIFORM.value
    case CrossingMethodsBin.SINGLE_POINT:
        crossover_type = SinglePointCrossover
        crossover_name = CrossingMethodsBin.SINGLE_POINT_STRING.value
    case CrossingMethodsBin.DOUBLE_POINT:
        crossover_type = TwoPointCrossover
        crossover_name = CrossingMethodsBin.DOUBLE_POINT_STRING.value
    case CrossingMethodsBin.TRIPLE_POINT:
        crossover_type = ThreePointCrossover
        crossover_name = CrossingMethodsBin.TRIPLE_POINT_STRING.value
    case CrossingMethodsBin.UNIFORM:
        crossover_type = UniformCrossover
        crossover_name = CrossingMethodsBin.UNIFORM_STRING.value
    case CrossingMethodsBin.GRAIN:
        crossover_type = GrainCrossover
        crossover_name = CrossingMethodsBin.GRAIN_STRING.value
    case CrossingMethodsBin.SCANNING:
        crossover_type = ScanningCrossover(num_of_dimensions).crossover
        crossover_name = CrossingMethodsBin.SCANNING_STRING.value
    case CrossingMethodsBin.PARTIAL:
        crossover_type = PartialCopyCrossover
        crossover_name = CrossingMethodsBin.PARTIAL_STRING.value
    case CrossingMethodsBin.MULTIVARIATE:
        crossover_type = MultivariateCrossover(q).crossover
        crossover_name = CrossingMethodsBin.MULTIVARIATE_STRING.value

def fitness_func_min(ga_instance, solution,
                     solution_idx):
    actual_value = get_actual_values(solution, decode_start, decode_end, num_of_dimensions)
    return 1. / func(actual_value)


def fitness_func_max(ga_instance, solution,
                     solution_idx):
    actual_value = get_actual_values(solution, decode_start, decode_end, num_of_dimensions)
    return func(actual_value)


fitness_function = fitness_func_min if func_min_max == MinMax.MIN else fitness_func_max

##(start) Parametry gwarantujące nam rozwiązanie binarne
init_range_low = 0 #The lower value of the random range from which the gene values in the initial population are selected
init_range_high = 2 # The upper value of the random range from which the gene values in the initial population are selected
#mutation_num_genes = 1
gene_type = "int"
##(end) Parametry gwarantujące nam rozwiązanie binarne

#Konfiguracja logowania

level = logging.DEBUG
name = 'logfile.txt'
logger = logging.getLogger(name)
logger.setLevel(level)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_format = logging.Formatter('%(message)s')
console_handler.setFormatter(console_format)
logger.addHandler(console_handler)

avg_fitness = []
std_fitness = []
best_fit = []

def get_time_for_bin():
    ga_instance = pygad.GA(num_generations=num_generations,
                           sol_per_pop=sol_per_pop,
                           num_parents_mating=num_parents_mating,
                           num_genes=num_genes,
                           fitness_func=fitness_function,
                           init_range_low=init_range_low,
                           init_range_high=init_range_high,
                           mutation_num_genes=mutation_num_genes,
                           parent_selection_type=parent_selection_type,
                           crossover_type=crossover_type,
                           mutation_type=mutation_type,
                           keep_elitism=1,
                           K_tournament=3,
                           gene_type=int,
                           random_mutation_max_val=1,
                           random_mutation_min_val=0,
                           parallel_processing=['thread', 4])
    start_time = time.time()

    ga_instance.run()

    end_time = time.time()
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    actual_value = get_actual_values(solution, decode_start, decode_end, num_of_dimensions)
    return {'time': end_time - start_time, 'sol': func(actual_value)}

