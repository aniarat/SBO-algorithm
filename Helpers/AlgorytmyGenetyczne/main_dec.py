import logging

import pygad
import numpy
import benchmark_functions as bf
import matplotlib.pyplot as plt
import os
import time

from Helpers.AlgorytmyGenetyczne.Consts.enums import CrossingMethodsDec, FunctionsOptions, MinMax
from Helpers.AlgorytmyGenetyczne.Helpers.crossingMethodsDec import SingleArithmeticalCrossover, ArithmeticalCrossover, \
    LinearCrossover, BlendCrossoverAlfaBeta, BlendCrossoverAlfa, AverageCrossover, SimpleCrossover, RandomCrossover
from Helpers.AlgorytmyGenetyczne.Helpers.mutationMethods import GaussMutation


############ RZECZYWISTA ############
print("\n---Reprezentacja rzeczywista---\n")
#Funkcje do pobierania wartości od użytkownika
def choose_option(prompt, options, default=0):
    print(prompt)
    for i, option in enumerate(options, 1):
        print(f"{i}. {option}")
    choice = input("Wybierz opcję: ")
    if choice == '':
        return options[default]
    if not choice.isdigit() or int(choice) < 1 or int(choice) > len(options):
        print("Nieprawidłowy wybór. Wybierz numer opcji z listy.")
    else:
        return options[int(choice) - 1]


def get_selection_type():
    options = ["rws", "random", "tournament"]
    return choose_option("Wybierz metodę selekcji:", options)


def get_mutation_type():
    options = ["random", "swap", "inversion", "adaptive", "gauss"]
    return choose_option("Wybierz rodzaj mutacji:", options)


def get_crossover_type():
    options = [
        CrossingMethodsDec.SINGLE_POINT_ARITHMETIC,
        CrossingMethodsDec.ARITHMETIC,
        CrossingMethodsDec.LINEAR,
        CrossingMethodsDec.BLEND_ALFA_BETA,
        CrossingMethodsDec.BLEND_ALFA,
        CrossingMethodsDec.AVERAGE,
        CrossingMethodsDec.SIMPLE,
        CrossingMethodsDec.RANDOM
    ]
    return choose_option("Wybierz typ krzyżowania:", options)


def get_function_enum():
    options = [FunctionsOptions.RASTRIGIN, FunctionsOptions.SCHWEFEL]
    return choose_option("Wybierz funkcję optymalizacji:", options)


def get_min_max():
    options = [MinMax.MIN, MinMax.MAX]
    return choose_option("Wybierz minimalizację lub maksymalizację:", options)


def get_user_input(prompt, default_value=None, input_type=int):
    while True:
        try:
            value = input(f"{prompt}: ")
            if value == '' and default_value is not None:
                return input_type(default_value)
            value = input_type(value)
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


def get_input_with_check(prompt, default, condition):
    value = get_user_input(prompt, default)
    while not condition(value):
        print(f"Nieprawidłowa wartość: {value}. Proszę, spróbuj ponownie.")
        value = get_user_input(prompt)
    return value


#Parametry podawane przez użytkownika
# num_genes = get_user_input("Podaj liczbę genów chromosomu (długość osobnika) (24)", 24)
num_of_dimensions = get_user_input("Podaj liczbę wymiarów (2)", 2)
num_generations = get_user_input("Podaj liczbę generacji (epok)(100)", 100)
sol_per_pop = get_user_input("Podaj Wielkość populacji(100) ", 100)
num_parents_mating = get_input_with_check(
    "Podaj liczbę osobników, które zostaną rodzicami(50%)",
    sol_per_pop / 2,
    lambda x: x <= sol_per_pop
)

#Metody wybierane przez użytkownika
func_enum = get_function_enum()
func_min_max = get_min_max()
parent_selection_type = get_selection_type()  #The parent selection type. Supported types are sss (for steady-state selection), rws (for roulette wheel selection), sus (for stochastic universal selection), rank (for rank selection), random (for random selection), and tournament (for tournament selection).
mutation_type = get_mutation_type()
crossover_type = get_crossover_type()
# num_genes = 2  #Liczba wymiarów
## FLAGI
# func_enum = FunctionsOptions.RASTRIGIN  #Tutaj wybieramy funkcje do optymalizacji
# func_min_max = MinMax.MIN  #Tutaj wybieramy czy liczymy maximum czy minimim
# selected_crossover = CrossingMethodsDec.RANDOM  #Tutaj wybieramy funkcje crossover
# parent_selection_type = "tournament"  #(rws)(random)
# #Przypisanie parametrów
# num_generations = 80
# sol_per_pop = 80
# num_parents_mating = 50

#When mutation_type='adaptive', then list/tuple/numpy.ndarray is expected
if mutation_type == "adaptive":
    mutation_num_genes = numpy.ones(num_of_dimensions, dtype=int)  #Lista o długości równej liczbie genów
else:
    mutation_num_genes = 1

func = bf.Rastrigin(n_dimensions=num_of_dimensions) \
    if func_enum == FunctionsOptions.RASTRIGIN \
    else bf.Schwefel(n_dimensions=num_of_dimensions)

decode_start = func.suggested_bounds()[0][0]  #zakres początkowy w szukanej funkcji
decode_end = func.suggested_bounds()[1][0]  ##zakres końcowy w szukanej funkcji

mutation_name = None

if mutation_type == "gauss":
    mutation_type = GaussMutation(num_of_dimensions, 0, 1, decode_start, decode_end, 0.2).mutate
    mutation_name = "gauss"

# mutation_type = "swap"  #(random)(None)(swap)(inversion)(adaptive)
#mutation_type = GaussMutation(num_genes, 0, 1, decode_start, decode_end, 0.2).mutate

# crossover_type = "uniform"  #(single_point)(two_points)(uniform)
match (crossover_type):  #przypisanie własnych funckji crossover
    case CrossingMethodsDec.SINGLE_POINT_ARITHMETIC:
        crossover_type = SingleArithmeticalCrossover
        crossover_name = CrossingMethodsDec.SINGLE_POINT_ARITHMETIC_STRING.value
    case CrossingMethodsDec.ARITHMETIC:
        crossover_type = ArithmeticalCrossover
        crossover_name = CrossingMethodsDec.ARITHMETIC_STRING.value
    case CrossingMethodsDec.LINEAR:
        crossover_type = LinearCrossover(func).crossover
        crossover_name = CrossingMethodsDec.LINEAR_STRING.value
    case CrossingMethodsDec.BLEND_ALFA_BETA:
        crossover_type = BlendCrossoverAlfaBeta
        crossover_name = CrossingMethodsDec.BLEND_ALFA_BETA_STRING.value
    case CrossingMethodsDec.BLEND_ALFA:
        crossover_type = BlendCrossoverAlfa
        crossover_name = CrossingMethodsDec.BLEND_ALFA_STRING.value
    case CrossingMethodsDec.AVERAGE:
        crossover_type = AverageCrossover
        crossover_name = CrossingMethodsDec.AVERAGE_STRING.value
    case CrossingMethodsDec.SIMPLE:
        crossover_type = SimpleCrossover
        crossover_name = CrossingMethodsDec.SIMPLE_STRING.value
    case CrossingMethodsDec.RANDOM:
        crossover_type = RandomCrossover
        crossover_name = CrossingMethodsDec.RANDOM_STRING.value


def fitness_func_min(ga_instance, solution,
                     solution_idx):
    if func(solution) == 0:
        return numpy.inf
    fitness = 1. / func(solution)
    return fitness


def fitness_func_max(ga_instance, solution,
                     solution_idx):
    fitness = func(solution)
    return fitness


fitness_function = fitness_func_min if func_min_max == MinMax.MIN else fitness_func_max

##(start) Parametry gwarantujące nam rozwiązanie binarne
init_range_low = decode_start
init_range_high = decode_end
mutation_num_genes = 2
gene_type = "float"

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
best_fit = []
std_fitness = []

solution_times = []

def get_time_for_dec():
    ga_instance = pygad.GA(num_generations=num_generations,
                           sol_per_pop=sol_per_pop,
                           num_parents_mating=num_parents_mating,
                           num_genes=num_of_dimensions,
                           fitness_func=fitness_function,
                           init_range_low=init_range_low,
                           init_range_high=init_range_high,
                           mutation_num_genes=mutation_num_genes,
                           parent_selection_type=parent_selection_type,
                           crossover_type=crossover_type,
                           mutation_type=mutation_type,
                           keep_elitism=1,
                           K_tournament=3,
                           gene_type=float,
                           random_mutation_max_val=1,
                           random_mutation_min_val=0,
                           parallel_processing=['thread', 4])
    start_time = time.time()

    ga_instance.run()

    end_time = time.time()
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    return {'time': end_time - start_time, 'sol': func(solution)}
