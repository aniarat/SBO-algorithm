import random
from typing import Tuple, List, Any

import numpy as np

from Helpers.AlgorytmyGenetyczne.Helpers.decimalBinaryMath import split


def TestCrossover(population: np.ndarray, offspring_size: tuple, ga_instance):
    print(population)
    print(offspring_size)
    return np.array(population)


def SinglePointCrossover(population: np.ndarray, offspring_size, ga_instance):
    offspring = []
    n_offspring = offspring_size[0]
    # n_spec = offspring_size[1]
    n_spec = population.shape[0]
    while len(offspring) != n_offspring:
        p1_idx = random.randint(0, n_spec - 1)
        p2_idx = random.randint(0, n_spec - 1)
        parent1, parent2 = population[p1_idx], population[p2_idx]
        crossing_point = random.randint(1, len(parent1) - 1)
        child1 = np.append(parent1[:crossing_point], parent2[crossing_point:])
        child2 = np.append(parent2[:crossing_point], parent1[crossing_point:])
        offspring.append(child1)
        offspring.append(child2)
        if len(offspring) == n_offspring + 1:
            return np.array(offspring[:-1])
    return np.array(offspring)
# def SinglePointCrossover(population: np.ndarray, offspring_size, ga_instance):
#     offspring = []
#     n_offspring = offspring_size[0]
#     n_genes = offspring_size[1]

#     while len(offspring) < n_offspring:
#         p1_idx = random.randint(0, population.shape[0] - 1)
#         p2_idx = random.randint(0, population.shape[0] - 1)
#         parent1, parent2 = population[p1_idx], population[p2_idx]

#         crossing_point = random.randint(1, len(parent1) - 1)

#         child1 = np.concatenate((parent1[:crossing_point], parent2[crossing_point:]))
#         child2 = np.concatenate((parent2[:crossing_point], parent1[crossing_point:]))

#         offspring.append(child1)
#         if len(offspring) < n_offspring:
#             offspring.append(child2)

#     return np.array(offspring)


def TwoPointCrossover(population: np.ndarray, offspring_size, ga_instance):
    offspring = []
    n_offspring = offspring_size[0]
    # n_spec = offspring_size[1]
    n_spec = population.shape[0]
    while len(offspring) != n_offspring:
        p1_idx = random.randint(0, n_spec - 1)
        p2_idx = random.randint(0, n_spec - 1)
        parent1, parent2 = population[p1_idx], population[p2_idx]
        crossing_point1, crossing_point2 = sorted(random.sample(range(1, len(parent1)), 2))
        child1 = np.append(np.append(parent1[:crossing_point1], parent2[crossing_point1:crossing_point2]),
                           parent1[crossing_point2:])
        child2 = np.append(np.append(parent2[:crossing_point1], parent1[crossing_point1:crossing_point2]),
                           parent2[crossing_point2:])
        offspring.append(child1)
        offspring.append(child2)
        if len(offspring) == n_offspring + 1:
            return np.array(offspring[:-1])
    return np.array(offspring)


def ThreePointCrossover(population: np.ndarray, offspring_size, ga_instance):
    offspring = []
    n_offspring = offspring_size[0]
    # n_spec = offspring_size[1]
    n_spec = population.shape[0]
    while len(offspring) != n_offspring:
        p1_idx = random.randint(0, n_spec - 1)
        p2_idx = random.randint(0, n_spec - 1)
        parent1, parent2 = population[p1_idx], population[p2_idx]
        crossing_point1, crossing_point2, crossing_point3 = sorted(random.sample(range(1, len(parent1)), 3))
        child1 = np.append(np.append(np.append(parent1[:crossing_point1], parent2[crossing_point1:crossing_point2]),
                                     parent1[crossing_point2:crossing_point3]), parent2[crossing_point3:])
        child2 = np.append(np.append(np.append(parent2[:crossing_point1], parent1[crossing_point1:crossing_point2]),
                                     parent2[crossing_point2:crossing_point3]), parent1[crossing_point3:])
        offspring.append(child1)
        offspring.append(child2)
        if len(offspring) == n_offspring + 1:
            return np.array(offspring[:-1])
    return np.array(offspring)


# def UniformCrossover(population: np.ndarray, offspring_size, ga_instance):
#     offspring = []
#     n_offspring = offspring_size[0]
#     n_spec = offspring_size[1]
#     while len(offspring) != n_offspring:
#         p1_idx = random.randint(0, n_spec)
#         p2_idx = random.randint(0, n_spec)
#         parent1, parent2 = population[p1_idx], population[p2_idx]
#         child1 = []
#         child2 = []
#         for gene1, gene2 in zip(parent1, parent2):
#             if random.random() <= 0.5:
#                 np.append(child1, gene2)
#                 np.append(child2, gene1)
#             else:
#                 np.append(child1, gene1)
#                 np.append(child2, gene2)
#         offspring.append(np.array(child1))
#         offspring.append(np.array(child2))
#         if len(offspring) == n_offspring + 1:
#             return np.array(offspring[:-1])
#     return np.array(offspring)

def UniformCrossover(population: np.ndarray, offspring_size, ga_instance):
    offspring = []
    n_offspring = offspring_size[0]
    n_genes = offspring_size[1]

    while len(offspring) < n_offspring:
        p1_idx = random.randint(0, population.shape[0] - 1)
        p2_idx = random.randint(0, population.shape[0] - 1)
        parent1, parent2 = population[p1_idx], population[p2_idx]

        child1, child2 = [], []

        for gene1, gene2 in zip(parent1, parent2):
            if random.random() < 0.5:
                child1.append(gene1)
                child2.append(gene2)
            else:
                child1.append(gene2)
                child2.append(gene1)

        offspring.append(child1)
        if len(offspring) < n_offspring:
            offspring.append(child2)

    return np.array(offspring[:n_offspring])


# def GrainCrossover(population: np.ndarray, offspring_size, ga_instance):
#     offspring = []
#     n_offspring = offspring_size[0]
#     n_spec = offspring_size[1]
#     while len(offspring) != n_offspring:
#         p1_idx = random.randint(0, n_spec)
#         p2_idx = random.randint(0, n_spec)
#         parent1, parent2 = population[p1_idx], population[p2_idx]
#         child = []
#         for gene1, gene2 in zip(parent1, parent2):
#             if random.random() <= 0.5:
#                 child.append(gene1)
#             else:
#                 child.append(gene2)
#         offspring.append(np.array(child))
#         if len(offspring) == n_offspring + 1:
#             return np.array(offspring[:-1])
#     return np.array(offspring)


def GrainCrossover(population: np.ndarray, offspring_size, ga_instance):
    offspring = []
    n_offspring = offspring_size[0]
    n_genes = offspring_size[1]

    while len(offspring) < n_offspring:
        # Losowe wybieranie dwóch rodziców z populacji
        p1_idx = random.randint(0, population.shape[0] - 1)
        p2_idx = random.randint(0, population.shape[0] - 1)
        parent1, parent2 = population[p1_idx], population[p2_idx]

        child = []

        for gene1, gene2 in zip(parent1, parent2):
            if random.random() < 0.5:
                child.append(gene1)
            else:
                child.append(gene2)

        offspring.append(child)

    return np.array(offspring)

def PartialCopyCrossover(population: np.ndarray, offspring_size, ga_instance):
    offspring = []
    n_offspring = offspring_size[0]
    # n_spec = offspring_size[1]
    n_spec = population.shape[0]

    while len(offspring) != n_offspring:
        p1_idx = random.randint(0, n_spec - 1)
        p2_idx = random.randint(0, n_spec - 1)
        parent1, parent2 = population[p1_idx], population[p2_idx]
        chromosome_length = len(parent1)
        cp1: int = random.randint(0, chromosome_length - 2)
        cp2: int = random.randint(cp1 + 1, chromosome_length - 1)
        child1 = np.append(np.append(parent1[:cp1], [parent1[j] if parent1[j] == 1 else
                                                     parent2[j] for j in range(cp1, cp2)]), parent1[cp2:])
        child2 = np.append(np.append(parent2[:cp1], [parent2[j] if parent2[j] == 1 else
                                                     parent1[j] for j in range(cp1, cp2)]), parent2[cp2:])
        offspring.append(np.array(child1))
        offspring.append(np.array(child2))
        if len(offspring) == n_offspring + 1:
            return np.array(offspring[:-1])
    return np.array(offspring)


class CrossoverMethod:
    def crossover(self, population):
        pass


class ScanningCrossover(CrossoverMethod):
    def __init__(self, number_of_dimensions):
        self.number_of_dimensions = number_of_dimensions

    def crossover(self, population: np.ndarray, offspring_size, ga_instance):
        offspring = []
        n_offspring = offspring_size[0]
        n_spec = offspring_size[1]
        # n_spec = population.shape[0]
        n_chrom = len(population[0])
        idx = 0
        while len(offspring) != n_offspring:
            parents = list(split(population[idx % n_chrom], self.number_of_dimensions))
            child = []
            for j in range(n_chrom):
                p_idx = random.randint(0, self.number_of_dimensions - 1)
                child.append(parents[p_idx][j % len(parents[p_idx])])
            offspring.append(np.array(child))
            idx += 1
        return np.array(offspring)


class MultivariateCrossover(CrossoverMethod):
    def __init__(self, q):
        self.q = q

    # def crossover(self, population: np.ndarray, offspring_size, ga_instance):
    #     offspring = []
    #     n_offspring = offspring_size[0]
    #     n_spec = offspring_size[1]
    #     while len(offspring) != n_offspring:
    #         p1_idx, p2_idx = random.randint(0, n_spec), random.randint(0, n_spec)
    #         parent1, parent2 = population[p1_idx], population[p2_idx]
    #         cp: int = random.randint(0, len(parent1) - 2)
    #         child1 = None
    #         child2 = None
    #         for j in range(self.q - 1):
    #             if random.random() <= 0.5:
    #                 child1 = np.append(parent1[:cp], parent2[cp:])
    #                 child2 = np.append(parent2[:cp], parent1[cp:])
    #             else:
    #                 child1 = parent1
    #                 child2 = parent2
    #         offspring.append(np.array(child1))
    #         offspring.append(np.array(child2))
    #         if len(offspring) == n_offspring + 1:
    #             return np.array(offspring[:-1])

    #     return np.array(offspring)

    def crossover(self, population: np.ndarray, offspring_size, ga_instance):
            offspring = []
            n_offspring = offspring_size[0]
            # n_spec = len(population)
            n_spec = population.shape[0]

            while len(offspring) < n_offspring:
                p1_idx = random.randint(0, n_spec - 1)
                p2_idx = random.randint(0, n_spec - 1)
                while p2_idx == p1_idx:
                    p2_idx = random.randint(0, n_spec - 1)

                parent1, parent2 = population[p1_idx], population[p2_idx]
                child1, child2 = parent1.copy(), parent2.copy()

                for _ in range(self.q - 1):
                    cp = random.randint(1, len(parent1) - 2)
                    if random.random() <= 0.5:
                        child1 = np.append(parent1[:cp], parent2[cp:])
                        child2 = np.append(parent2[:cp], parent1[cp:])
                    else:
                        child1, child2 = parent1, parent2

                offspring.append(child1)
                offspring.append(child2)

                if len(offspring) > n_offspring:
                    offspring = offspring[:n_offspring]

            return np.array(offspring)
