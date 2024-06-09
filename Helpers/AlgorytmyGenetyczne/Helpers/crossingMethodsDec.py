import copy
import random

import numpy as np


def SingleArithmeticalCrossover(population: np.ndarray, offspring_size: tuple, ga_instance):
    alpha = 0.3
    offspring = []
    n_offspring = offspring_size[0]
    #n_spec = offspring_size[1]
    n_spec = population.shape[0]
    while len(offspring) != n_offspring:
        p1_idx = random.randint(0, n_spec - 1)
        p2_idx = random.randint(0, n_spec - 1)
        parent1, parent2 = population[p1_idx], population[p2_idx]
        gen_idx = np.random.randint(0, len(parent1))
        child1, child2 = copy.copy(parent1), copy.copy(parent2)
        p1_gen, p2_gen = parent1[gen_idx], parent2[gen_idx]
        child1[gen_idx] = (1 - alpha) * p1_gen + alpha * p2_gen
        child2[gen_idx] = (1 - alpha) * p1_gen + alpha * p2_gen
        offspring.append(child1)
        offspring.append(child2)
        if len(offspring) == n_offspring + 1:
            return np.array(offspring[:-1])
    return np.array(offspring)


def ArithmeticalCrossover(population: np.ndarray, offspring_size: tuple, ga_instance):
    alpha = 0.3
    offspring = []
    n_offspring = offspring_size[0]
    #n_spec = offspring_size[1]
    n_spec = population.shape[0]
    while len(offspring) != n_offspring:
        p1_idx = random.randint(0, n_spec - 1)
        p2_idx = random.randint(0, n_spec - 1)
        parent1, parent2 = population[p1_idx], population[p2_idx]
        child1 = np.array([(1 - alpha) * parent2[x] + alpha * parent1[x] for x in range(len(parent1))])
        child2 = np.array([(1 - alpha) * parent1[x] + alpha * parent2[x] for x in range(len(parent1))])
        offspring.append(child1)
        offspring.append(child2)
        if len(offspring) == n_offspring + 1:
            return np.array(offspring[:-1])
    return np.array(offspring)


def BlendCrossoverAlfaBeta(population: np.ndarray, offspring_size: tuple, ga_instance):
    offspring = []
    n_offspring = offspring_size[0]
    #n_spec = offspring_size[1]
    n_spec = population.shape[0]
    while len(offspring) != n_offspring:
        p1_idx = random.randint(0, n_spec - 1)
        p2_idx = random.randint(0, n_spec - 1)
        p1, p2 = population[p1_idx], population[p2_idx]
        child1 = np.empty_like(p1)
        child2 = np.empty_like(p1)

        alpha = np.random.uniform(low=0, high=1)
        beta = np.random.uniform(low=0, high=1)

        for i in range(len(p1)):
            x1 = p1[i]
            x2 = p2[i]
            distancia = abs(x1 - x2)

            u1 = np.random.uniform(min(x1, x2) - alpha * distancia, max(x1, x2) + beta * distancia)
            u2 = np.random.uniform(min(x1, x2) - alpha * distancia, max(x1, x2) + beta * distancia)

            child1[i] = u1
            child2[i] = u2
        offspring.append(child1)
        offspring.append(child2)
        if len(offspring) == n_offspring + 1:
            return np.array(offspring[:-1])
    return np.array(offspring)


def BlendCrossoverAlfa(population: np.ndarray, offspring_size: tuple, ga_instance):
    offspring = []
    n_offspring = offspring_size[0]
    #n_spec = offspring_size[1]
    n_spec = population.shape[0]
    while len(offspring) != n_offspring:
        p1_idx = random.randint(0, n_spec - 1)
        p2_idx = random.randint(0, n_spec - 1)
        p1, p2 = population[p1_idx], population[p2_idx]
        child1 = np.empty_like(p1)
        child2 = np.empty_like(p1)
        alpha = np.random.uniform(low=0, high=1)
        for i in range(len(p1)):
            x1 = p1[i]
            x2 = p2[i]
            distancia = abs(x1 - x2)
            u1 = np.random.uniform(min(x1, x2) - alpha * distancia, max(x1, x2) + alpha * distancia)
            u2 = np.random.uniform(min(x1, x2) - alpha * distancia, max(x1, x2) + alpha * distancia)

            child1[i] = u1
            child2[i] = u2

        offspring.append(child1)
        offspring.append(child2)
        if len(offspring) == n_offspring + 1:
            return np.array(offspring[:-1])
    return np.array(offspring)


def AverageCrossover(population: np.ndarray, offspring_size: tuple, ga_instance):
    offspring = []
    n_offspring = offspring_size[0]
    n_spec = offspring_size[1]
    while len(offspring) != n_offspring:
        child = np.array([np.mean([values for values in population[:, x]]) for x in range(len(population[0]))])
        offspring.append(child)
        if len(offspring) == n_offspring + 1:
            return np.array(offspring[:-1])
    return np.array(offspring)


def SimpleCrossover(population: np.ndarray, offspring_size: tuple, ga_instance):
    offspring = []
    n_offspring = offspring_size[0]
    #n_spec = offspring_size[1]
    n_spec = population.shape[0]
    while len(offspring) != n_offspring:
        alpha = np.random.uniform(low=0, high=1)
        p1_idx = random.randint(0, n_spec - 1)
        p2_idx = random.randint(0, n_spec - 1)
        p1, p2 = population[p1_idx], population[p2_idx]
        child1 = []
        child2 = []
        crossing_point = np.random.randint(1, len(p1))

        for i in range(0, crossing_point):
            child1.append(p1[i])
            child2.append(p2[i])
        for i in range(crossing_point, len(p1)):
            child1.append(alpha * p2[i] + (1 - alpha) * p1[i])
            child2.append(alpha * p1[i] + (1 - alpha) * p2[i])

        offspring.append(child1)
        offspring.append(child2)
        if len(offspring) == n_offspring + 1:
            return np.array(offspring[:-1])
    return np.array(offspring)


def RandomCrossover(population: np.ndarray, offspring_size: tuple, ga_instance):
    offspring = []
    n_offspring = offspring_size[0]
    #n_spec = offspring_size[1]
    n_spec = population.shape[0]
    n_dim = len(population[0])
    while len(offspring) != n_offspring:
        p1_idx = random.randint(0, n_spec - 1)
        p2_idx = random.randint(0, n_spec - 1)
        p1, p2 = population[p1_idx], population[p2_idx]
        chromosome_Z = np.array([np.random.uniform() for i in range(n_dim)])
        chromosome_W = np.array([np.random.uniform() for i in range(n_dim)])

        crossover_point_X = np.random.randint(0, n_dim)
        crossover_point_Y = np.random.randint(0, n_dim)

        child1 = np.concatenate((p1[:crossover_point_X], chromosome_Z[crossover_point_X:]))
        child2 = np.concatenate((p2[:crossover_point_Y], chromosome_W[crossover_point_Y:]))

        offspring.append(child1)
        offspring.append(child2)
        if len(offspring) == n_offspring + 1:
            return np.array(offspring[:-1])
    return np.array(offspring)


class CrossoverMethod:
    def crossover(self, population):
        pass


class LinearCrossover(CrossoverMethod):
    def __init__(self, func):
        self.func = func

    def crossover(self, population: np.ndarray, offspring_size: tuple, ga_instance):
        offspring = []
        n_offspring = offspring_size[0]
        #n_spec = offspring_size[1]
        n_spec = population.shape[0]
        while len(offspring) != n_offspring:
            p1_idx = random.randint(0, n_spec - 1)
            p2_idx = random.randint(0, n_spec - 1)
            p1, p2 = population[p1_idx], population[p2_idx]
            Z = np.array([p1[x] / 2 + p2[x] / 2 for x in range(len(p1))])
            V = np.array([p1[x] * 3 / 2 + p2[x] / (-2) for x in range(len(p1))])
            W = np.array([p1[x] / (-2) + p2[x] * 3 / 2 for x in range(len(p1))])
            values = [self.func(Z), self.func(W), self.func(V)]
            zip = np.column_stack((values, [Z, W, V]))
            sorted = zip[zip[:, 0].argsort()]
            best_linear = np.array([spec[1:] for spec in sorted[:2]])
            offspring.append(best_linear[0])
            offspring.append(best_linear[1])
            if len(offspring) == n_offspring + 1:
                return np.array(offspring[:-1])
        return np.array(offspring)

    # def crossover(self, population: np.ndarray, offspring_size: tuple, ga_instance):
    #     offspring = []
    #     n_offspring = offspring_size[0]
    #     n_spec = offspring_size[1]
    #     while len(offspring) != n_offspring:
    #         p1_idx = random.randint(0, n_spec - 1)
    #         p2_idx = random.randint(0, n_spec - 1)
    #         p1, p2 = population[p1_idx], population[p2_idx]
    #         Z = np.array([p1[x] / 2 + p2[x] / 2 for x in range(len(p1))])
    #         V = np.array([p1[x] * 3 / 2 + p2[x] / (-2) for x in range(len(p1))])
    #         W = np.array([p1[x] / (-2) + p2[x] * 3 / 2 for x in range(len(p1))])
    #         values = [self.func(Z), self.func(W), self.func(V)]
    #         zip = np.column_stack((values, [Z, W, V]))
    #         sorted = zip[zip[:, 0].argsort()]
    #         best_linear = np.array([spec[1:] for spec in sorted[:2]])
    #         offspring.append(best_linear[0])
    #         offspring.append(best_linear[1])
    #         if len(offspring) == n_offspring + 1:
    #             return np.array(offspring[:-1])
    #     return np.array(offspring)
