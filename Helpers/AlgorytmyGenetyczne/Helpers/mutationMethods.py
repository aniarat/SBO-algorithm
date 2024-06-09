import copy
import random

import numpy as np


class MutationMethod:
    def mutate(self, individual, mutation_rate):
        pass


class GaussMutation(MutationMethod):
    def __init__(self, number_of_dimensions, mean, sigma, start_value, end_value, mutation_rate):
        self.number_of_dimensions = number_of_dimensions
        self.mean = mean
        self.sigma = sigma
        self.mutation_rate = mutation_rate
        if start_value > end_value:
            self.start_value = end_value
            self.end_value = start_value
        else:
            self.start_value = start_value
            self.end_value = end_value

    def mutate(self, offspring, ga_instance):
        for x in range(len(offspring)):
            mutated_individual = offspring[x].copy()
            for i in range(self.number_of_dimensions):
                if random.random() < self.mutation_rate:
                    mutation_value = random.gauss(self.mean, self.sigma)
                    mutated_individual[i] += mutation_value
                    mutated_individual[i] = min(max(mutated_individual[i], self.start_value), self.end_value)
                else:
                    mutated_individual[i] = offspring[x][i]
            offspring[x] = mutated_individual
        return offspring

