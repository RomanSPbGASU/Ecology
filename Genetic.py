from __future__ import annotations
from collections import UserList

import numpy as np
from numpy.random import randint
from random import random as rnd
from random import gauss, randrange, sample
from enum import Enum


class SelectionMethod(Enum):
    FITNEST_HALF = 'Fitnest Half'
    ROULETTE_WHEEL = 'Roulette Wheel'
    RANDOM = 'Random'


class PairingMethod(Enum):
    FITTEST = 'Fittest'
    RANDOM = 'Random'
    WEIGHTED_RANDOM = 'Weighted Random'


class MatingMethod(Enum):
    SINGLE_POINT = 'Single Point'
    TWO_POINTS = 'Two Pionts'


class MutationMethod(Enum):
    RESET = 'Reset'
    GAUSS = 'Gauss'


# представляет набор генов для одной особи

class Individual:
    def __init__(self):
        self.genes = None

    @staticmethod
    def create(number_of_genes, upper_limit, lower_limit) -> Individual:
        individual = Individual()
        individual.fill(number_of_genes, upper_limit, lower_limit)
        return individual

    def fill(self, number_of_genes, upper_limit, lower_limit):
        self.genes = [round(rnd() * (upper_limit - lower_limit) + lower_limit, 1) for x in range(number_of_genes)]


class GenerationMember:
    def __init__(self, genes: Individual):
        self.genes = genes
        self.fitness = fitness_calculation(genes)
        self.normalized_fitness = None
        self.cumulative_sum = None

    def __get__(self, instance, owner):
        return instance


class Generation(UserList):
    def __init__(self, individuals: [Individual] = None):
        super().__init__()
        self.data = [GenerationMember(individual) for individual in individuals]
        self.calc_normalized_fitnesses()
        self.calc_cumulative_sum()

    @classmethod
    def initialize_generation(cls, number_of_individuals, number_of_genes, upper_limit, lower_limit) -> Generation:
        individuals = [Individual.create(number_of_genes, upper_limit, lower_limit) for x in
                       range(number_of_individuals)]
        generation = cls(individuals)
        generation.sort(k=lambda member: member.fitness)
        return generation

    def calc_normalized_fitnesses(self) -> None:
        total = sum(self.fitnesses)
        for member in self.data:
            member.normalized_fitness = member.fitness / sum
        self.data.sort(key=lambda member: member.normalized_fitness)

    def calc_cumulative_sum(self) -> None:
        if not self.normalized_fitnesses:
            self.calc_normalized_fitnesses()
        else:
            self.data.sort(key=lambda member: member.normalized_fitness)
        sum = 0
        for member in self.data:
            sum += member.normalized_fitness
            member.cumulative_sum = sum

    @property
    def individuals(self):
        return [member.individual for member in self.data]

    @property
    def fitnesses(self):
        return [member.fitness for member in self.data]

    @property
    def normalized_fitnesses(self):
        return [member.normalized_fitness for member in self.data]

    @property
    def cumulative_sums(self):
        return [member.cumulative_sum for member in self.data]


def fitness_calculation(individual: Individual):
    return sum(individual.genes)


def roulette(cum_sum, chance):
    return sorted([*cum_sum.copy(), chance]).index(chance)


def selection(generation: Generation, method: SelectionMethod = SelectionMethod.FITNEST_HALF) -> Generation:
    half_count = int(len(generation.individuals) // 2)

    new_population = set()

    if method == SelectionMethod.ROULETTE_WHEEL:
        while len(new_population.individuals) != half_count:
            new_population.add(generation.individuals[roulette(generation.cumulative_sums, rnd())])

    elif method == SelectionMethod.FITNEST_HALF:
        generation.data.sort(key=lambda member: member.fitness)
        new_population.individuals = generation.individuals[half_count]

    elif method == SelectionMethod.RANDOM:
        new_population.individuals = sample(generation.individuals, half_count)

    return Generation(new_population)


def pairing(elit: Generation, selected: Generation, method: PairingMethod = PairingMethod.FITTEST):
    generation = Generation()
    generation.individuals = [*elit.individuals, *selected.individuals]
    generation.fitness = [*elit.fitness, *selected.fitness]
    half_count = len(generation.individuals) // 2

    parents = []
    if method == PairingMethod.FITTEST:
        parents = [[generation.individuals[x],
                    generation.individuals[x + 1]]
                   for x in range(half_count)]

    elif method == PairingMethod.RANDOM:
        last_index = len(generation.individuals) - 1
        for x in range(half_count):
            parents.append(
                [generation.individuals[randint(0, last_index)],
                 generation.individuals[randint(0, last_index)]])
            while parents[x][0] == parents[x][1]:
                parents[x][1] = generation.individuals[
                    randint(0, last_index)]
    elif method == PairingMethod.WEIGHTED_RANDOM:
        normalized_fitness = sorted([generation.fitness[x] / sum(generation.fitness)
                                     for x in range(half_count)],
                                    reverse=True)
        cumulative_sum = np.array(normalized_fitness).cumsum()
        for x in range(half_count):
            parents.append(
                [generation.individuals[roulette(cumulative_sum, rnd())],
                 generation.individuals[roulette(cumulative_sum, rnd())]])
            while parents[x][0] == parents[x][1]:
                parents[x][1] = generation.individuals[
                    roulette(cumulative_sum, rnd())]
    else:
        raise ValueError
    return parents


def mating(parents, method: MatingMethod = MatingMethod.SINGLE_POINT):
    if method == MatingMethod.SINGLE_POINT:
        pivot_point = randint(1, len(parents[0]))
        offsprings = [parents[0][0:pivot_point] + parents[1][pivot_point:], parents[1]
        [0:pivot_point] + parents[0][pivot_point:]]
    elif method == MatingMethod.TWO_POINTS:
        pivot_point_1 = randint(1, len(parents[0] - 1))
        pivot_point_2 = randint(1, len(parents[0]))
        while pivot_point_2 < pivot_point_1:
            pivot_point_2 = randint(1, len(parents[0]))
        offsprings = [parents[0][0:pivot_point_1] +
                      parents[1][pivot_point_1:pivot_point_2] +
                      [parents[0][pivot_point_2:]], [parents[1][0:pivot_point_1] +
                                                     parents[0][pivot_point_1:pivot_point_2] +
                                                     [parents[1][pivot_point_2:]]]]
    else:
        raise ValueError
    return offsprings


def mutation(individual, upper_limit, lower_limit, mutation_rate=2,
             method: MutationMethod = MutationMethod.RESET, standard_deviation=0.001):
    gene = [randint(0, 7)]
    for x in range(mutation_rate - 1):
        gene.append(randint(0, 7))
        while len(set(gene)) < len(gene):
            gene[x] = randint(0, 7)
    mutated_individual = individual.copy()
    if method == MutationMethod.GAUSS:
        for x in range(mutation_rate):
            mutated_individual[x] = \
                round(individual[x] + gauss(0, standard_deviation), 1)
    if method == MutationMethod.RESET:
        for x in range(mutation_rate):
            mutated_individual[x] = round(rnd() * (upper_limit - lower_limit) + lower_limit, 1)
    return mutated_individual


def next_generation(generation: Generation, upper_limit, lower_limit):
    elit = Generation()
    elit.individuals = generation.individuals.pop(-1)
    elit.fitness = generation.fitness.pop(-1)

    selected = selection(generation)
    parents = pairing(elit, selected)
    offsprings = [[[mating(parents[x])
                    for x in range(len(parents))]
                   [y][z] for z in range(2)]
                  for y in range(len(parents))]
    offsprings1 = [offsprings[x][0]
                   for x in range(len(parents))]
    offsprings2 = [offsprings[x][1]
                   for x in range(len(parents))]

    unmutated = selected.individuals + offsprings1 + offsprings2

    mutated = [mutation(unmutated[x], upper_limit, lower_limit)
               for x in range(len(generation.individuals))]
    unsorted_individuals = mutated + [elit.individuals]
    unsorted_next_gen = [fitness_calculation(mutated[x]) for x in range(len(mutated))]
    unsorted_fitness = [unsorted_next_gen[x]
                        for x in range(len(generation.fitness))] + [elit.fitness]
    sorted_next_gen = sorted([[unsorted_individuals[x], unsorted_fitness[x]]
                              for x in range(len(unsorted_individuals))],
                             key=lambda x: x[1])

    next_gen = Generation()
    for x in range(len(sorted_next_gen)):
        next_gen.individuals.append(sorted_next_gen[x][0])
        next_gen.fitness.append(sorted_next_gen[x][1])
    generation.individuals.append(elit.individuals)
    generation.fitness.append(elit.fitness)
    return next_gen


def fitness_similarity_check(max_fitness, number_of_similarity):
    result = False
    similarity = 0
    for n in range(len(max_fitness) - 1):
        if max_fitness[n] == max_fitness[n + 1]:
            similarity += 1
        else:
            similarity = 0
    if similarity == number_of_similarity - 1:
        result = True
    return result


def first_generation(pop) -> Generation:
    fitness = [fitness_calculation(pop[x])
               for x in range(len(pop))]
    sorted_fitness = sorted([[pop[x], fitness[x]]
                             for x in range(len(pop))], key=lambda x: x[1])
    population = [sorted_fitness[x][0]
                  for x in range(len(sorted_fitness))]
    fitness = [sorted_fitness[x][1]
               for x in range(len(sorted_fitness))]
    generation = Generation()
    generation.individuals = population
    generation.fitness = sorted(fitness)
    return generation


# Generations and fitness values will be written to this file
Result_file = 'GA_Results.txt'  # Creating the First Generation

pop = population(20, 8, 1, 0)

gen = [first_generation(pop)]
fitness_avg = np.array([sum(gen[0].fitness) / len(gen[0].fitness)])
fitness_max = np.array([max(gen[0].fitness)])
res = open(Result_file, 'a')
res.write('\n' + str(gen) + '\n')
res.close()
finish = False
while not finish:
    if max(fitness_max) > 6:
        break
    if max(fitness_avg) > 5:
        break
    if fitness_similarity_check(fitness_max, 50):
        break
    gen.append(next_generation(gen[-1], 1, 0))
    fitness_avg = np.append(fitness_avg, sum(
        gen[-1].fitness) / len(gen[-1].fitness))
    fitness_max = np.append(fitness_max, max(gen[-1].fitness))
    res = open(Result_file, 'a')
    res.write('\n' + str(gen[-1]) + '\n')
    res.close()
