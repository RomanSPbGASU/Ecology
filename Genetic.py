from __future__ import annotations
from collections import UserList
from typing import Iterable, Iterator, Sequence
import numpy as np
from numpy.random import randint
from random import random as rnd
from random import gauss, randrange, sample, choices, shuffle, uniform
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
    RANDOM = 'Random'


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

    def copy(self) -> Individual:
        new = Individual()
        new.genes = self.genes
        return new


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
        self.calc()

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

    def calc(self):
        self.calc_normalized_fitnesses()
        self.calc_cumulative_sum()

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


def selection(generation: Generation, method: SelectionMethod = SelectionMethod.FITNEST_HALF) -> Generation:
    half_count = int(len(generation.individuals) // 2)

    new_population = set()

    if method == SelectionMethod.ROULETTE_WHEEL:
        while len(new_population.individuals) != half_count:
            new_population.add(choices(generation, cum_weights=generation.cumulative_sums))

    elif method == SelectionMethod.FITNEST_HALF:
        generation.data.sort(key=lambda member: member.fitness)
        new_population.individuals = generation.individuals[half_count]

    elif method == SelectionMethod.RANDOM:
        new_population.individuals = sample(generation.individuals, half_count)

    return Generation(new_population)


def pairing(elit: Generation, selected: Generation, method: PairingMethod = PairingMethod.FITTEST):
    generation = Generation([elit, *selected.individuals])
    half_count = len(generation.individuals) // 2

    parents = []

    if method == PairingMethod.FITTEST:
        parents = [[generation.individuals[x],
                    generation.individuals[x + 1]]
                   for x in range(0, half_count, 2)]

    elif method == PairingMethod.RANDOM:
        parents = set()
        while len(parents) != half_count:
            parents.add(set(choices(generation, k=2)))
        parents = [parents]

    elif method == PairingMethod.WEIGHTED_RANDOM:
        parents = set()
        while len(parents) != half_count:
            parents.add(set(choices(generation, cum_weights=generation.cumulative_sums, k=2)))

    return parents


def mating(couple: [Individual], method: MatingMethod = MatingMethod.SINGLE_POINT):
    offsprings = []

    if method == MatingMethod.SINGLE_POINT:
        pivot_point = randint(1, len(couple[0].genes))
        couple[0][pivot_point:], couple[1][pivot_point:] = couple[1][pivot_point:], couple[0][pivot_point:]
        offsprings = couple

    elif method == MatingMethod.TWO_POINTS:
        pivot_points = sample(range(1, len(couple[0])), 2)
        couple[0][pivot_points[0]:], couple[1][pivot_points[0]:] = \
            couple[1][pivot_points[0]:], couple[0][pivot_points[0]:]
        couple[0][:pivot_points[1]], couple[1][:pivot_points[1]] = \
            couple[1][:pivot_points[1]], couple[0][:pivot_points[1]]
        offsprings = couple

    elif method == MatingMethod.RANDOM:
        offsprings = [zip(shuffle(gen_pair for gen_pair in zip(individual.genes for individual in couple)))]

    return offsprings


def mutation(individual: Individual, upper_limit, lower_limit, mutation_rate=2,
             method: MutationMethod = MutationMethod.RESET, standard_deviation=0.001):
    gene_indexes = sample(range(len(individual.genes)), mutation_rate)
    mutated_individual = individual.copy()

    if method == MutationMethod.GAUSS:
        for x in gene_indexes:
            mutated_individual.genes[x] = \
                round(individual.genes[x] + gauss(0, standard_deviation), 1)

    if method == MutationMethod.RESET:
        for x in gene_indexes:
            mutated_individual.genes[x] = round(randint(lower_limit, upper_limit), 1)

    return mutated_individual


def next_generation(generation: Generation, upper_limit, lower_limit) -> Generation:
    elite_member = generation.pop(-1)

    selected = selection(generation)

    parents = pairing(elite_member, selected)
    offsprings = [[[mating(parents[x])
                    for x in range(len(parents))]
                   [y][z] for z in range(2)]
                  for y in range(len(parents))]
    offsprings1 = [offsprings[x][0]
                   for x in range(len(parents))]
    offsprings2 = [offsprings[x][1]
                   for x in range(len(parents))]

    unmutated = selected + offsprings1 + offsprings2

    mutated = Generation([mutation(unmutated_member, upper_limit, lower_limit) for unmutated_member in unmutated])
    mutated.append(elite_member)
    mutated.calc()
    mutated.sort(key=lambda member: member.fitness)

    return mutated


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


# Generations and fitness values will be written to this file
Result_file = 'GA_Results.txt'  # Creating the First Generation

generations = [Generation.initialize_generation(20, 4, 0, 10)]
fitness_average = np.average(generations[0].fitnesses)
fitness_max = np.max(generations[0].fitnesses)

res = open(Result_file, 'a')
res.write('\n' + str(generations[0].fitnesses) + '\n')
res.close()
finish = False
while not finish:
    if max(fitness_max) > 6:
        break
    if max(fitness_average) > 5:
        break
    if fitness_similarity_check(fitness_max, 50):
        break
    generations.append(next_generation(generations[-1], 1, 0))

    fitness_average = np.average(generations[-1].fitnesses)
    fitness_max = np.max(generations[-1].fitnesses)

    res = open(Result_file, 'a')
    res.write('\n' + str(generations[-1].fitnesses) + '\n')
    res.close()
