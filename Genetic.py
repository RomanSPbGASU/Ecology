from __future__ import annotations

from collections import UserList
from enum import Enum
from random import gauss, sample, choices, shuffle
from typing import Callable, Collection, Union

import numpy as np
from numpy.random import randint

Num = Union[int, float]


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
    def create(number_of_genes, lower_limit, upper_limit) -> Individual:
        individual = Individual()
        individual.fill(number_of_genes, lower_limit, upper_limit)
        return individual

    def fill(self, number_of_genes, lower_limit, upper_limit):
        self.genes = [randint(lower_limit, upper_limit) for x in range(number_of_genes)]

    def copy(self) -> Individual:
        new = Individual()
        new.genes = self.genes
        return new


class GenerationMember:
    def __init__(self, individual: Individual):
        self.individual = individual
        self.fitness = None

    def __get__(self, instance, owner):
        return instance


class Generation(UserList):
    def __init__(self, individuals: [Individual] = None, fitness_function: Callable[[Collection], Num] = sum):
        super().__init__()
        self.data = [GenerationMember(individual) for individual in individuals]
        self._fitness_function = fitness_function
        self.calc_fitness()

    @classmethod
    def initialize_generation(cls, number_of_individuals, number_of_genes, lower_limit, upper_limit) -> Generation:
        individuals = [Individual.create(number_of_genes, lower_limit, upper_limit) for x in
                       range(number_of_individuals)]
        generation = cls(individuals)
        generation.sort(key=lambda member: member.fitness)
        return generation

    @property
    def fitness_function(self) -> Callable:
        return self._fitness_function

    @fitness_function.setter
    def fitness_function(self, function: Callable[[Collection], Num]):
        self._fitness_function = function

    def calc_fitness(self) -> None:
        for member in self.data:
            member.fitness = self.fitness_function(member.individual.genes)

    @property
    def individuals(self):
        return [member.individual for member in self.data]

    @property
    def fitnesses(self):
        return [member.fitness for member in self.data]


def fitness_calculation(individual: Individual):
    return sum(individual.genes)


def selection(generation: Generation, method: SelectionMethod = SelectionMethod.FITNEST_HALF) -> Generation:
    half_count = int(len(generation.individuals) // 2)

    new_population = set()

    if method == SelectionMethod.ROULETTE_WHEEL:
        while len(new_population) != half_count:
            new_population.add(choices(generation, weights=generation.fitnesses))

    elif method == SelectionMethod.FITNEST_HALF:
        generation.data.sort(key=lambda member: member.fitness)
        new_population.update(generation.individuals[half_count:])

    elif method == SelectionMethod.RANDOM:
        new_population.update(sample(generation.individuals, half_count))

    return Generation(new_population)


def pairing(generation: Generation, method: PairingMethod = PairingMethod.FITTEST):
    half_count = len(generation.individuals) // 2
    parents = []

    if method == PairingMethod.FITTEST:
        parents = [[generation.individuals[x],
                    generation.individuals[x + 1]]
                   for x in range(half_count)]

    elif method == PairingMethod.RANDOM:
        parents = set()
        while len(parents) != half_count:
            parents.add(set(choices(generation, k=2)))
        parents = [parents]

    elif method == PairingMethod.WEIGHTED_RANDOM:
        parents = set()
        while len(parents) != half_count:
            parents.add(set(choices(generation, weights=generation.fitnesses, k=2)))

    return parents


def mating(couple: [Individual], method: MatingMethod = MatingMethod.SINGLE_POINT):
    offsprings = []

    if method == MatingMethod.SINGLE_POINT:
        pivot_point = randint(1, len(couple[0].genes))
        genotype0 = couple[0].genes
        genotype1 = couple[1].genes
        genotype0[pivot_point:], genotype1[pivot_point:] = genotype1[pivot_point:], genotype0[pivot_point:]
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


def mutation(individual: Individual, lower_limit, upper_limit, mutation_rate=2,
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


def next_generation(generation: Generation, lower_limit, upper_limit) -> Generation:
    elite_member = generation.pop(-1)

    selected = selection(generation)

    parents = pairing(selected)
    offsprings = [[[mating(parents[x])
                    for x in range(len(parents))]
                   [y][z] for z in range(2)]
                  for y in range(len(parents))]
    offsprings1 = [offsprings[x][0]
                   for x in range(len(parents))]
    offsprings2 = [offsprings[x][1]
                   for x in range(len(parents))]

    unmutated = [member.individual for member in selected] + offsprings1 + offsprings2

    mutated = Generation([mutation(unmutated_member, lower_limit, upper_limit) for unmutated_member in unmutated])
    mutated.append(elite_member)
    mutated.calc_fitness()
    mutated.sort(key=lambda member: member.fitness)

    return mutated


def fitness_similarity_check(max_fitness, number_of_similarity):
    similarity = 0
    for n in range(len(max_fitness) - 1):
        if max_fitness[n] == max_fitness[n + 1]:
            similarity += 1
        else:
            similarity = 0
    return similarity == number_of_similarity - 1


# Generations and fitness values will be written to this file
Result_file = 'GA_Results.txt'  # Creating the First Generation

generations = [Generation.initialize_generation(20, 4, 0, 10)]
fitness_average = [np.average(generations[0].fitnesses)]
fitness_max = [np.max(generations[0].fitnesses)]

res = open(Result_file, 'a')
res.write(f'\n{generations[0].fitnesses}\t{max(fitness_average)}\t{max(fitness_max)}')
res.close()

while True:
    if max(fitness_max) > 60:
        break
    if max(fitness_average) > 50:
        break
    if fitness_similarity_check(fitness_max, 50):
        break
    generations.append(next_generation(generations[-1], 0, 10))

    fitness_average.append(np.average(generations[-1].fitnesses))
    fitness_max.append(np.max(generations[-1].fitnesses))

    res = open(Result_file, 'a')
    ans = f'\n{generations[-1].fitnesses}\t{max(fitness_average)}\t{max(fitness_max)}'
    res.write(ans)
    res.close()
