from __future__ import annotations

import copy
from collections import UserList
from enum import Enum
from random import gauss, sample, choices, shuffle, uniform
from typing import Callable, Collection, Union

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import use
from numpy import clip
from numpy.random import randint

use('TkAgg')

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
        self.genes = [uniform(lower_limit, upper_limit) for x in range(number_of_genes)]

    def copy(self) -> Individual:
        new = Individual()
        new.genes = self.genes
        return new


class GenerationMember:
    def __init__(self, individual: Individual):
        self.individual = individual
        self.fitness = None


class Generation(UserList):
    def __init__(self, individuals: [Individual] = None, fitness_function: Callable[[Collection], Num] = sum):
        super().__init__()
        self.data = [GenerationMember(individual) for individual in individuals]
        self._fitness_function = fitness_function
        self.calc_fitness()
        self.sort()


    @classmethod
    def initialize_generation(cls, number_of_individuals, number_of_genes, lower_limit, upper_limit) -> Generation:
        individuals = [Individual.create(number_of_genes, lower_limit, upper_limit) for x in
                       range(number_of_individuals)]
        generation = cls(individuals)
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

    def sort(self):
        UserList.sort(self, key=lambda member: member.fitness)


def fitness_calculation(individual: Individual):
    return sum(individual.genes)


def selection(generation: Generation, method: SelectionMethod = SelectionMethod.FITNEST_HALF) -> Generation:
    half_count = len(generation.individuals) // 2

    new_population = set()

    if method == SelectionMethod.ROULETTE_WHEEL:
        while len(new_population) != half_count:
            new_population.add(choices(generation, weights=generation.fitnesses))

    elif method == SelectionMethod.FITNEST_HALF:
        generation.sort()
        new_population.update(generation.individuals[half_count:])

    elif method == SelectionMethod.RANDOM:
        new_population.update(sample(generation.individuals, half_count))

    return Generation(list(new_population))


def pairing(generation: Generation, method: PairingMethod = PairingMethod.FITTEST) -> list:
    half_count = len(generation.individuals) // 2
    parents = []

    if method == PairingMethod.FITTEST:
        parents = [[generation.individuals[x],
                    generation.individuals[x + 1]]
                   for x in range(0, len(generation.individuals), 2)]

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
    offsprings = copy.deepcopy(couple)

    if method == MatingMethod.SINGLE_POINT:
        pivot_point = randint(1, len(offsprings[0].genes))
        genotype0 = offsprings[0].genes
        genotype1 = offsprings[1].genes
        genotype0[pivot_point:], genotype1[pivot_point:] = genotype1[pivot_point:], genotype0[pivot_point:]

    elif method == MatingMethod.TWO_POINTS:
        pivot_points = sample(range(1, len(couple[0])), 2)
        couple[0][pivot_points[0]:], couple[1][pivot_points[0]:] = \
            couple[1][pivot_points[0]:], couple[0][pivot_points[0]:]
        couple[0][:pivot_points[1]], couple[1][:pivot_points[1]] = \
            couple[1][:pivot_points[1]], couple[0][:pivot_points[1]]

    elif method == MatingMethod.RANDOM:
        offsprings = [zip(shuffle(gen_pair for gen_pair in zip(individual.genes for individual in offsprings)))]

    return offsprings


def mutation(individual: Individual, lower_limit, upper_limit, mutation_rate=2,
             method: MutationMethod = MutationMethod.RESET, standard_deviation=2):
    gene_indexes = sample(range(len(individual.genes)), mutation_rate)
    mutated_individual = individual.copy()

    if method == MutationMethod.GAUSS:
        for x in gene_indexes:
            mutated_individual.genes[x] = \
                clip(individual.genes[x] + gauss(0, standard_deviation), lower_limit, upper_limit)

    if method == MutationMethod.RESET:
        for x in gene_indexes:
            mutated_individual.genes[x] = uniform(lower_limit, upper_limit)

    return mutated_individual


def next_generation(generation: Generation, lower_limit, upper_limit, mutation_deviation) -> Generation:
    elite_member = generation.pop(-1)

    selected = selection(generation)

    parents = pairing(selected)

    offsprings = [[[mating(parents[x])
                    for x in range(len(parents))]
                   [y][z] for z in range(2)]
                  for y in range(len(parents))]
    offsprings = [individual for couple in offsprings for individual in couple]

    unmutated = [member.individual for member in selected] + offsprings

    mutated = Generation(
        [mutation(unmutated_member, lower_limit, upper_limit, 1, MutationMethod.GAUSS, mutation_deviation) for
         unmutated_member in unmutated], generation.fitness_function)
    mutated.append(elite_member)
    mutated.calc_fitness()
    mutated.sort()

    return mutated


def fitness_similarity_check(max_fitness, number_of_similarity):
    similarity = 0
    for n in range(len(max_fitness) - 1):
        if max_fitness[n] == max_fitness[n + 1]:
            similarity += 1
        else:
            similarity = 0
    return similarity == number_of_similarity - 1


def run(generation_count=21,
        genes_count=4,
        lower_limit=0,
        upper_limit=10,
        max_fitness_condition=39.,
        average_fitness_condition=35.,
        similarity_length=20,
        mutation_deviation=2.,
        fitness_function=sum,
        result_file='GA_Results.txt'):
    first_generation = Generation.initialize_generation(generation_count, genes_count, lower_limit, upper_limit)

    first_generation.fitness_function = fitness_function
    first_generation.calc_fitness()
    first_generation.sort()

    generations = [first_generation]
    fitness_average = [np.average(generations[0].fitnesses)]
    fitness_max = [np.max(generations[0].fitnesses)]

    res = open(result_file, 'w')
    res.write(f'\n{len(generations)}\t{max(fitness_average)}\t{max(fitness_max)}\t{generations[0].fitnesses}')
    res.close()

    while True:
        if max(fitness_max) > max_fitness_condition or \
                max(fitness_average) > average_fitness_condition or \
                fitness_similarity_check(fitness_max, similarity_length):
            break
        generations.append(next_generation(generations[-1], lower_limit, upper_limit, mutation_deviation))

        fitness_average.append(np.average(generations[-1].fitnesses))
        fitness_max.append(np.max(generations[-1].fitnesses))

        res = open(result_file, 'a')
        ans = f'\n{len(generations)}\t{max(fitness_average)}\t{max(fitness_max)}\t{generations[-1].fitnesses}'
        res.write(ans)
        res.close()

    plt.plot([np.average(generation.fitnesses) for generation in generations], color='#f12')
    plt.plot([max(generation.fitnesses) for generation in generations], color='#38a')
    plt.show()

    # data for the surface
    x = np.linspace(-2, 2, 50)
    X, Y = np.meshgrid(x, x)
    Z = fitness_function(X, Y)

    # data for the scatter
    xx = [generation.individuals[8].genes[0] for generation in generations]
    yy = [generation.individuals[8].genes[1] for generation in generations]
    zz = [generation.fitnesses[8] for generation in generations]

    zzz = []
    for i, z in enumerate(zz):
        zzz.append(fitness_function([xx[i], yy[i]]))
        print(z, zzz[i])
    #
    fig = plt.figure(figsize=(8, 6))
    ax = plt.axes(projection='3d')
    ax.scatter(xx, yy, zz, c='r', marker='o')
    ax.plot(xx, yy, zz, 'ro', alpha=0.5)  # note the 'ro' (no '-') and the alpha

    ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                    cmap='viridis', edgecolor='none')

    plt.ion()
    plt.show()
    print('This is it')


def fitness_function(variables):
    x, y = variables
    return surface(x, y)


def surface(x, y) -> float:
    z = np.exp(-np.sqrt(x ** 2 + y ** 2)) * np.cos(4 * x) * np.cos(4 * y)
    return z


# run(41, 2, -2, 2, 2, 0.7, 500, 0.2, fitness_function)
run(9, 2, -2, 2, 4, 3.95, 500, 0.2, sum)
