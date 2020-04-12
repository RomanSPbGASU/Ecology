import numpy as np
from numpy.random import randint
from random import random as rnd
from random import gauss, randrange
from enum import Enum


def individual(number_of_genes, upper_limit, lower_limit):
    return [round(rnd() * (upper_limit - lower_limit) + lower_limit, 1) for x in range(number_of_genes)]


def population(number_of_individuals, number_of_genes, upper_limit, lower_limit):
    return [individual(number_of_genes, upper_limit, lower_limit) for x in range(number_of_individuals)]


def fitness_calculation(individual):
    return sum(individual)


def roulette(cum_sum, chance):
    variable = list(cum_sum.copy())
    variable.append(chance)
    variable = sorted(variable)
    return variable.index(chance)


class Method(Enum):
    FITNEST_HALF = 'Fitnest Half'
    ROULETTE_WHEEL = 'Roulette Wheel'
    RANDOM = 'Random'


class Generation(Enum):
    FITNESS = 'Fitness'
    CUMULATIVE_SUM = 'Cumulative Sum'
    NORMALIZED_FITNESS = 'Normalized Fitness'
    INDIVIDUALS = 'Individuals'


def selection(generation: Generation, method: Method = Method.FITNEST_HALF):
    # для каждого из фитнесов в генерэйшене вычислим его нормализованное значение разделив его на сумму всех фитнесов
    generation[Generation.NORMALIZED_FITNESS] = sorted(
        [fitness_value / sum(generation[Generation.FITNESS]) for fitness_value in generation[Generation.FITNESS]],
        reverse=True)
    generation[Generation.CUMULATIVE_SUM] = np.array(generation[Generation.NORMALIZED_FITNESS]).cumsum()
    if method == Method.ROULETTE_WHEEL:
        selected = []
        for x in range(len(generation[Generation.INDIVIDUALS]) // 2):
            selected.append(roulette(generation[Generation.CUMULATIVE_SUM], rnd()))
            while len(set(selected)) != len(selected):
                selected[x] = (roulette(generation[Generation.CUMULATIVE_SUM], rnd()))
                selected = {Generation.INDIVIDUALS: [generation[Generation.INDIVIDUALS][int(selected[x])] for x in
                                                     range(len(generation[Generation.INDIVIDUALS]) // 2)]}
    elif method == Method.FITNEST_HALF:
        selected_individuals = [generation[Generation.INDIVIDUALS][-x - 1] for x in
                                range(int(len(generation[Generation.INDIVIDUALS]) // 2))]
        selected_fitnesses = [generation[Generation.FITNESS][-x - 1] for x in
                              range(int(len(generation[Generation.INDIVIDUALS]) // 2))]
        selected = {Generation.INDIVIDUALS: selected_individuals, Generation.FITNESS: selected_fitnesses}
    elif method == Method.RANDOM:
        selected_individuals = [generation[Generation.INDIVIDUALS][randint(1, len(generation[Generation.FITNESS]))] for
                                x in range(int(len(generation[Generation.INDIVIDUALS]) // 2))]
        selected_fitnesses = [generation[Generation.FITNESS][-x - 1] for x in
                              range(int(len(generation[Generation.INDIVIDUALS]) // 2))]
        selected = {Generation.INDIVIDUALS: selected_individuals, Generation.FITNESS: selected_fitnesses}
    else:
        raise ValueError
    return selected


def pairing(elit, selected, method='Fittest'):
    individuals = [elit['Individuals']] + selected['Individuals']
    fitness = [elit['Fitness']] + selected['Fitness']
    parents = []
    if method == 'Fittest':
        parents = [[individuals[x], individuals[x + 1]]
                   for x in range(len(individuals) // 2)]
    elif method == 'Random':
        for x in range(len(individuals) // 2):
            parents.append(
                [individuals[randint(0, (len(individuals) - 1))],
                 individuals[randint(0, (len(individuals) - 1))]])
            while parents[x][0] == parents[x][1]:
                parents[x][1] = individuals[
                    randint(0, (len(individuals) - 1))]
    elif method == 'Weighted Random':
        normalized_fitness = sorted(
            [fitness[x] / sum(fitness)
             for x in range(len(individuals) // 2)], reverse=True)
        cummulitive_sum = np.array(normalized_fitness).cumsum()
        for x in range(len(individuals) // 2):
            parents.append(
                [individuals[roulette(cummulitive_sum, rnd())],
                 individuals[roulette(cummulitive_sum, rnd())]])
            while parents[x][0] == parents[x][1]:
                parents[x][1] = individuals[
                    roulette(cummulitive_sum, rnd())]
    else:
        raise ValueError
    return parents


def mating(parents, method='Single Point'):
    if method == 'Single Point':
        pivot_point = randint(1, len(parents[0]))
        offsprings = [parents[0][0:pivot_point] + parents[1][pivot_point:], parents[1]
        [0:pivot_point] + parents[0][pivot_point:]]
    elif method == 'Two Pionts':
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


def mutation(individual, upper_limit, lower_limit, muatation_rate=2,
             method='Reset', standard_deviation=0.001):
    gene = [randint(0, 7)]
    for x in range(muatation_rate - 1):
        gene.append(randint(0, 7))
        while len(set(gene)) < len(gene):
            gene[x] = randint(0, 7)
    mutated_individual = individual.copy()
    if method == 'Gauss':
        for x in range(muatation_rate):
            mutated_individual[x] = \
                round(individual[x] + gauss(0, standard_deviation), 1)
    if method == 'Reset':
        for x in range(muatation_rate):
            mutated_individual[x] = round(rnd() * \
                                          (upper_limit - lower_limit) + lower_limit, 1)
    return mutated_individual


def next_generation(gen, upper_limit, lower_limit):
    elit = {}
    next_gen = {}
    elit['Individuals'] = gen['Individuals'].pop(-1)
    elit['Fitness'] = gen['Fitness'].pop(-1)
    selected = selection(gen)
    parents = pairing(elit, selected)
    offsprings = [[[mating(parents[x])
                    for x in range(len(parents))]
                   [y][z] for z in range(2)]
                  for y in range(len(parents))]
    offsprings1 = [offsprings[x][0]
                   for x in range(len(parents))]
    offsprings2 = [offsprings[x][1]
                   for x in range(len(parents))]
    unmutated = selected['Individuals'] + offsprings1 + offsprings2
    mutated = [mutation(unmutated[x], upper_limit, lower_limit)
               for x in range(len(gen['Individuals']))]
    unsorted_individuals = mutated + [elit['Individuals']]
    unsorted_next_gen = \
        [fitness_calculation(mutated[x])
         for x in range(len(mutated))]
    unsorted_fitness = [unsorted_next_gen[x]
                        for x in range(len(gen['Fitness']))] + [elit['Fitness']]
    sorted_next_gen = \
        sorted([[unsorted_individuals[x], unsorted_fitness[x]]
                for x in range(len(unsorted_individuals))],
               key=lambda x: x[1])
    next_gen['Individuals'] = [sorted_next_gen[x][0]
                               for x in range(len(sorted_next_gen))]
    next_gen['Fitness'] = [sorted_next_gen[x][1]
                           for x in range(len(sorted_next_gen))]
    gen['Individuals'].append(elit['Individuals'])
    gen['Fitness'].append(elit['Fitness'])
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


# Generations and fitness values will be written to this file
Result_file = 'GA_Results.txt'  # Creating the First Generation


def first_generation(pop):
    fitness = [fitness_calculation(pop[x])
               for x in range(len(pop))]
    sorted_fitness = sorted([[pop[x], fitness[x]]
                             for x in range(len(pop))], key=lambda x: x[1])
    population = [sorted_fitness[x][0]
                  for x in range(len(sorted_fitness))]
    fitness = [sorted_fitness[x][1]
               for x in range(len(sorted_fitness))]
    return {'Individuals': population, 'Fitness': sorted(fitness)}


pop = population(20, 8, 1, 0)

gen = [first_generation(pop)]
fitness_avg = np.array([sum(gen[0]['Fitness']) /
                        len(gen[0]['Fitness'])])
fitness_max = np.array([max(gen[0]['Fitness'])])
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
        gen[-1]['Fitness']) / len(gen[-1]['Fitness']))
    fitness_max = np.append(fitness_max, max(gen[-1]['Fitness']))
    res = open(Result_file, 'a')
    res.write('\n' + str(gen[-1]) + '\n')
    res.close()
