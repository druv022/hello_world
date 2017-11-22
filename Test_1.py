from random import randint,random
import numpy as np
from functools import reduce

def individual(length, min, max):
    'Create a member of the population.'
    return [randint(min,max) for x in range(length)]

def population(count,length,min, max):
    'Create a population of given count and the each individual is of given length withing random min nad max'
    return [individual(length,min,max,) for i in range(count)]

def fitness(individual,target):
    'Determine the fitness of the individual'
    fitness_ = reduce(lambda x,y: x+y,individual)
    #print(fitness_,target)
    return abs(fitness_-target)

def grade(population,target):
    'Take average of fitness of a population'
    total = reduce(lambda x,y: x + y,[fitness(i,target) for i in population] )
    return total/len(population)


def evolve(population,target,retain = 0.2,random_select=0.05,mutate =0.01):
    graded = [(fitness(x,target),x) for x in population]
    graded = [x[1] for x in sorted(graded)]
    retain_length = int(len(graded) * retain)
    parents = graded[:retain_length]

    # select random parents with lower fitness
    for i in graded[retain_length:]:
        if random_select > random():
            parents.append(i)

    # mutation
    for i in parents:
        if mutate > random():
            pos_to_mutate = randint(0,len(i)-1)
            # Some mutation strategy
            i[pos_to_mutate] = randint(min(i),max(i))

    #crossover parents to create children
    parents_length = len(parents)
    desired_length = len(population)-parents_length
    children = []
    while len(children) < desired_length:
        male = randint(0,parents_length-1)
        female = randint(0,parents_length-1)
        if male != female:
            male = parents[male]
            female = parents[female]
            half = int(len(male)/2)
            child = male[:half] + female[half:]
            children.append(child)

    parents.extend(children)
    return parents


#print(individual(5,0,100))
#print(population(3,5,0,100))
# x = individual(5,0,100)
# print(fitness(x,200))
#pop = population(5,5,0,100)
#target = 200
#print(grade(pop,target))
#print(evolve(pop,target))

target = 371
p_count = 100
i_length = 5
i_min = 0
i_max = 100
p = population(p_count,i_length,i_min,i_max)
fitness_history = [grade(p,target)]

for i in range(100):
    p = evolve(p,target)
    fitness_history.append(grade(p,target))

[print(i) for i in fitness_history]