#!/usr/bin/env python3

import imdb
import os
import pickle
import random

import pandas as pd
import numpy as np
import random
from deap import base
from deap import creator
from deap import tools

from datetime import timedelta

CACHE = '/tmp/movie_cache'
SIZE=250

MAX_DURATION = 24 * 60 * 28
TOTAL_RATING = SIZE * 10

GENERATIONS=100
POPULATIONS = 500

WEIGHTS = (-1., -1./0.3)

# CTXB - probability with which two inviduals are crossed
CXPB = 0.5
# MUTPB -probability for mutating an individual
MUTPB = 0.2

# INDPB - independatd probability of gene to mutate
INDPB = 0.05

# The number of times to repeat init function
N = 1

# The number of individuals participating in each tournament
TOURNSIZE = 3


def generate_data(size):
    if os.path.exists(CACHE):
        print('Reading data from cache')
        with open(CACHE, "rb") as fp:
            return pickle.load(fp)

    data = []
    ia = imdb.IMDb()
    search = ia.get_top250_movies()
    for i in range(size):
        print(f"Generating {i+1}/{size}")
        movie_id = search[i].getID()
        movie = ia.get_movie(movie_id)
        data.append([movie['title'], int(movie['runtimes'][0]), movie['rating']])

    with open(CACHE, "wb") as fp:
        pickle.dump(data, fp)

    return data
    
def n_per_movie():
    return random.choices(range(0,2), k=SIZE)

def evaluate(individual):
    individual = individual[0]
    total_duration = sum(x*y for x,y in zip(duration_data,individual))
    total_rating = sum(x*y for x,y in zip(rating_data,individual))
    
    return (abs(total_duration - MAX_DURATION), abs(total_rating - TOTAL_RATING))

data = generate_data(SIZE)

movie_table = pd.DataFrame.from_records(data)
movie_table.columns = ['Name', 'Duration', 'Rating']

movie_data = movie_table[['Name', 'Duration', 'Rating']]
duration_data = list(movie_data['Duration'])
rating_data = list(movie_data['Rating'])

creator.create("FitnessMin", base.Fitness, weights=WEIGHTS)
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

toolbox.register("n_per_movie", n_per_movie)

toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.n_per_movie, n=N)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=INDPB)
toolbox.register("select", tools.selTournament, tournsize=TOURNSIZE)

result0 = open('/tmp/result0.txt', "w")
result1 = open('/tmp/result1.txt', "w")
results = (result0, result1)

def main():
    pop = toolbox.population(n=POPULATIONS)
    
    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    
    # Variable keeping track of the number of generations
    g = 0
    
    # Begin the evolution
    while g < GENERATIONS:
        print(f"GENERATION {g+1}/{GENERATIONS}", end="\r")
        g = g + 1
        
        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))
        
        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1[0], child2[0])
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant[0])
                del mutant.fitness.values
            
                
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
            
        pop[:] = offspring
        
        # Gather all the fitnesses in one list and print the stats
        for val in (0,1):
            fits = [ind.fitness.values[val] for ind in pop]
            
            length = len(pop)
            mean = sum(fits) / length
            sum2 = sum(x*x for x in fits)
            std = abs(sum2 / length - mean**2)**0.5
            
            print(f"{g}, {min(fits)}, {max(fits)}, {mean}, {std}", file=results[val])
    
    best = pop[np.argmin([sum(toolbox.evaluate(x)) for x in pop])]
    result0.close()
    result1.close()
    return best

best_solution = main()
movie_table['multivariate_choice'] = pd.Series(best_solution[0])
filtered_movie_table = movie_table[(movie_table.multivariate_choice != 0)]
print(filtered_movie_table)

dur = int(filtered_movie_table['Duration'].sum())
print(f"Total duration: {str(timedelta(minutes=dur))[:-3]}")

count = len(filtered_movie_table)
print(f"Average rating: {filtered_movie_table['Rating'].sum()/count}")
