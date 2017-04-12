import array
import random

import numpy
import multiprocessing 

from deap import algorithms
from deap import base
from deap import creator
from deap import tools

from individual import Individual, initIndividual
from fitness import Fitness
from mutation import Mutation 
from crossover import Crossover

creator.create("FitnessMax", base.Fitness, weights=(-1.0,))
creator.create("Individual", Individual, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Structure initializers
toolbox.register("individual", initIndividual, creator.Individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

pool = multiprocessing.Pool(5)
toolbox.register("map", pool.map)

fit = Fitness("data/CO-nrm-part1.train.csv")
mut = Mutation()
cross = Crossover()

toolbox.register("evaluate", fit.evaluate)
toolbox.register("mate", cross.cxOnePoint)
toolbox.register("mutate", mut.mutate)
toolbox.register("select", tools.selTournament, tournsize=3)

def main():
    # random.seed(64)
    
    pop = toolbox.population(n=10)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)
    
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.6, mutpb=0.2, ngen=10, 
                                   stats=stats, halloffame=hof, verbose=True)
    
    return pop, log, hof


if __name__ == "__main__":
    pop, log, hof = main() 

    network = hof[0].createNetwork()
    network.summary()
    print( hof[0].fitness )
