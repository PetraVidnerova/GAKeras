import sys
import array
import random
import pickle
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
import alg 

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

def main(checkpoint_name=None):
    # random.seed(64)

    if checkpoint_name:
        # A file name has been given, then load the data from the file
        cp = pickle.load(open(checkpoint_name, "rb"))
        pop = cp["population"]
        start_gen = cp["generation"]
        hof = cp["halloffame"]
        logbook = cp["logbook"]
        random.setstate(cp["rndstate"])
    else:
        pop = toolbox.population(n=10)
        start_gen = 0
        hof = tools.HallOfFame(1)
        logbook = tools.Logbook()
    
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)
    
    pop, log = alg.myEASimple(pop, start_gen, toolbox, cxpb=0.6, mutpb=0.2, ngen=10, 
                              stats=stats, halloffame=hof, logbook=logbook, verbose=True)

    return pop, log, hof


if __name__ == "__main__":

    # first command line argument (if present) is the name
    # of the checkpoint file 
    if len(sys.argv) > 1:
        pop, log, hof = main(sys.argv[1])
    else:
        pop, log, hof = main()
    
    network = hof[0].createNetwork()
    network.summary()
    print( hof[0].fitness )
