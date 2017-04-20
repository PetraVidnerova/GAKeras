import sys
import array
import random
import pickle
import numpy as np
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
from dataset import load_data

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

def main(id, checkpoint_name=None):
    # random.seed(64)

    if checkpoint_name:
        # A file name has been given, then load the data from the file
        cp = pickle.load(open(checkpoint_name, "rb"))
        pop = cp["population"]
        start_gen = cp["generation"] + 1
        hof = cp["halloffame"]
        logbook = cp["logbook"]
        random.setstate(cp["rndstate"])
    else:
        pop = toolbox.population(n=10)
        start_gen = 0
        hof = tools.HallOfFame(1)
        logbook = tools.Logbook()
    
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    pop, log = alg.myEASimple(pop, start_gen, toolbox, cxpb=0.6, mutpb=0.2, ngen=10, 
                              stats=stats, halloffame=hof, logbook=logbook, verbose=True,
                              id=id)

    return pop, log, hof


if __name__ == "__main__":

     
    # first command line argument (if present) is the name
    # of the checkpoint file 
    if len(sys.argv) > 2:
        pop, log, hof = main(sys.argv[1], sys.argv[2])
    else:
        pop, log, hof = main(sys.argv[1])
    
    network = hof[0].createNetwork()
    network.summary()
    print( hof[0] )
    print( hof[0].fitness )

    # learn on the whole set 
    X_train, y_train = load_data("data/CO-nrm-part1.train.csv")
    X_test, y_test = load_data("data/CO-nrm-part1.test.csv")

    E_train, E_test = [], []  
    for _ in range(10):
        network.fit(X_train, y_train,
                    batch_size=100, epochs=500, verbose=0)

        yy_train = network.predict(X_train)
        diff = y_train - yy_train
        E = 100 * sum(diff*diff) / len(yy_train)
        E_train.append(E) 
        
        yy_test = network.predict(X_test)
        diff = y_test - yy_test
        E = 100 * sum(diff * diff) / len(yy_test)
        E_test.append(E) 

    def print_stat(E, name):
        print("E_{:6} avg={:.3f} std={:.3f} min={:.3f} max={:.3f}".format(name,
                                                                          np.mean(E),
                                                                          np.std(E),
                                                                          np.min(E),
                                                                          np.max(E)))
        
    print_stat(E_train, "train")
    print_stat(E_test, "test")

    
