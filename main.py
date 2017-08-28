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
from config import Config, load_config
from utils import error

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--trainset', help='filename of training set')
parser.add_argument('--testset', help='filename of test set')
parser.add_argument('--id', help='computation id')
parser.add_argument('--checkpoint', help='checkpoint file to load the initial state from')
parser.add_argument('--config', help='json config filename')

args = parser.parse_args()
trainset_name = args.trainset
testset_name = args.testset 
id = args.id
if id is None:
    id = "" 
checkpoint_file = args.checkpoint
config_name = args.config
if config_name is not None:
    load_config(config_name)

# for classification fitness is accuracy, for approximation fitness is error
if Config.task_type == "classification":
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
else:
    creator.create("FitnessMax", base.Fitness, weights=(-1.0,))
creator.create("Individual", Individual, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Structure initializers
toolbox.register("individual", initIndividual, creator.Individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# use multiple processors 
pool = multiprocessing.Pool(5)
toolbox.register("map", pool.map)

# register operators 
fit = Fitness("data/"+trainset_name)
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
        pop = toolbox.population(n=Config.pop_size)
        start_gen = 0
        hof = tools.HallOfFame(1)
        logbook = tools.Logbook()
    
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    pop, log = alg.myEASimple(pop, start_gen, toolbox, cxpb=0.6, mutpb=0.2, ngen=Config.ngen, 
                              stats=stats, halloffame=hof, logbook=logbook, verbose=True,
                              id=id)

    return pop, log, hof


if __name__ == "__main__":

    # load the whole data 
    X_train, y_train = load_data("data/"+trainset_name)
    X_test, y_test = load_data("data/"+testset_name)
    
    # set cfg
    Config.input_shape = X_train[0].shape 
    Config.noutputs = y_train.shape[1]
    #    print(Config.input_shape, Config.noutputs)
    
    if checkpoint_file is None:
        pop, log, hof = main(id)
    else:
        pop, log, hof = main(id, checkpoint_file)
    
    network = hof[0].createNetwork()
    network.summary()
    print( hof[0] )
    print( hof[0].fitness )


    # learn on the whole set
    #    
    E_train, E_test = [], []  
    for _ in range(10):
        network = hof[0].createNetwork() 
        network.fit(X_train, y_train,
                    batch_size=Config.batch_size, nb_epoch=Config.epochs, verbose=0)

        yy_train = network.predict(X_train)
        E_train.append(error(yy_train, y_train)) 
        
        yy_test = network.predict(X_test)
        E_test.append(error(yy_test, y_test))

        
    def print_stat(E, name):
        print("E_{:6} avg={:.4f} std={:.4f}  min={:.4f} max={:.4f}".format(name,
                                                                           np.mean(E),
                                                                           np.std(E),
                                                                           np.min(E),
                                                                           np.max(E)))
        
    print_stat(E_train, "train")
    print_stat(E_test, "test")

    
