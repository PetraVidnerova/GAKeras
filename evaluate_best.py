import pickle

from multiprocessing import Pool

from deap import base 
from deap import creator 

from convindividual import ConvIndividual
from dataset import load_data
from config import Config
from utils import error

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--trainset', help='filename of training set')
parser.add_argument('--testset', help='filename of test set')
parser.add_argument('--checkpoint', help='checkpoint file to load the initial state from')

args = parser.parse_args()
trainset_name = args.trainset
testset_name = args.testset
checkpoint_file = args.checkpoint

if Config.task_type == "classification":
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
else:
    creator.create("FitnessMax", base.Fitness, weights=(-1.0,))
creator.create("Individual", ConvIndividual, fitness=creator.FitnessMax)


def evaluate(ind, X_train, y_train, X_test, y_test):
    network = ind.createNetwork()
    
    E_train, E_test = 0, 0
    network.fit(X_train, y_train,
                batch_size=Config.batch_size, nb_epoch=1, verbose=0)

    yy_train = network.predict(X_train)
    E_train = error(yy_train, y_train)
    
    yy_test = network.predict(X_test)
    E_test = error(yy_test, y_test)

    return E_train, E_test 
    


if __name__ == "__main__":
    # load the data sets
    X_train, y_train = load_data("data/"+trainset_name)
    X_test, y_test = load_data("data/"+testset_name)

    # load checkpoint 
    cp = pickle.load(open(checkpoint_file, "rb"))
    best_ind = cp["halloffame"][0]

    def myeval(x):
        return evaluate(best_ind, X_train, y_train, X_test, y_test)

    N = 10
    p = Pool(N)
    errors = list(p.map(myeval, range(N)))

    print(errors)
        
    E_train = sum([x[0] for x in errors]) / N
    E_test = sum([x[1] for x in errors]) / N

    print("E_train: {}".format(E_train))
    print("E_test: {}".format(E_test))
