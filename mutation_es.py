import random
import math
from utils import roulette
from individual import Individual, Layer
from strategy import Block
from config import Config

PROB_MUTATE_LAYER = 0.5
PROB_ADD_LAYER = 0.25
PROB_DEL_LAYER = 0.25

PROB_CHANGE_LAYER_SIZE = 0.4
PROB_CHANGE_DROPOUT = 0.3
PROB_CHANGE_ACTIVATION = 0.3

class MutationES:

    def __init__(self):
        pass


    def mutateSize(self, layer, strategy, size):
        # strategy coeficients 
        t = 1.0 / math.sqrt(2. * math.sqrt(size))
        t0 = 1.0 / math.sqrt(2. * size)
        n = random.gauss(0, 1)
        t0_n = t0 * n
        # modify strategy
        strategy.size *= math.exp(t0_n + t * random.gauss(0, 1))
        # modify size 
        layer.size += round(strategy.size * random.gauss(0, 1))
        layer.size = int(layer.size)
        if layer.size < Config.MIN_LAYER_SIZE:
            layer.size = Config.MIN_LAYER_SIZE
        if layer.size > Config.MAX_LAYER_SIZE:
            layer.size = Config.MAX_LAYER_SIZE

    def mutateDropout(self, layer, strategy, size):
        # strategy coeficients 
        t = 1.0 / math.sqrt(2. * math.sqrt(size))
        t0 = 1.0 / math.sqrt(2. * size)
        n = random.gauss(0, 1)
        t0_n = t0 * n
        # modify strategy
        strategy.dropout *= math.exp(t0_n + t * random.gauss(0, 1))
        # modify dropout 
        layer.dropout += strategy.dropout * random.gauss(0, 1)
        if layer.dropout < 0.0:
            layer.dropout = 0
        if layer.dropout > 0.9:
            layer.dropout = 0.9 
        
    def mutateActivation(self, layer, strategy, size):
        layer.activation = random.choice(Config.ACTIVATIONS)


    def mutateLayer(self, individual):

        # select layer random 
        l = random.randint(0, len(individual.layers)-1)
        layer = individual.layers[l]
        strategy = individual.strategy.blocks[l]
        
        mutfunc = roulette([self.mutateSize, self.mutateDropout, self.mutateActivation],
                           [PROB_CHANGE_LAYER_SIZE, PROB_CHANGE_DROPOUT, PROB_CHANGE_ACTIVATION])

        if mutfunc:
            mutfunc(layer, strategy, len(individual.layers))
        
        return individual,

    def addLayer(self, individual):
        l = random.randint(0, len(individual.layers)-1)
        individual.layers.insert(l, Layer().randomInit())
        individual.strategy.blocks.insert(l, Block().randomInit())
        return individual, 

    def delLayer(self, individual):
        if len(individual.layers)>1:
            l = random.randint(0, len(individual.layers)-1)
            del individual.layers[l]
            del individual.strategy.blocks[l]
        return individual, 
    
    def mutate(self, individual): 

        mutfunc = roulette([self.mutateLayer, self.addLayer, self.delLayer],
                           [PROB_MUTATE_LAYER, PROB_ADD_LAYER, PROB_DEL_LAYER])
        if mutfunc:
            return mutfunc(individual)

        assert len(individual.layers) == len(individual.strategy.blocks)
        
        return individual, 
