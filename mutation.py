import random
from utils import roulette
from individual import Individual, Layer
from config import Config

PROB_MUTATE_LAYER = 0.5
PROB_ADD_LAYER = 0.25
PROB_DEL_LAYER = 0.25

PROB_CHANGE_LAYER_SIZE = 0.4
PROB_CHANGE_DROPOUT = 0.25
PROB_CHANGE_ACTIVATION = 0.25
PROB_CHANGE_ALL = 0.1 

PROB_RANDOM_LAYER_SIZE = 0.4
PROB_ADD_NEURON = 0.3
PROB_DEL_NEURON = 0.3 

class Mutation:

    def __init__(self):
        pass


    def changeSize(self, layer):
        layer.size = random.randint(Config.MIN_LAYER_SIZE, Config.MAX_LAYER_SIZE)

    def addNeuron(self, layer):
        layer.size += 1

    def delNeuron(self, layer):
        if layer.size > 0:
            layer.size -= 1 
    
    def mutateSize(self, layer):

        mutfunc = roulette([self.changeSize, self.addNeuron, self.delNeuron],
                           [PROB_RANDOM_LAYER_SIZE, PROB_ADD_NEURON, PROB_DEL_NEURON])
        if mutfunc:
            mutfunc(layer)


    def mutateDropout(self, layer):
        layer.dropout = random.choice(Config.DROPOUT)
        
    def mutateActivation(self, layer):
        layer.activation = random.choice(Config.ACTIVATIONS)

    def randomInit(self, layer):
        layer.randomInit()

    def mutateLayer(self, individual):

        # select layer random 
        l = random.randint(0, len(individual.layers)-1)
        layer = individual.layers[l]

        mutfunc = roulette([self.mutateSize, self.mutateDropout, self.mutateActivation,
                            self.randomInit],
                           [PROB_CHANGE_LAYER_SIZE, PROB_CHANGE_DROPOUT, PROB_CHANGE_ACTIVATION,
                            PROB_CHANGE_ALL])

        if mutfunc:
            mutfunc(layer)
        
        return individual,

    def addLayer(self, individual):
        l = random.randint(0, len(individual.layers)-1)
        individual.layers.insert(l, Layer().randomInit())
        return individual, 

    def delLayer(self, individual):
        if len(individual.layers)>1:
            l = random.randint(0, len(individual.layers)-1)
            del individual.layers[l]
        return individual, 
    
    def mutate(self, individual): 
        
        mutfunc = roulette([self.mutateLayer, self.addLayer, self.delLayer],
                           [PROB_MUTATE_LAYER, PROB_ADD_LAYER, PROB_DEL_LAYER])
        if mutfunc:
            return mutfunc(individual)
        
        return individual, 




    
from convindividual import ConvIndividual, Layer, ConvLayer, MaxPoolLayer

PROB_CHANGE_NUM_FILTERS = 0.3
PROB_MUTATE_KERNEL_SIZE = 0.3
PROB_CHANGE_CONV_ACTIVATION = 0.3
PROB_CHANGE_CONV_ALL = 0.1



class MutationConv(Mutation):

    def __init__(self):
        pass


    def mutateFilters(self, layer):
        layer.filters = random.randint(Config.MIN_FILTERS, Config.MAX_FILTERS)

    def mutateKernelSize(self, layer):
        layer.kernel_size = random.randint(Config.MIN_KERNEL_SIZE, Config.MAX_KERNEL_SIZE)

    def mutatePoolSize(self, layer):
        layer.pool_size =  random.randint(Config.MIN_POOL_SIZE, Config.MAX_POOL_SIZE)
        
        
    def mutateLayer(self, individual):

        layers = individual.conv_layers + individual.dense_layers
        # select layer random 
        l = random.randint(0, len(layers)-1)
        layer = layers[l]

        if type(layer) is Layer:
            mutfunc = roulette([self.mutateSize, self.mutateDropout, self.mutateActivation,
                                self.randomInit],
                               [PROB_CHANGE_LAYER_SIZE, PROB_CHANGE_DROPOUT, PROB_CHANGE_ACTIVATION,
                               PROB_CHANGE_ALL])
            if mutfunc:
                mutfunc(layer)
        elif type(layer) is ConvLayer:
            mutfunc = roulette([self.mutateFilters, self.mutateKernelSize, self.mutateActivation,
                                self.randomInit],
                               [PROB_CHANGE_NUM_FILTERS, PROB_MUTATE_KERNEL_SIZE, PROB_CHANGE_CONV_ACTIVATION,
                                PROB_CHANGE_CONV_ALL])
            if mutfunc:
                mutfunc(layer)
        elif type(layer) is MaxPoolLayer:
            self.mutatePoolSize(layer)
        else:
            raise(TypeError("unknown type of layer"))
            
                
        return individual,

    def addLayer(self, individual):
        conv_part = random.randint(0,1)
        if (conv_part):
            l = random.randint(0, len(individual.conv_layers)-1)
            if random.randint(0,1):
                individual.conv_layers.insert(l, ConvLayer().randomInit())
            else:
                individual.conv_layers.insert(l, MaxPoolLayer().randomInit())
        else:
            l = random.randint(0, len(individual.dense_layers)-1)
            individual.dense_layers.insert(l, Layer().randomInit())
            
        return individual, 

    def delLayer(self, individual):
        conv_part = random.randint(0,1)
        if (conv_part):
            if len(individual.conv_layers)>1:
                l = random.randint(0, len(individual.conv_layers)-1)
                del individual.conv_layers[l]
        else:
            if len(individual.dense_layers)>1:
                l = random.randint(0, len(individual.dense_layers)-1)
                del individual.dense_layers[l]
                
        return individual,
    
    
    def mutate(self, individual): 
        
        mutfunc = roulette([self.mutateLayer, self.addLayer, self.delLayer],
                           [PROB_MUTATE_LAYER, PROB_ADD_LAYER, PROB_DEL_LAYER])
        if mutfunc:
            return mutfunc(individual)
        
        return individual, 
