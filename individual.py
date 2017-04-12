import random
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop

INPUT_SHAPE = (8,)
NOUTPUTS = 1 

MAX_LAYERS = 3
MAX_LAYER_SIZE = 20
MIN_LAYER_SIZE = 10 
DROPOUT = [ 0.0, 0.2, 0.3, 0.4 ] 
ACTIVATIONS = [ 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear' ] 

class Layer:
    """ Specification of one layer.
        Includes size, regularization, activaton. 
    """
    def __init__(self):
        pass

    def randomInit(self):
        self.size = random.randint(MIN_LAYER_SIZE, MAX_LAYER_SIZE)
        self.dropout = random.choice(DROPOUT) 
        self.activation = random.choice(ACTIVATIONS)  
        return self 


class Individual:
    """  Individual coding network architecture. 
    """
    
    def __init__(self):
        self.input_shape = INPUT_SHAPE
        self.noutputs = NOUTPUTS
    
    def randomInit(self):
        self.layers = []
        num_layers = random.randint(1, MAX_LAYERS)
        for l in range(num_layers):
            layer = Layer().randomInit() 
            self.layers.append(layer)

    def createNetwork(self):

        model = Sequential()

        firstlayer = True
        for l in self.layers:
            if firstlayer:
                model.add(Dense(l.size, input_shape=self.input_shape))
                firstlayer = False
            else:
                model.add(Dense(l.size))
            model.add(Activation(l.activation))
            if l.dropout > 0:
                model.add(Dropout(l.dropout))
        model.add(Dense(self.noutputs)) 
            
        model.compile(loss='mean_squared_error',
                      optimizer=RMSprop())
        
        return model 


def initIndividual(indclass):
    ind = indclass()
    ind.randomInit()
    return ind
