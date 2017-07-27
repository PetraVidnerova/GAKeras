import random
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop

from config import Config


class Layer:
    """ Specification of one layer.
        Includes size, regularization, activaton. 
    """
    def __init__(self):
        pass

    def randomInit(self):
        self.size = random.randint(Config.MIN_LAYER_SIZE, Config.MAX_LAYER_SIZE)
        self.dropout = random.choice(Config.DROPOUT) 
        self.activation = random.choice(Config.ACTIVATIONS)  
        return self

    def __str__(self):
        return " #{} dropout={} activation={}".format(self.size, self.dropout, self.activation)


class Individual:
    """  Individual coding network architecture. 
    """
    
    def __init__(self):
        self.input_shape = Config.input_shape
        self.noutputs = Config.noutputs 
        #print(self.input_shape, self.noutputs)

        
    def randomInit(self):
        self.layers = []
        num_layers = random.randint(1, Config.MAX_LAYERS)
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

        # final part 
        model.add(Dense(self.noutputs))
        if Config.task_type == "classification":
            model.add(Activation('softmax'))
            
        model.compile(loss=Config.loss,
                      optimizer=RMSprop())
        
        return model 

    def __str__(self):

        ret = "------------------------\n"
        for l in self.layers:
            ret += str(l)
            ret += "\n" 
            ret += "------------------------\n"
        return ret 

def initIndividual(indclass):
    ind = indclass()
    ind.randomInit()
    return ind
