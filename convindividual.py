import random
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import RMSprop
from utils import roulette 
from individual import Layer
from config import Config

class ConvLayer:
    """ Specification of one convolutional layer.
        Includes number of filters, kernel size, activation.
    """
    def __init__(self):
        pass

    def randomInit(self):
        self.filters = random.randint(Config.MIN_FILTERS, Config.MAX_FILTERS)
        # filters are squares kernel_size x kernel_size 
        self.kernel_size = random.randint(Config.MIN_KERNEL_SIZE, Config.MAX_KERNEL_SIZE)
        self.activation = random.choice(Config.ACTIVATIONS)
        return self
    
    def __str__(self):
        return "conv #{} kernelsize={} activation={}".format(self.filters, self.kernel_size, self.activation)
        

class MaxPoolLayer:
    """ Specification of one max pooling layer.
    """
    def __init__(self):
        pass
    
    def randomInit(self):
        # pooling size is (pool_size, pool_size)
        self.pool_size =  random.randint(Config.MIN_POOL_SIZE, Config.MAX_POOL_SIZE)
        return self

    def __str__(self):
        return "pool poolsize={} ".format(self.pool_size)


def createRandomLayer():

    create = roulette([lambda: ConvLayer(),
                       lambda: MaxPoolLayer()],
                      [Config.CONV_LAYER, Config.MAX_POOL_LAYER])
    if create:
        return create()
    else:
        return ConvLayer() 
    
class ConvIndividual:
    """ Individual coding convolutional network architecture.
        Individual consists of two parts, first of convolutioanal
        and max pooling layers, the second part of dense layers.
    """

    def __init__(self):
        self.input_shape = Config.input_shape
        self.noutputs = Config.noutputs
        self.nparams = None
        
    def randomInit(self):
        self.conv_layers = []
        num_conv_layers = random.randint(1, Config.MAX_CONV_LAYERS)
        for l in range(num_conv_layers):
            layer = createRandomLayer().randomInit()
            self.conv_layers.append(layer)
            
        self.dense_layers = []
        num_dense_layers = random.randint(1, Config.MAX_DENSE_LAYERS)
        for l in range(num_dense_layers):
            layer = Layer().randomInit()
            self.dense_layers.append(layer)


    def createNetwork(self):

        model = Sequential()


        firstlayer = True

        # convolutional part 
        for l in self.conv_layers:
            if type(l) is ConvLayer:
                if firstlayer:
                    model.add(Conv2D(l.filters, (l.kernel_size, l.kernel_size),
                                     padding='same', # let not the shape vanish
                                     input_shape=self.input_shape))
                    firstlayer = False
                else:
                    model.add(Conv2D(l.filters, (l.kernel_size, l.kernel_size), padding='same'))
                model.add(Activation(l.activation))
                
            elif type(l) is MaxPoolLayer:
                if firstlayer:
                    model.add(MaxPooling2D(pool_size=(l.pool_size,l.pool_size),
                                           input_shape=self.input_shape))
                    firstlayer = False
                else:
                    # check if pooling is possible
                    if model.layers[-1].output_shape[1] >= l.pool_size and model.layers[-1].output_shape[2] >= l.pool_size:
                        model.add(MaxPooling2D(pool_size=(l.pool_size, l.pool_size)))
                    
            else:
                raise TypeError("unknown type of layer") 
            
        # dense part
        model.add(Flatten())
        for l in self.dense_layers:
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

        self.nparams = model.count_params()
                
        return model 

    
    def __str__(self):

        ret = "------------------------\n"
        for l in self.conv_layers+self.dense_layers:
            ret += str(l)
            ret += "\n" 
            ret += "------------------------\n"
        return ret 
