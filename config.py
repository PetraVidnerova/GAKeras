

class CfgSensors:

    batch_size = 100
    epochs = 10 #500
    loss = 'mean_squared_error'

    task_type = 'approximation'

    pop_size = 10
    ngen = 30

    MAX_LAYERS = 5
    MAX_LAYER_SIZE = 100
    MIN_LAYER_SIZE = 5
    DROPOUT = [ 0.0, 0.2, 0.3, 0.4 ] 
    ACTIVATIONS = [ 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear' ] 


class CfgSensorsES:

    batch_size = 100
    epochs = 500
    loss = 'mean_squared_error'

    task_type = 'approximation'

    MU = 10
    LAMBDA = 30
    ngen = 100

    MAX_LAYERS = 5
    MAX_LAYER_SIZE = 100
    MIN_LAYER_SIZE = 5
    DROPOUT = [ 0.0, 0.2, 0.3, 0.4 ] 
    ACTIVATIONS = [ 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear' ] 

    SIZE_MIN_STRATEGY = 5
    SIZE_MAX_STRATEGY = 50
    DROPOUT_MIN_STRATEGY = 0.05
    DROPOUT_MAX_STRATEGY = 0.2

    
    
class CfgMnist:

    batch_size = 128
    epochs = 1
    loss = 'categorical_crossentropy'
    #loss = 'mean_squared_error'
    
    task_type = "classification"

    pop_size = 10
    ngen = 300

    MAX_LAYERS = 5
    MAX_LAYER_SIZE = 300
    MIN_LAYER_SIZE = 10 
    DROPOUT = [ 0.0, 0.2, 0.3, 0.4 ] 
    ACTIVATIONS = [ 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear' ] 

    # for convolutional networks
    MIN_FILTERS = 10
    MAX_FILTERS = 50 
    MIN_KERNEL_SIZE = 2
    MAX_KERNEL_SIZE = 5
    MIN_POOL_SIZE = 2
    MAX_POOL_SIZE = 3
    MAX_CONV_LAYERS = 3
    MAX_DENSE_LAYERS = 3
    
    #DENSE_LAYER = 0.5
    CONV_LAYER = 0.7 
    MAX_POOL_LAYER = 0.3 

    
    
class CfgMnistES:

    batch_size = 128
    epochs = 20 
    #loss = 'categorical_crossentropy'
    loss = 'mean_squared_error'
    
    task_type = "classification"

    MU = 5
    LAMBDA = 10
    ngen = 10

    SIZE_MIN_STRATEGY = 5
    SIZE_MAX_STRATEGY = 50
    DROPOUT_MIN_STRATEGY = 0.05
    DROPOUT_MAX_STRATEGY = 0.2
    
    MAX_LAYERS = 5
    MAX_LAYER_SIZE = 1000
    MIN_LAYER_SIZE = 10 
    DROPOUT = [ 0.0, 0.2, 0.3, 0.4 ] 
    ACTIVATIONS = [ 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear' ] 

#Config = CfgSensors()    
Config = CfgMnist()
#Config = CfgSensorsES()  
#Config = CfgMnist()

import json

def load_config(name):

    attrlist = [] 
    with open(name,"r") as f:
        attrlist = json.load(f)

    for attr, value in attrlist:
        setattr(Config, attr, value)

        
