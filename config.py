
class CfgSensors:

    batch_size = 100
    epochs = 500
    loss = 'mean_squared_error'

    task_type = 'approximation'

    pop_size = 10
    ngen = 10

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

    MU = 5
    LAMBDA = 10
    ngen = 10

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
    epochs = 20 
    #loss = 'categorical_crossentropy'
    loss = 'mean_squared_error'
    
    task_type = "classification"

    pop_size = 30
    ngen = 100 

    MAX_LAYERS = 5
    MAX_LAYER_SIZE = 1000
    MIN_LAYER_SIZE = 10 
    DROPOUT = [ 0.0, 0.2, 0.3, 0.4 ] 
    ACTIVATIONS = [ 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear' ] 

    

Config = CfgSensorsES()  
#Config = CfgMnist()

