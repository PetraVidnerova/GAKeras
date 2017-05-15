
class CfgSensors:

    batch_size = 100
    epochs = 500
    loss = 'mean_squared_error'

    task_type = 'approximation'

    pop_size = 30
    ngen = 100

    MAX_LAYERS = 5
    MAX_LAYER_SIZE = 100
    MIN_LAYER_SIZE = 5
    DROPOUT = [ 0.0, 0.2, 0.3, 0.4 ] 
    ACTIVATIONS = [ 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear' ] 


class CfgMnist:

#    input_shape = (8,)
#    noutputs = 1
    
    batch_size = 128
    epochs = 20 
    loss = 'categorical_crossentropy'
    #loss = 'mean_squared_error'
    
    task_type = "classification"

    pop_size = 10
    ngen = 10 

    MAX_LAYERS = 3
    MAX_LAYER_SIZE = 20
    MIN_LAYER_SIZE = 10 
    DROPOUT = [ 0.0, 0.2, 0.3, 0.4 ] 
    ACTIVATIONS = [ 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear' ] 

    

    
Config = CfgMnist()

