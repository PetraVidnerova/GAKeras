import configparser

config = configparser.ConfigParser()

config['DEFAULT'] = { 'batch_size': 128,
                      'epochs': 20,
                      'loss': 'mean_squared_error',
                      'task_type': 'classification' }

config['ES'] = { 'mu': 5,
                 'lambda': 10,
                 'ngen': 20 }

config['STRATEGY_COEFS'] = { 'size_min_strategy': 5,
                             'size_max_strategy': 50,
                             'dropout_min_strategy': 0.05,
                             'dropout_max_strategy': 0.2 }

config['NETWORK'] = {'max_layers': 5,
                     'max_layer_size': 1000,
                     'min_layer_size': 10,
                     'dropout': [ 0.0, 0.2, 0.3, 0.4 ],
                     'activations': [ 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear' ] }

with open('example.ini', 'w') as configfile:
    config.write(configfile)
