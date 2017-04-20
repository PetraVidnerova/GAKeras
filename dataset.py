import pandas as pd


def load_data(filename):
    data = pd.read_csv(filename, sep=';')
    X  = data[data.columns[:-1]].as_matrix()
    y = data[data.columns[-1]].as_matrix()
    y = y.reshape(y.shape[0], 1)
    return X, y 

