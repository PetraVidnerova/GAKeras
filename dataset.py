import pandas as pd
from keras.datasets import mnist, cifar10, fashion_mnist
from keras.utils import np_utils


def load_data(filename):

    # load mnist data set
    if filename == "data/mnist.train":
        
        (X_train, y_train), (X_test, y_test) = mnist.load_data()

        X_train = X_train.reshape(60000, 784)
        X_train = X_train.astype('float32')
        X_train /= 255

        Y_train = np_utils.to_categorical(y_train, 10)

        return X_train, Y_train

    if filename == "data/mnist2d.train":
        (X_train, y_train), (X_test, y_test) = mnist.load_data()

        X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
        X_train = X_train.astype('float32')
        X_train /= 255

        Y_train = np_utils.to_categorical(y_train, 10)

        return X_train, Y_train
        
    
    if filename == "data/mnist.test":
        (X_train, y_train), (X_test, y_test) = mnist.load_data()

        X_test = X_test.reshape(10000, 784)
        X_test = X_test.astype('float32')
        X_test /= 255

        Y_test = np_utils.to_categorical(y_test, 10)

        return X_test, Y_test 
    
    if filename == "data/mnist2d.test":
        (X_train, y_train), (X_test, y_test) = mnist.load_data()

        X_test = X_test.reshape(10000, 28, 28, 1)
        X_test = X_test.astype('float32')
        X_test /= 255

        Y_test = np_utils.to_categorical(y_test, 10)

        return X_test, Y_test 
    
    # load cifar data set
    if filename == "data/cifar10.train":
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
        
        X_train = X_train.astype('float32')
        X_train /= 255

        Y_train = np_utils.to_categorical(y_train, 10)

        return X_train, Y_train 

    if filename == "data/cifar10.test":
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
        
        X_test = X_test.astype('float32')
        X_test /= 255

        Y_test = np_utils.to_categorical(y_test, 10)

        return X_test, Y_test 
       
    # load fashion-mnist set
    if filename == "data/fashion_mnist.train":
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

        X_train = X_train.reshape(-1, 28, 28, 1)
        X_train = X_train.astype('float32')
        X_train /= 255

        Y_train = np_utils.to_categorical(y_train, 10)

        return X_train, Y_train
    

    if filename == "data/fashion_mnist.test":
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

        X_test = X_test.reshape(-1, 28, 28, 1)
        X_test = X_test.astype('float32')
        X_test /= 255

        Y_test = np_utils.to_categorical(y_test, 10)

        return X_test, Y_test 

    
    
    # load sensor data from csv file 
    data = pd.read_csv(filename, sep=';')
    X  = data[data.columns[:-1]].as_matrix()
    y = data[data.columns[-1]].as_matrix()
    y = y.reshape(y.shape[0], 1)
    return X, y 

