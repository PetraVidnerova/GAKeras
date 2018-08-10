import pickle

from multiprocessing import Pool

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import RMSprop


from dataset import load_data
from config import Config
from utils import error




def createNetwork(X_train):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=X_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop())

    return model
    

def evaluate(x, X_train, y_train, X_test, y_test):

    from numpy.random import seed
    seed(x)
    
    network = createNetwork(X_train)
    
    E_train, E_test = 0, 0
    network.fit(X_train, y_train,
                batch_size=Config.batch_size, nb_epoch=20, verbose=0)

    yy_train = network.predict(X_train)
    E_train = error(yy_train, y_train)
    
    yy_test = network.predict(X_test)
    E_test = error(yy_test, y_test)

    return E_train, E_test 
    


if __name__ == "__main__":
    # load the data sets
    X_train, y_train = load_data("data/cifar10.train")
    X_test, y_test = load_data("data/cifar10.test")


    def myeval(x):
        return evaluate(x, X_train, y_train, X_test, y_test)

    N = 10
    p = Pool(N)
    errors = list(p.map(myeval, range(N)))

    print(errors)
        
    E_train = sum([x[0] for x in errors]) / N
    E_test = sum([x[1] for x in errors]) / N

    print("E_train: {}".format(E_train))
    print("E_test: {}".format(E_test))
