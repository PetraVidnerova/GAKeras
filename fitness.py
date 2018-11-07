import random 
import numpy as np 
import pickle
import sklearn
from sklearn.model_selection import KFold
from dataset import load_data
from config import Config
from utils import error
from keras import backend as K 


class Database:

    def __init__(self):
        self.data = []
    
    def insert(self, individual, fitness):
        self.data.append((individual, fitness))
        
    def save(self, name):
        with open(name, "wb") as f:
            pickle.dump(self.data, f)

class Fitness:

    def __init__(self, train_name):
        
        # load train data 
        self.X, self.y = load_data(train_name)
                
    def evaluate(self, individual):
        #print(" *** evaluate *** ")

        #model = individual.createNetwork()
        #return random.random(), 
         
        random.seed(42) 
        # perform KFold crossvalidation 
        kf = KFold(n_splits=5)
        scores = []
        for train, test in kf.split(self.X):   # train, test are indicies 
            X_train, X_test = self.X[train], self.X[test]
            y_train, y_test = self.y[train], self.y[test]
                
            model = individual.createNetwork()
            model.fit(X_train, y_train,
                      batch_size=Config.batch_size, epochs=Config.epochs, verbose=0)
            
            yy_test = model.predict(X_test)
            scores.append(error(y_test, yy_test))
            
        fitness = np.mean(scores)
            
        # I try this to prevent memory leaks in nsga2-keras 
        K.clear_session()

        return fitness,
