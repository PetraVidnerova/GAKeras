import random 
import numpy as np 
import pickle
import sklearn
from sklearn.cross_validation import KFold
from dataset import load_data
from config import Config
from utils import error

class Database:

    def __init__(self):
        self.data = []
    
    def insert(self, individual, fitness):
        self.data.append((individual, fitness))
        print("individual inserted")
        
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
        kf = KFold(len(self.X), n_folds=3)
        scores = []
        for train, test in kf:   # train, test are indicies 
            X_train, X_test = self.X[train], self.X[test]
            y_train, y_test = self.y[train], self.y[test]
                
            model = individual.createNetwork()
            model.fit(X_train, y_train,
                      batch_size=Config.batch_size, nb_epoch=Config.epochs, verbose=0)
            
            yy_test = model.predict(X_test)
            scores.append(error(y_test, yy_test))
            
        fitness = np.mean(scores)
            
        return fitness,
