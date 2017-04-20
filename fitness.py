import random 
import numpy as np 
import sklearn
from sklearn.cross_validation import KFold
from dataset import load_data

class Fitness:

    def __init__(self, train_name):
        
        # load train data 
        self.X, self.y = load_data(train_name)

    def evaluate(self, individual):

        random.seed(42) 
        # perform KFold crossvalidation 
        kf = KFold(len(self.X), n_folds=5)
        scores = []
        for train, test in kf:   # train, test are indicies 
            X_train, X_test = self.X[train], self.X[test]
            y_train, y_test = self.y[train], self.y[test]
                
            model = individual.createNetwork()
            model.fit(X_train, y_train,
                      batch_size=100, epochs=500, verbose=0)
            
            yy_test = model.predict(X_test)
            diff = y_test - yy_test
            train_error = 100 * sum(diff * diff) / len(yy_test)

            scores.append(train_error[0])
            
        return np.mean(scores),
