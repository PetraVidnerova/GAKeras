import numpy as np 
import pandas as pd 
import sklearn
from sklearn.cross_validation import KFold

class Fitness:

    def __init__(self, train_name):
        
        # load train data 
        data = pd.read_csv(train_name,sep=';')
        X_train = data[data.columns[:-1]]
        y_train = data[data.columns[-1]]
        self.X = X_train.as_matrix()
        self.y = y_train.as_matrix()
        self.y = self.y.reshape(self.y.shape[0], 1)


    def evaluate(self, individual):

        model = individual.createNetwork()

        # perform KFold crossvalidation 
        kf = KFold(len(self.X), n_folds=5)
        scores = []
        for train, test in kf:   # train, test are indicies 
            X_train, X_test = self.X[train], self.X[test]
            y_train, y_test = self.y[train], self.y[test]
                
            model.fit(X_train, y_train,
                      batch_size=100, epochs=500, verbose=0)
            
            yy_test = model.predict(X_test)
            diff = y_test - yy_test
            train_error = 100 * sum(diff * diff) / len(yy_test)

            scores.append(train_error[0])
            
        return np.mean(scores),
