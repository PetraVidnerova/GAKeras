import unittest
from convindividual import ConvIndividual
from mutation import MutationConv
from crossover import CrossoverConv
from fitness import Fitness
from config import Config

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D


class IndividualTest(unittest.TestCase):

    def xtest_net(self):

        input_shape = (28,28,1)

        model = Sequential()
        model.add(MaxPooling2D(pool_size=(3,3), input_shape = input_shape))
        print("----->", model.layers[-1].output_shape)
        model.add(MaxPooling2D(pool_size=(3,3)))
        print("----->", model.layers[-1].output_shape)
        model.add(MaxPooling2D(pool_size=(3,3)))
        print("----->", model.layers[-1].output_shape)

        if model.layers[-1].output_shape[1] >= 2 and model.layers[-1].output_shape[2] >= 2:
            model.add(MaxPooling2D(pool_size=(2,2)))
            print("----->", model.layers[-1].output_shape)
        model.add(Flatten())
        
        #model.add(Convolution2D(20, 5, 5, border_mode='same'))
        #model.add(MaxPooling2D(pool_size=(2,2)))
        #model.add(MaxPooling2D(pool_size=(2,2)))
        #model.add(MaxPooling2D(pool_size=(2,2)))
        #model.add(Flatten())

        model.summary()

        
    
    def test_create_individual(self):
        Config.input_shape = (28,28,1)
        Config.noutputs = 10

        for i in range(1000):
            print("--------------------- start {} -------------------".format(i))
            ind = ConvIndividual()
            ind.randomInit()
            print(ind)
            net = ind.createNetwork()
            net.summary()

    def xtest_evaluate(self):
        ind = ConvIndividual()
        ind.randomInit()
        print(ind)
        
        fit = Fitness("data/mnist2d.train")
        print("evaluating")
        print( fit.evaluate(ind) )


    def xtest_mutation(self):
        print(" *** test mutation *** ")
        Config.input_shape = (28,28,1)
        Config.noutputs = 10
        ind = ConvIndividual()
        ind.randomInit()
        print(ind)

        mut = MutationConv()
        mut.mutate(ind)
        print(ind)

    def xtest_crossover(self):
        print(" *** test crossover *** ")
        Config.input_shape = (28,28,1)
        Config.noutputs = 10
        ind1 = ConvIndividual()
        ind1.randomInit()
        print(ind1)
        ind2 = ConvIndividual()
        ind2.randomInit()
        print(ind2)

        cross = CrossoverConv()
        off1, off2 = cross.cxOnePoint(ind1, ind2)
        print(off1)
        print(off2)
        
if __name__ == "__main__":
    unittest.main()
