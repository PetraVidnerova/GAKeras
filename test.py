import unittest
from convindividual import ConvIndividual
from mutation_conv import MutationConv
from crossover_conv import CrossoverConv
from fitness import Fitness
from config import Config

class IndividualTest(unittest.TestCase):

    def xtest_create_individual(self):
        Config.input_shape = (28,28,1)
        Config.noutputs = 10
         
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

    def test_crossover(self):
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
