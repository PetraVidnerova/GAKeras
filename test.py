import unittest
from individual import Individual 
from fitness import Fitness

class IndividualTest(unittest.TestCase):

    def test_create_individual(self):
        ind = Individual()
        ind.randomInit()
        net = ind.createNetwork()
        net.summary()
        
    def test_evaluate(self):
        ind = Individual()
        ind.randomInit()

        fit = Fitness("data/CO-nrm-part1.train.csv")
        print( fit.evaluate(ind) )

        
if __name__ == "__main__":
    unittest.main()
