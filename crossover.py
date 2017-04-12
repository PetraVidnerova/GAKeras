import random 

class Crossover:

    def cxOnePoint(self, ind1, ind2):
        #do one-point crossover on list of layers 
        cxpoint1 = random.randint(0, len(ind1.layers) - 1)
        cxpoint2 = random.randint(0, len(ind2.layers) - 1)
        ind1.layers[cxpoint1:], ind2.layers[cxpoint2:] = ind2.layers[cxpoint2:], ind1.layers[cxpoint1:]

        assert len(ind1.layers) > 0
        assert len(ind2.layers) > 0 
        
        return ind1, ind2 
