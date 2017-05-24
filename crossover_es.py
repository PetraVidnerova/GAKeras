import random 

class CrossoverES:

    def cxOnePoint(self, ind1, ind2):
        #do one-point crossover on list of layers 
        cxpoint1 = random.randint(0, len(ind1.layers) - 1)
        cxpoint2 = random.randint(0, len(ind2.layers) - 1)

        ind1.layers[cxpoint1:], ind2.layers[cxpoint2:] = ind2.layers[cxpoint2:], ind1.layers[cxpoint1:]

        s1, s2  = ind1.strategy, ind2.strategy
        s1.blocks[cxpoint1:], s2.blocks[cxpoint2:] = s2.blocks[cxpoint2:], s1.blocks[cxpoint1:] 
        
        assert len(ind1.layers) > 0
        assert len(ind2.layers) > 0 
        assert len(ind1.layers) == len(ind1.strategy.blocks)
        assert len(ind2.layers) == len(ind2.strategy.blocks)
        
        return ind1, ind2 
