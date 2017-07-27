import random 

class CrossoverConv:

    def cxOnePoint(self, ind1, ind2):
        #do one-point crossover on list of layers
        #convolutional layers 
        cxpoint1 = random.randint(0, len(ind1.conv_layers) - 1)
        cxpoint2 = random.randint(0, len(ind2.conv_layers) - 1)
        ind1.conv_layers[cxpoint1:], ind2.conv_layers[cxpoint2:] = ind2.conv_layers[cxpoint2:], ind1.conv_layers[cxpoint1:]

        assert len(ind1.conv_layers) > 0
        assert len(ind2.conv_layers) > 0

        #dense layers 
        cxpoint1 = random.randint(0, len(ind1.dense_layers) - 1)
        cxpoint2 = random.randint(0, len(ind2.dense_layers) - 1)
        ind1.dense_layers[cxpoint1:], ind2.dense_layers[cxpoint2:] = ind2.dense_layers[cxpoint2:], ind1.dense_layers[cxpoint1:]

        assert len(ind1.dense_layers) > 0
        assert len(ind2.dense_layers) > 0
        
        return ind1, ind2 
