import random 

def cxListOnePoint(list1, list2):
    assert len(list1) > 0 and len(list2) > 0
    
    cxpoint1 = random.randint(0, len(list1) - 1)
    cxpoint2 = random.randint(0, len(list2) - 1)
    list1[cxpoint1:], list2[cxpoint2:] = list2[cxpoint2:], list1[cxpoint1:]
    
    assert len(list1) > 0 and len(list2) > 0 
    

class Crossover:

    def cxOnePoint(self, ind1, ind2):
        #do one-point crossover on list of layers 
        cxListOnePoint(ind1.layers, ind2.layers) 
        return ind1, ind2 

    
class CrossoverConv:

    def cxOnePoint(self, ind1, ind2):
        #do one-point crossover on list of layers
        #convolutional layers
        cxListOnePoint(ind1.conv_layers, ind2.conv_layers)
        #dense layers 
        cxListOnePoint(ind1.dense_layers, ind2.dense_layers)
        
        return ind1, ind2 
