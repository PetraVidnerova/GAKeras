import random
from config import Config


class Block:
    """ Strategy for one layer.
    """ 
    def __init__(self):
        pass

    def randomInit(self):
        self.size = random.uniform(Config.SIZE_MIN_STRATEGY, Config.SIZE_MAX_STRATEGY)
        self.dropout = random.uniform(Config.DROPOUT_MIN_STRATEGY, Config.DROPOUT_MAX_STRATEGY)
        return self


class Strategy:

    def randomInit(self, size):
        self.blocks = [ Block().randomInit() for _ in range(size) ]

        
def initIndStrategy(indclass, strategyclass):
    ind = indclass()
    ind.randomInit()
    ind.strategy = strategyclass()
    ind.strategy.randomInit(len(ind.layers))
    return ind 
