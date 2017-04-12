import random 


def roulette(functions, probs):
    r = random.random()
    for func, prob in zip(functions, probs):
        if r < prob:
            return func
        else:
            r -= prob
    return None
                
