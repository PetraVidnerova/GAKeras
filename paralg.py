import random
import pickle
import time
import datetime 
from operator import attrgetter 

from deap import algorithms 

from pool import Pool 
from fitness import Database

PROCESSORS = 10

def myAsyncEA(population, start_gen, toolbox, cxpb, mutpb, ngen,
              stats, halloffame, logbook, verbose, id=None):

    """ Asynchronous EA """ 

    def eval_individual(ind):
        ind.fitness.values = toolbox.evaluate(ind)
        return ind

    def gen_new_offspring():
        valid_offspring = True
        offspring = None
        while valid_offspring:
            # select two individuals and clone them 
            offspring = map(toolbox.clone, toolbox.select(population, 2))
            # apply crossover and mutation, take the first offspring
            offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)[0]
            valid_offspring = offspring.fitness.valid
            # if the offspring is valid it is already in the population
        return offspring 

    def log_stats():
        record = stats.compile(population)
        logbook.record(gen=start_gen + (num_evals // popsize),  **record)
        if verbose:
            print(logbook.stream)

    def save_checkpoint():
        cp = dict(population=population, generation=start_gen + (num_evals // popsize), halloffame=halloffame,
                  logbook=logbook, rndstate=random.getstate())
        if id is None:
            cp_name = "checkpoint_ea.pkl"
        else:
            cp_name = "checkpoint_ea_{}.pkl".format(id)
        pickle.dump(cp, open(cp_name, "wb"))
            
    
    popsize = len(population)
    MIN_POP_SIZE = popsize // 2 
    total_time = datetime.timedelta(seconds=0)

    pool = Pool(processors=PROCESSORS, evalfunc=eval_individual)    

    db = Database()
    
    if all([ind.fitness.valid for ind in population]):
        print("All individuals have valid fitness")
        # all inds are valid (loaded from checkpoint), generate some new to fill the pool 
        for _ in range(PROCESSORS):
            offspring = gen_new_offspring() 
            pool.putQuerry(offspring)
    else:
        # put all individuals to the pool 
        for ind in population:
            pool.putQuerry(ind)
        population = [] # we get them back later 

    num_evals = 0

    start_time = datetime.datetime.now()
    while num_evals < (ngen-start_gen)*popsize:

        # get finished individual and add him to population 
        ind = pool.getAnswer()
        assert ind.fitness.valid
        # add fitness and individual to the database
        db.insert(ind, ind.fitness.values[0])
        population.append(ind)
        num_evals += 1
        
        # update halloffame
        halloffame.update([ind])

        if len(population) < MIN_POP_SIZE:
            continue

        if num_evals % popsize == 0:
            # record statistics and save checkpoint
            log_stats()            
            save_checkpoint()
            
        # check time    
        eval_time = datetime.datetime.now() - start_time
        total_time = total_time + eval_time
        #print("Time ", total_time)
        if total_time > datetime.timedelta(hours=8*24):
            print("Time limit exceeded.")
            break
        start_time = datetime.datetime.now() 

        
        if len(population) > popsize:
            # delete the worst one
            population = list(sorted(population, key=attrgetter('fitness')))[1:]

        # generate new offspring and put him to the pool 
        offspring = gen_new_offspring() 
        pool.putQuerry(offspring)


    pool.close()
    #    print(db.data)
    db.save("database." + id + ".pkl")
    return population, logbook




 
