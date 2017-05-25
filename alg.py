import random
import pickle
import time
import datetime 

from deap import algorithms 


def myEASimple(population, start_gen, toolbox, cxpb, mutpb, ngen, 
               stats, halloffame, logbook, verbose, id=None):

    total_time = datetime.timedelta(seconds=0) 
    for gen in range(start_gen, ngen):
        start_time = datetime.datetime.now()
        population = algorithms.varAnd(population, toolbox, cxpb=cxpb, mutpb=mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        halloffame.update(population)
        record = stats.compile(population)
        logbook.record(gen=gen, evals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)


        population = toolbox.select(population, k=len(population))

        if gen % 1 == 0:
            # Fill the dictionary using the dict(key=value[, ...]) constructor
            cp = dict(population=population, generation=gen, halloffame=halloffame,
                      logbook=logbook, rndstate=random.getstate())
            if id is None:
                cp_name = "checkpoint_ea.pkl"
            else:
                cp_name = "checkpoint_ea_{}.pkl".format(id)
            pickle.dump(cp, open(cp_name, "wb"))
            
        gen_time = datetime.datetime.now() - start_time
        total_time = total_time + gen_time
        #print("Time ", total_time)
        if total_time > datetime.timedelta(hours=4*24):
            print("Time limit exceeded.")
            break 

    return population, logbook


def myEAMuCommaLambda(population, startgen, toolbox, mu, lambda_, cxpb, mutpb, ngen,
                      stats=None, halloffame=None, logbook=None, verbose=False, id=None):

    assert lambda_ >= mu, "lambda must be greater or equal to mu."

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    if logbook is None:
        logbook = tools.Logbook()
        logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    record = stats.compile(population) if stats is not None else {}
    logbook.record(gen=startgen, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    total_time = datetime.timedelta(seconds=0) 
    for gen in range(startgen+1, ngen):
        start_time = datetime.datetime.now()
        
        # Vary the population
        offspring = algorithms.varOr(population, toolbox, lambda_, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Select the next generation population
        population[:] = toolbox.select(offspring, mu)

        # Update the statistics with the new population
        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

        if gen % 1 == 0:
            # Fill the dictionary using the dict(key=value[, ...]) constructor
            cp = dict(population=population, generation=gen, halloffame=halloffame,
                      logbook=logbook, rndstate=random.getstate())
            if id is None:
                cp_name = "checkpoint_es.pkl"
            else:
                cp_name = "checkpoint_es_{}.pkl".format(id)
            pickle.dump(cp, open(cp_name, "wb"))

        gen_time = datetime.datetime.now() - start_time
        total_time = total_time + gen_time
        if total_time > datetime.timedelta(hours=4*24):
            print("Time limit exceeded.")
            break 

            
            
    return population, logbook
