from multiprocessing import Process, Queue

def worker(name, evalfunc, querries, answers):
    while True:
        querry = querries.get()
        #print("worker ", name, "evaluating")
        answer = evalfunc(querry)
        #print("worker ", name, "finished")
        answers.put(answer)

    
class Pool:
    """ Pool of workers. Workers consume tasks from the querries queue and 
        feed answers to the answers queue.
    """

    
    def __init__(self, processors, evalfunc):

        self.querries = Queue()
        self.answers = Queue()

        self.workers = []
        for i in range(processors):
             worker_i = Process(target=worker, args=(i, evalfunc, self.querries, self.answers))
             self.workers.append(worker_i)
             worker_i.start()        # Launch worker() as a separate python process

    def putQuerry(self, querry):
        self.querries.put(querry)
             
    def getAnswer(self):
        return self.answers.get()


    def close(self):
        for w in self.workers:
            w.terminate()
        #print("pool killed")
        
