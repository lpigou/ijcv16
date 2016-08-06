import multiprocessing as mp
import numpy as np


def map_unordered(function, iterable, n_jobs=mp.cpu_count()):
    queue_in   = mp.Queue()
    queue_out  = mp.Queue() 

    def worker(function):
        for element in iter(queue_in.get, None): 
            queue_out.put(function(element))

    for _ in range(n_jobs):
        job = mp.Process(target=worker, args=(function,)) 
        job.daemon = True
        job.start()

    for element in iterable: queue_in.put(element)
    for _ in range(n_jobs): queue_in.put(None)

    return [queue_out.get() for _ in range(len(iterable))]


def apply_unordered(function, iterable, n_jobs=mp.cpu_count()):
    queue_in   = mp.Queue()

    def worker(function):
        for element in iter(queue_in.get, None): function(element) 

    jobs = []
    for _ in range(n_jobs):
        job = mp.Process(target=worker, args=(function,)) 
        job.daemon = True
        job.start()
        jobs.append(job)

    for element in iterable: queue_in.put(element)
    for _ in range(n_jobs): queue_in.put(None)

    for job in jobs: job.join()


def test():
    def f(e):
        for i in xrange(500): np.sqrt(e+1)
        return np.sqrt(e+1)

    class A():
        def __init__(self, param):
            self.param = param
            self.lock = mp.Lock()
            self.lock.acquire()

        def f(self, e):
            for i in xrange(100): np.sqrt(e+1)
            return np.sqrt(self.param+1)

        def run(self):
            apply_unordered(self.f, np.zeros(100), 4)
            self.lock.release()

    a = A(5)
    a.run()

    n = 1000

    from time import time

    t0 = time()
    [f(e) for e in np.zeros(n)]
    print "python", "%.3fms"% ((time()-t0)*1000,)

    t0 = time()
    pool = mp.Pool(4)
    gen = pool.imap_unordered(f_global, np.zeros(n))
    for g in gen: pass
    print "multiprocessing pool", "%.3fms"% ((time()-t0)*1000,)
        

    t0 = time()
    res = map_unordered(f, np.zeros(n), 4)
    print "map_unordered", "%.3fms"% ((time()-t0)*1000,)

    t0 = time()
    apply_unordered(f, np.zeros(n), 4)
    print "apply_unordered", "%.3fms"% ((time()-t0)*1000,)


def f_global(e):
    for i in xrange(500): np.sqrt(e+1)
    return np.sqrt(e+1)


if __name__ == '__main__': test()