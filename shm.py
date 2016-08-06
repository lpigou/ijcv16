import os
import numpy as np
from time import time, sleep

EXPIRE_T = 60*5 # in seconds
FOLDER_PATH="/dev/shm/shmemmap/"


class memmap(object):

    def __init__(self, data=None, shape=None, dtype=None):
        if not data is None:
            shape = data.shape
            if not dtype: dtype = data.dtype

        if not os.path.exists(FOLDER_PATH): os.makedirs(FOLDER_PATH)

        sleep(0.0001) # make sure the name is unique

        file_path = FOLDER_PATH + "shm%f"%time()
        mmp = np.memmap(file_path, dtype=dtype, mode='w+', shape=shape)
        if not data is None: mmp[:] = data 
        del mmp

        self.file_path = file_path
        self.dtype = dtype
        self.shape = shape

    def get(self):
        return np.memmap(self.file_path, dtype=self.dtype, mode='r+', shape=self.shape)

    def clean(self): os.remove(self.file_path)


def test_basic_usage():
    import numpy.random as npr

    global UPDATE_T
    UPDATE_T = 1

    shape = (1000000,)
    x = memmap(data=npr.rand(*shape), dtype="float32")
    x_ = x.get()

    print x_[:5]

    x.clean()


def test_multiprocessing():
    import numpy.random as npr
    import mp_utils

    shape = (10000,1000)
    x_shm = memmap(data=npr.rand(*shape), dtype="float32")

    def f(e):
        x_shm.get()[e] = npr.rand(1000)
        return e

    n = 1000
    iterable = npr.permutation(n)

    times=[]
    for i in xrange(5):   
        t0 = time()
        x_shm.poke()
        mp_utils.map_unordered(f, iterable, 8)
        times.append((time()-t0)*1000)
    print np.mean(times)

    times=[]
    for i in xrange(5):   
        t0 = time()
        x_shm.poke()
        mp_utils.apply_unordered(f, iterable, 8)
        times.append((time()-t0)*1000)
    print np.mean(times)
    x_shm.clean()

def test_interrupt():
    import signal, sys
    import numpy.random as npr

    shape = (1000000,)
    x = memmap(data=npr.rand(*shape), dtype="float32")

    def siginthndlr(sig, frame):
        x.clean()
        os.killpg(0, signal.SIGKILL)
        sys.exit()

    signal.signal(signal.SIGINT, siginthndlr) #Register SIGINT handler function

    print "interrupt me"
    sleep(100000)


if __name__ == '__main__':
    test_basic_usage()
    test_multiprocessing()
    test_interrupt()
    


