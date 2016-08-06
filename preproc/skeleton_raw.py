import os
import sys
from numpy import *
from glob import glob
import shutil
import zipfile
from time import sleep, time
import csv
from functools import partial
# from multiprocessing.pool import ThreadPool as Pool
from multiprocessing import Pool


N_PROCS = 8


def exit(): sys.exit("Usage: python preproc/skeleton_raw.py [subset=train]")


def main():

    if not len(sys.argv) in [1,2]: exit()
    elif len(sys.argv) == 1: subset = "train"
    else:
        subset = sys.argv[1]
        if not subset in ["train", "test"]: exit()

    # cleanup: delete open folders, if any
    folder_paths = glob("data/%s/Sample*/"%subset)
    for p in folder_paths: shutil.rmtree(p)

    # grab sample files
    file_paths = glob("data/%s/*.zip"%subset)
    file_paths.sort()
    print len(file_paths), "files found"
    assert len(file_paths) > 0

    # make sure no file is missing
    offset = 1 if not subset == "test" else 701
    for i, path in enumerate(file_paths):
        assert int(path.split("/")[-1][-8:-4]) == i + offset

    # preprocess
    pool = Pool(N_PROCS)
    x = pool.map(partial(extract_skeletons, subset=subset), file_paths)

    # store
    dst_path = "data/skeleton_raw_%s.npy"%subset
    print "Storing in " + dst_path
    save(dst_path, x)

    # testing
    print "Testing"
    x_test = load(dst_path)
    assert len(x_test) == len(x)
    for _ in range(100):
        r = random.randint(len(x))
        assert array_equal(x_test[r], x[r])


def extract_skeletons(file_path, subset="train"):
    print "Processing", file_path

    archive = zipfile.ZipFile(file_path, 'r')
    csvfile = archive.open(file_path.split("/")[-1][:-4]+'_skeleton.csv')
    filereader = csv.reader(csvfile, delimiter=',')

    skeletons = [row for row in filereader]
    skeletons = array(skeletons, dtype=float32)
    assert skeletons.shape[1] == 180

    csvfile.close()
    archive.close()
    return skeletons


if __name__ == '__main__':
    import signal

    def siginthndlr(sig, frame):
        os.killpg(0, signal.SIGKILL)
        sys.exit()

    signal.signal(signal.SIGINT, siginthndlr) #Register SIGINT handler function
    main()