import os
import sys
from numpy import *
from glob import glob
import shutil
import zipfile
from time import sleep, time
import csv
# from multiprocessing.pool import ThreadPool as Pool
from multiprocessing import Pool


N_PROCS = 8


def exit(): sys.exit("Usage: python preproc/labels_raw_test.py")


def main():

    # cleanup: delete open folders, if any
    folder_paths = glob("../data/test/Sample*/")
    for p in folder_paths: shutil.rmtree(p)

    # grab sample files
    file_paths = glob("../gt/*_labels.csv")
    file_paths.sort()
    print len(file_paths), "files found"
    assert len(file_paths) > 0

    # make sure no file is missing
    # for i, path in enumerate(file_paths):
    #     assert int(path.split("/")[-1][-8:-4]) == i + 1

    # preprocess
    pool = Pool(N_PROCS)
    y = pool.map(extract_labels, file_paths)

    # extract_labels(file_paths[0])


    # store
    dst_path = "../data/labels_raw_test.npy"
    print "Storing in " + dst_path
    save(dst_path, y)

    # testing
    print "Testing"
    y_test = load(dst_path)
    assert len(y_test) == len(y)
    for _ in range(10):
        r = random.randint(len(y))
        assert array_equal(y_test[r], y[r])
        print y_test[r].shape


def extract_labels(file_path):
    print "Processing", file_path

    csvfile = open(file_path.replace("labels","data"),"r")
    filereader = csv.reader(csvfile, delimiter=',')
    n_frames = int(filereader.next()[0])
    del filereader
    csvfile.close()

    # grab labels
    csvfile = open(file_path,"r")
    filereader = csv.reader(csvfile, delimiter=',')
    labels = zeros(n_frames, uint8)
    for row in filereader:
        labels[int(row[1])-1:int(row[2])-1] = int(row[0])

    csvfile.close()
    return labels


if __name__ == '__main__':
    import signal

    def siginthndlr(sig, frame):
        os.killpg(0, signal.SIGKILL)
        sys.exit()

    signal.signal(signal.SIGINT, siginthndlr) #Register SIGINT handler function
    main()