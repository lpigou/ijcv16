import theano
import os
import errno
import numpy as np
import csv
import re
import cPickle


floatX = theano.config.floatX
cast_floatX = np.float32 if floatX=="float32" else np.float64


def save_pkl(obj, path, protocol=cPickle.HIGHEST_PROTOCOL):
    with file(path, 'wb') as f: 
        cPickle.dump(obj, f, protocol=protocol)


def load_pkl(path):
    with file(path, 'rb') as f: 
        obj = cPickle.load(f)
    return obj


def resample_list(list_, size):
    orig_size = len(list_)
    ofs = orig_size//size//2
    delta = orig_size/float(size)
    return [ list_[ofs + int(i * delta)] for i in range(size) ]


def resample_arr(arr, size):
    orig_size = arr.shape[0]
    ofs = orig_size//size//2
    delta = orig_size/float(size)
    idxs = [ofs + int(i * delta) for i in range(size)]
    return arr[idxs]


def asarrayX(value):
    return theano._asarray(value, dtype=theano.config.floatX)


def one_hot(vec, m=None):
    if m is None: m = int(np.max(vec)) + 1
    return np.eye(m)[vec]


def make_sure_path_exists(path):
    """Try to create the directory, but if it already exist we ignore the error"""
    try: os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST: raise


def shared_mmp(data=None, file_name="shm", shape=(0,), dtype=floatX):
    """ Shared memory, only works for linux """
    if not data is None: shape = data.shape
    path = "/dev/shm/lio/"
    make_sure_path_exists(path)
    mmp = np.memmap(path+file_name+".mmp", dtype=dtype, mode='w+', shape=shape)
    if not data is None: mmp[:] = data
    return mmp


def open_shared_mmp(filename,  shape=None, dtype=floatX):
    path = "/dev/shm/lio/"
    return np.memmap(path+filename+".mmp", dtype=dtype, mode='r', shape=shape)


def normalize_zmuv(x, axis=0, epsilon=1e-9):
    """ Zero Mean Unit Variance Normalization"""
    mean = x.mean(axis=axis)
    std = np.sqrt(x.var(axis=axis) + epsilon)
    return (x - mean[np.newaxis,:]) / std[np.newaxis,:]


def jaccard_index(p, t):
    p[np.arange(p.shape[0]), np.argmax(p,axis=-1)] = 1.
    p[p<1.] = 0.
    p = p[:,1:]
    t = t[:,1:]
    gestures = t.sum(axis=0) > 0
    p_ = p[:,gestures]
    t_ = t[:,gestures]
    intersect = (p_*t_).sum(axis=0)
    aux = p_ + t_
    union = (aux > 0).sum(axis=0)+1e-15
    overlap = (intersect / union).sum() / (gestures.sum()+1e-15)
    return overlap


class struct:
    def __init__(self, **entries): 
        self.__dict__.update(entries)
    def __repr__(self):
        return '{%s}' % str(', '.join('%s : %s' % (k, repr(v)) for
      (k, v) in self.__dict__.iteritems()))
    def keys(self):
        return self.__dict__.keys()


def jaccard_index_chalearn(prediction_dir,truth_dir):
    """ Perform the overlap evaluation for a set of samples """
    worseVal=10000

    # Get the list of samples from ground truth
    gold_list = os.listdir(truth_dir)

    # For each sample on the GT, search the given prediction
    numSamples=0.0;
    score=0.0;
    for gold in gold_list:
        # Avoid double check, use only labels file
        if not gold.lower().endswith("_labels.csv"):
            continue

        # Build paths for prediction and ground truth files
        sampleID=re.sub('\_labels.csv$', '', gold)
        labelsFile = os.path.join(truth_dir, sampleID + "_labels.csv")
        dataFile = os.path.join(truth_dir, sampleID + "_data.csv")
        predFile = os.path.join(prediction_dir, sampleID + "_prediction.csv")

        # Get the number of frames for this sample
        with open(dataFile, 'rb') as csvfile:
            filereader = csv.reader(csvfile, delimiter=',')
            for row in filereader:
                numFrames=int(row[0])
            del filereader

        # Get the score
        numSamples+=1
        score+=gesture_overlap_csv(labelsFile, predFile, numFrames)

    return score/numSamples

def gesture_overlap_csv(csvpathgt, csvpathpred, seqlenght):
    """ Evaluate this sample agains the ground truth file """
    maxGestures=20

    # Get the list of gestures from the ground truth and frame activation
    gtGestures = []
    binvec_gt = np.zeros((maxGestures, seqlenght))
    with open(csvpathgt, 'rb') as csvfilegt:
        csvgt = csv.reader(csvfilegt)
        for row in csvgt:
            binvec_gt[int(row[0])-1, int(row[1])-1:int(row[2])-1] = 1
            gtGestures.append(int(row[0]))

    # Get the list of gestures from prediction and frame activation
    predGestures = []
    binvec_pred = np.zeros((maxGestures, seqlenght))
    if os.path.exists(csvpathpred):
        with open(csvpathpred, 'rb') as csvfilepred:
            csvpred = csv.reader(csvfilepred)
            for row in csvpred:
                binvec_pred[int(row[0])-1, int(row[1])-1:int(row[2])-1] = 1
                predGestures.append(int(row[0]))

    # Get the list of gestures without repetitions for ground truth and predicton
    gtGestures = np.unique(gtGestures)
    predGestures = np.unique(predGestures)

    # Find false positives
    falsePos=np.setdiff1d(gtGestures, np.union1d(gtGestures,predGestures))

    # Get overlaps for each gesture
    overlaps = []
    for idx in gtGestures:
        intersec = sum(binvec_gt[idx-1] * binvec_pred[idx-1])
        aux = binvec_gt[idx-1] + binvec_pred[idx-1]
        union = sum(aux > 0)
        overlaps.append(intersec/union)

    # Use real gestures and false positive gestures to calculate the final score
    return sum(overlaps)/(len(overlaps)+len(falsePos))