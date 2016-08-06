import numpy as np
import numpy.random as npr
import theano
import os
import multiprocessing as mp
from time import time, sleep
import skimage.transform as tf
from functools import partial
import ctypes
from glob import glob
import signal
import shutil

import utils
import shm
import mp_utils
import split
import yaml

with open("paths.yaml","r") as f: PATH = yaml.load(f)

floatX = theano.config.floatX
fast_warp = partial(tf._warps_cy._warp_fast, mode='nearest', order=0)
resize_im = partial(tf.resize, order=0, mode='constant', cval=0)
DEFAULT_AUG = {
    'scale': (1,1),
    't_scale': (1,1),
    'shear': (0,0),
    'rot': (0,0),
    'trans_x': (0,0),
    'trans_y': (0,0),
}


class DataLoader(object):

    def __init__(self, 
        seq_length=64,
        chunk_size=10,
        batch_size=1,
        cut_off=16,
        channels=slice(0,4),
        im_shape=(64,64),
        data_path=PATH["preproc"]+"rgbd_i64_train/",
        labels_path=PATH["labels"]+"labels_raw.npy",
        n_jobs=mp.cpu_count(),
        n_classes=21,
        aug_params=DEFAULT_AUG,
        block=True,
        n_mem_slots = 20,
        ):
        self.seq_length = seq_length
        self.chunk_size = chunk_size
        self.batch_size = batch_size
        self.cut_off = cut_off
        self.channels = channels
        self.im_shape = im_shape
        self.data_path = data_path
        self.labels_path = labels_path
        self.n_jobs = n_jobs
        self.n_classes = n_classes
        self.aug_params = DEFAULT_AUG
        self.aug_params.update(aug_params)
        self.block = block
        self.n_mem_slots = n_mem_slots

        if isinstance(channels, slice): 
            n_channels = channels.stop - channels.start  
        else: n_channels = len(channels)
        self.n_channels = n_channels
        self.x_shape = (chunk_size, seq_length, n_channels)+im_shape
        self.y_shape = (chunk_size, seq_length, n_classes)

    def initialize(self, x_shared, y_shared, m_shared, start=True):
        self.x_shared, self.y_shared, self.m_shared = x_shared, y_shared, m_shared

        labels = np.load(self.labels_path)
        labels[:] = map(lambda y: utils.one_hot(y,self.n_classes), labels)
        self.labels_shm = shm.memmap(data=labels)

        self.x_shm, self.y_shm = [], []
        for _ in range(3):
            self.x_shm.append(shm.memmap(shape=self.x_shape, dtype=floatX))
            self.y_shm.append(shm.memmap(shape=self.y_shape, dtype=floatX))

        self.is_terminated = mp.Value(ctypes.c_bool, False)
        self.write_locks = [mp.Lock(), mp.Lock()]
        self.read_locks = [mp.Lock(), mp.Lock()]
        for lock in self.write_locks: lock.acquire()
        for lock in self.read_locks: lock.acquire()
        self.validating = mp.Lock()

        self.loader_path = "/dev/shm/loader/"
        if not os.path.exists(self.loader_path): os.mkdir(self.loader_path)
        self.loader_path += "%f"%time()

        self.mem_locks = []
        for i in range(self.n_mem_slots): 
            self.mem_locks.append(mp.Lock())
            video_number = split.train_idxs[npr.randint(len(split.train_idxs))]
            path = self.data_path+str(video_number).zfill(4)+".npy"
            shutil.copyfile(path, self.loader_path+"%i_%i"%(i, video_number))

        if start: self.start()

        signal.signal(signal.SIGINT, self.terminate)
        signal.signal(signal.SIGTERM, self.terminate)

    def start(self):
        jobs = []

        jobs.append(mp.Process(target=self.master))
        jobs.append(mp.Process(target=self.memory_loader))

        for job in jobs: job.start()

        self.jobs = jobs

    def memory_loader(self):
        while not self.is_terminated.value:
            self.validating.acquire()
            self.validating.release()
            video_number = split.train_idxs[npr.randint(len(split.train_idxs))]
            path = self.data_path+str(video_number).zfill(4)+".npy"

            mid = -1
            while mid < 0:
                for i in npr.permutation(self.n_mem_slots):
                    if self.mem_locks[i].acquire(False):
                        mid  = i
                        break

            old_path = glob(self.loader_path+"%i_*"%mid)[0]
            os.remove(old_path)
            shutil.copyfile(path, self.loader_path+"%i_%i"%(mid, video_number))
            self.mem_locks[i].release()

    def load_video(self):
        mid = -1
        while mid < 0:
            for i in npr.permutation(self.n_mem_slots):
                if self.mem_locks[i].acquire(False):
                    mid  = i
                    break
        path = glob(self.loader_path+"%i_*"%mid)[0]
        video_number = int(path.split("_")[-1])
        video = np.load(path)
        self.mem_locks[mid].release()
        return video, video_number

    def master(self):
        chunk_count = 0
        try: self.write_locks[0].release()
        except ValueError: pass
        print self.__class__.__name__, "running"

        while not self.is_terminated.value:
            npr.seed()
            idx = chunk_count % 2
            chunk_count += 1
            idxs_train = npr.randint(len(split.train_idxs), size=self.chunk_size)
            idxs_train = split.train_idxs[idxs_train]
            idxs_chunk = np.arange(self.chunk_size)

            mp_utils.apply_unordered(
                function=self.slave, 
                iterable=zip(idxs_chunk, idxs_train),
                n_jobs=self.n_jobs)

            if self.block:
                # wait for writing permission
                self.write_locks[idx].acquire()

                self.x_shm[idx].get()[:] = self.x_shm[-1].get().copy()
                self.y_shm[idx].get()[:] = self.y_shm[-1].get().copy()

                # unlock reading
                try: self.read_locks[idx].release()
                except ValueError: pass

    def slave(self, indices): 
        npr.seed()
        idx_chunk, idx_train = indices

        # path = self.data_path+str(idx_train).zfill(4)+".npy"
        # video = np.load(path)
        video, idx_train = self.load_video()

        video = video[:, self.channels] / 255.

        # calculate wich frame indices will be used
        t_scale = npr.uniform(*self.aug_params["t_scale"])
        orig_size = int(np.round(self.seq_length * t_scale))
        max_ofs = len(video) - orig_size
        offset = npr.randint(max_ofs)
        sample_idxs = utils.resample_arr(offset+np.arange(orig_size), self.seq_length)

        sample = video[sample_idxs]

        flip_image = True if npr.rand() < .5 else False
        shear = np.radians(npr.uniform(*self.aug_params["shear"]))
        rot = np.radians(npr.uniform(*self.aug_params["rot"]))
        scale_y = npr.uniform(*self.aug_params["scale"])
        scale_x = npr.uniform(*self.aug_params["scale"])
        scale = (scale_y,scale_x)

        # recenter the image (it's decentered by the scaling and shearing)
        size = sample.shape[-1]
        trans = [size/2.*(1.-scale_y), size/2.*(1.-scale_x)]
        if shear: trans[0] += np.tan(shear)*size/2

        trans[0] += npr.uniform(npr.uniform(*self.aug_params["trans_x"]))
        trans[1] += npr.uniform(npr.uniform(*self.aug_params["trans_y"]))

        at = tf.AffineTransform(matrix=None, scale=scale, rotation=rot, 
                                shear=shear, translation=trans)

        aug_sample = np.empty(sample.shape[:-2]+self.im_shape, dtype=sample.dtype)
        for i in range(sample.shape[0]):
            for j in range(sample.shape[1]):
                img = sample[i,j]
                if flip_image: img = np.fliplr(img)
                if img.shape != self.im_shape:
                    if self.im_shape[0] == 32: img = img[::2,::2]
                    else: img = resize_im(img, self.im_shape)
                img = fast_warp(img, at._matrix, output_shape=self.im_shape)
                aug_sample[i,j] = img

        self.x_shm[-1].get()[idx_chunk] = aug_sample
        label = self.labels_shm.get()[idx_train-1]
        self.y_shm[-1].get()[idx_chunk] = label[sample_idxs]

    def create_train_generator(self):
        chunk_count = 0
        x = [x_shm.get() for x_shm in self.x_shm]
        y = [y_shm.get() for y_shm in self.y_shm]

        if self.block:
            while not self.is_terminated.value:
                idx = chunk_count % 2
                chunk_count += 1

                # unlock writing to the other shm array
                try: self.write_locks[(idx + 1) % 2].release()
                except ValueError: pass

                # wait for reading permission
                self.read_locks[idx].acquire()

                self.x_shared.set_value(x[idx], borrow=True)
                self.y_shared.set_value(y[idx], borrow=True)

                yield
        else:
            while not self.is_terminated.value:
                self.x_shared.set_value(x[-1], borrow=True)
                self.y_shared.set_value(y[-1], borrow=True)
                yield

    def create_valid_generator(self, n_samples=len(split.valid_idxs)):
        labels = self.labels_shm.get()[split.valid_idxs-1]
        n_batches = n_samples // self.batch_size

        self.validating.acquire()

        for b in range(n_batches):    
            max_len = np.max([len(i) for i in labels[b*self.batch_size:(b+1)*self.batch_size]])
            eff_len = self.seq_length - 2*self.cut_off
            valid_len = self.seq_length + int(np.ceil((max_len-self.seq_length) / \
                float(eff_len))*eff_len)
            m = np.zeros((self.batch_size, valid_len), dtype="uint8")
            x = np.zeros((self.batch_size, valid_len, self.n_channels)+self.im_shape, dtype=floatX)

            for idx in range(self.batch_size):

                path = self.data_path+str(split.valid_idxs[self.batch_size*b+idx]).zfill(4)+".npy"
                smpl = np.load(path)
                smpl = smpl[:,self.channels] / 255.

                # resize if necessary
                if self.im_shape != smpl.shape[-2:]:
                    new_smpl = np.empty(smpl.shape[:-2]+self.im_shape, dtype=smpl.dtype)
                    for i in range(smpl.shape[0]):
                        for j in range(smpl.shape[1]):
                            new_smpl[i,j] = resize_im(smpl[i,j], self.im_shape)
                    smpl = new_smpl

                x[idx, :len(smpl)] = smpl
                m[idx, :len(smpl)] = 1

            self.x_shared.set_value(x, borrow=True)
            self.m_shared.set_value(m, borrow=True)

            yield

        self.validating.release()

    def get_labels(self): return self.labels_shm.get()

    def terminate(self, sig=None, frame=None): 
        self.is_terminated.value = True
        for l in self.write_locks: 
            try: l.release()
            except ValueError: pass
        for l in self.read_locks: 
            try: l.release()
            except ValueError: pass
        try: self.validating.release()
        except ValueError: pass
        for shmem in self.x_shm + self.y_shm: shmem.clean()
        self.labels_shm.clean()
        paths = glob(self.loader_path+"*")
        for path in paths: os.remove(path)
        for job in self.jobs: job.join()
        print self.__class__.__name__, "terminated"


def test_generator():
    aug_params = {
        'scale': (1 / 1.1, 1.1),
        't_scale': (1 / 1.2, 1.2),
        'shear': (-2, 2),
        'rot': (-2, 2),
        'trans_x': (-10, 10),
        'trans_y': (-5, 5),
    }


    loader = DataLoader(
        chunk_size=8,
        seq_length=64,
        data_path=PATH["preproc"]+"rgbd_i64_train/",
        labels_path=PATH["labels"]+"labels_raw.npy",
        im_shape=(64,64),
        n_jobs=4,
        n_mem_slots=10,
        block=True,
        aug_params=aug_params)

    x_shared = theano.shared(np.empty((0,0,0,0,0),dtype=floatX))
    y_shared = theano.shared(np.empty((0,0,0),dtype=floatX))
    m_shared = theano.shared(np.empty((0,0),dtype=floatX))

    # print x_shared.get_value().shape

    generator = loader.create_train_generator()

    loader.initialize(x_shared, y_shared, m_shared)

    # sleep(10000)
    t0 = time()
    generator.next()
    print (time()-t0)*1000, "ms"

    x = x_shared.get_value()
    y = y_shared.get_value()

    # for _ in range(2):
    #     sleep(1)
    #     t0 = time()
    #     generator.next()
    #     print (time()-t0)*1000, "ms"

    import cv2
    cv2.namedWindow("img")
    cv2.moveWindow("img", 500, 500)
    # cv2.resizeWindow("img", 1024,1024)
    cv2.namedWindow("d")
    cv2.moveWindow("d", 500, 800)
    # cv2.resizeWindow("d", 1024,1024)
    size = 512
    print x.shape
    print y.shape

    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            print np.argmax(y[i,j]),
            rgb = x[i,j,:3].swapaxes(0,2).swapaxes(0,1)
            d = x[i,j,-1]
            # cv2.normalize(rgb, dst=rgb, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            # d = cv2.normalize(d, dst=d, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            rgb = cv2.resize(rgb, (size,size), interpolation=cv2.INTER_NEAREST)
            d = cv2.resize(d, (size,size), interpolation=cv2.INTER_NEAREST)
            cv2.imshow("img",rgb)
            cv2.imshow("d",d)
            # cv2.moveWindow("img",0,0)
            # cv2.moveWindow("d",500,0)
            cv2.waitKey(25)

    cv2.destroyAllWindows()
    loader.terminate()

def test_validation():
    loader = DataLoader(
        chunk_size=1,
        seq_length=64,
        data_path=PATH["preproc"]+"rgbd_i64_train/",
        labels_path=PATH["labels"] + "labels_raw.npy",
        im_shape=(64,64),
        n_jobs=4,
        block=True)

    x_shared = theano.shared(np.empty((0,0,0,0,0),dtype=floatX))
    y_shared = theano.shared(np.empty((0,0,0),dtype=floatX))
    m_shared = theano.shared(np.empty((0,0),dtype=floatX))

    print x_shared.get_value().shape

    loader.initialize(x_shared, y_shared, m_shared)

    valid_gen = loader.create_valid_generator()

    t0 = time()
    valid_gen.next()
    print (time()-t0)*1000, "ms"

    t0 = time()
    valid_gen.next()
    print (time()-t0)*1000, "ms"

    x = x_shared.get_value()
    m = m_shared.get_value()

    import cv2
    size = 200
    print x.shape
    print m.shape

    y = loader.get_labels()[-len(split.valid_idxs):]

    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            print np.argmax(y[i][j]),
            rgb = x[i,j,:3].swapaxes(0,2).swapaxes(0,1)
            d = x[i,j,-1]
            rgb = cv2.normalize(rgb, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            d = cv2.normalize(d, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            rgb = cv2.resize(rgb, (size,size), interpolation=cv2.INTER_NEAREST)
            d = cv2.resize(d, (size,size), interpolation=cv2.INTER_NEAREST)
            cv2.imshow("img",rgb)
            cv2.imshow("d",d)
            cv2.moveWindow("img",0,0)
            cv2.moveWindow("d",500,0)
            cv2.waitKey(50)


if __name__ == '__main__':
    test_generator()
    # test_validation()

