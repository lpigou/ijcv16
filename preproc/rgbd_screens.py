import os
import sys
from numpy import *
from glob import glob
import shutil
import zipfile
from time import sleep, time
import cv2
import csv
from functools import partial
# from multiprocessing.pool import ThreadPool as Pool
from multiprocessing import Pool

sys.path.append("..")
import skeleton
os.chdir("..")

import yaml

with open("paths.yaml","r") as f: PATH = yaml.load(f)


COMPRESS = False
SAVE = False
N_PROCS = 1
V_SHAPE = (64, 64)
DST = ""

def exit(): 
    this_file = os.path.basename(__file__)
    sys.exit("Usage: python %s [video_size=64] [n_procs=4] [subset=train]"%this_file)


def main():
    global V_SHAPE, N_PROCS, DST
    if not len(sys.argv) in [1,2,3,4]: exit()
    subset = "train"
    if len(sys.argv) > 1: V_SHAPE = (int(sys.argv[1]),)*2
    if len(sys.argv) > 2: N_PROCS = int(sys.argv[2])
    if len(sys.argv) > 3: subset = sys.argv[3]
    if not subset in ["train", "test"]: exit()
    DST = PATH["preproc"]+"rgbd_i%i_%s/"%(V_SHAPE[0], subset)
    print "size", V_SHAPE, "jobs", N_PROCS, "subset", subset

    # cleanup
    to_delete = glob(PATH["videos"]+"%s/Sample*/"%subset) # folders
    for p in to_delete: shutil.rmtree(p)
    to_delete = glob("/dev/shm/Sample*.mp4")
    for p in to_delete: os.remove(p)

    # grab sample files
    file_paths = glob(PATH["videos"]+"%s/*.zip"%subset)
    file_paths.sort()
    print len(file_paths), "files found"
    assert len(file_paths) > 0

    # make sure no file is missing
    offset = 1 if not subset == "test" else 701
    for i, path in enumerate(file_paths):
        assert int(path.split("/")[-1][-8:-4]) == i + offset

    if not os.path.exists(DST): os.makedirs(DST)

    # preprocess
    # pool = Pool(N_PROCS)
    # if subset=="train": del file_paths[417-1] # sample 417 is missing, because it had corrput data
    # pool.map(partial(extract_rgbd, subset=subset), file_paths)
    extract_rgbd(file_paths[6])


import scipy.misc
FRAME = 250

def extract_rgbd(file_path, subset="train"):
    try:    
        vid_id = file_path.split("/")[-1][-8:-4]
        if SAVE and os.path.exists(DST+"%s.npy"%vid_id): return
        
        print "Processing", file_path

        archive = zipfile.ZipFile(file_path, 'r')
        paths = {}
        for ext in ['color', 'depth']:
            file_name = file_path.split("/")[-1][:-4]+'_%s.mp4'%ext
            vid_path = "/dev/shm/%s"%file_name
            paths[ext] = vid_path
            vid_data = archive.read(file_name)
            with open(vid_path, 'w') as f: f.write(vid_data)
            del vid_data

        # get skeleton
        csvfile = archive.open(file_path.split("/")[-1][:-4]+'_skeleton.csv')
        filereader = csv.reader(csvfile, delimiter=',')
        skeletons = [row for row in filereader]
        skeletons = array(skeletons, dtype=float32)
        head = skeleton.select(skeletons, [["Head", "PixelCoord"]])
        hip = skeleton.select(skeletons, [["HipCenter", "PixelCoord"]])

        _head = tuple(head[FRAME])
        _hip = tuple(hip[FRAME])

        hip = hip[unique(hip.nonzero()[0])]
        head = head[unique(head.nonzero()[0])]
        height = mean(hip[:,1]-head[:,1])
        center = [mean(head[:,1])+height*0.7, mean(head[:,0])]
        ofs = height

        if center[0] - ofs < 0: center[0] = ofs
        if center[0] + ofs > 640: center[0] = 640-ofs
        if center[1] - ofs < 0: center[1] = ofs
        if center[1] + ofs > 480: center[1] = 480-ofs

        # for i, vid_path in enumerate(paths.values()):
        cap = cv2.VideoCapture(paths["color"])
        assert cap.isOpened()
        # n_frames = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        vid = empty((n_frames,4)+V_SHAPE, uint8)
        for i in range(n_frames):
            imgo = cap.read()[1]
            # img = cv2.cvtColor(img, cv2.cv.CV_RGB2GRAY)
            img = imgo[center[0]-ofs:center[0]+ofs, center[1]-ofs:center[1]+ofs]

            if i == FRAME:
                img = img[...,[2, 1, 0]]
                # img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_NEAREST)
                print img.shape, n_frames
                scipy.misc.toimage(img, cmin=0.0, cmax=255.0).save('imgs/rgbcrop.png')

                img = imgo[:, 80:-80]
                img = img[..., [2, 1, 0]]
                # img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_NEAREST)
                print img.shape, n_frames
                scipy.misc.toimage(img, cmin=0.0, cmax=255.0).save('imgs/rgb.png')

                cv2.circle(imgo, _head, 10, (255,255,0,128), thickness=2)
                cv2.circle(imgo, _hip, 10, (0,255,255,128), thickness=2)
                cv2.rectangle(imgo, (int(center[1]-ofs),int(center[0]-ofs)), (int(center[1]+ofs),int(center[0]+ofs)), (0,0,255,255), thickness=3)
                img = imgo[:, 80:-80]
                img = img[..., [2, 1, 0]]
                # img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_NEAREST)
                print img.shape, n_frames
                scipy.misc.toimage(img, cmin=0.0, cmax=255.0).save('imgs/rgbinfo.png')
                break

            img = cv2.resize(img, V_SHAPE, interpolation=cv2.INTER_LINEAR)
            # cv2.imshow("img",img)
            # cv2.waitKey(1)
            vid[i,0], vid[i,1], vid[i,2] = img[...,0], img[...,1], img[...,2]

        cap = cv2.VideoCapture(paths["depth"])
        assert cap.isOpened()
        for i in range(n_frames):
            img = cap.read()[1]
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = img[center[0]-ofs:center[0]+ofs, center[1]-ofs:center[1]+ofs]

            if i == FRAME:
                # img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_NEAREST)
                print img.shape, n_frames
                scipy.misc.toimage(img, cmin=0.0, cmax=255.0).save('imgs/depth.png')
                sys.exit()

            img = cv2.resize(img, V_SHAPE, interpolation=cv2.INTER_LINEAR)

            # cv2.imshow("img",img)
            # cv2.waitKey(1)
            vid[i,-1] = img


        # ZMUV normalisation
        # vid[:,:3] = scale_minmax(normalize(vid[:,:3]),0,255).astype("uint8")
        # vid[:,-1] = scale_minmax(normalize(vid[:,-1]),0,255).astype("uint8")
        if SAVE:
            if COMPRESS: savez_compressed(DST+"%s.npz"%vid_id, vid)
            else: save(DST+"%s.npy"%vid_id, vid)

        csvfile.close()
        archive.close()
        os.remove(vid_path)
    except e: print "err:", file_path, e.message()


# def scale_minmax(x, new_min=0, new_max=255):
#     old_min = x.min()
#     return 1.*(x-old_min)*(new_max-new_min)/(x.max()-old_min)+new_min


# def normalize(x): 
#     # ZMUV normalization
#     mu = x.mean()
#     sigma = np.sqrt(x.var() + 1e-9)
#     x = (x.astype(float)-mu)/sigma
#     return x


if __name__ == '__main__':
    import signal

    def siginthndlr(sig, frame):
        os.killpg(0, signal.SIGKILL)
        sys.exit()

    signal.signal(signal.SIGINT, siginthndlr) #Register SIGINT handler function
    main()