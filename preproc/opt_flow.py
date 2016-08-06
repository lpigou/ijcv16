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
# sys.path.append("/home/lio/deepflow")
# from py_fastdeepflow import fastdeepflow

sys.path.append("..")
import skeleton
os.chdir("..")

import yaml

with open("paths.yaml","r") as f: PATH = yaml.load(f)


COMPRESS = False
N_PROCS = 4
V_SHAPE = (64, 64)
DST = "data/optfl_i64_train/"

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
    DST = PATH["preproc"]+"optfl_i%i_%s/"%(V_SHAPE[0], subset)
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
    pool = Pool(N_PROCS)
    if subset=="train": del file_paths[417-1] # sample 417 is missing, because it had corrput data
    pool.map(partial(extract_rgbd, subset=subset), file_paths)
    # extract_rgbd(file_paths[583])

def draw_hsv(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[:,:,0], flow[:,:,1]
    ang = arctan2(fy, fx) + pi
    v = sqrt(fx*fx+fy*fy)
    hsv = zeros((h, w, 3), uint8)
    hsv[...,0] = ang*(180/pi/2)
    hsv[...,1] = 255
    hsv[...,2] = minimum(v*4, 255)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr

def extract_rgbd(file_path, subset="train"):
    try:    
        vid_id = file_path.split("/")[-1][-8:-4]
        if os.path.exists(DST+"%s.npy"%vid_id): return
        
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
        n_frames = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
        # n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        vid = empty((n_frames,2)+V_SHAPE, float32)

        # prev = zeros(V_SHAPE, dtype="uint8")

        for i in range(n_frames):
            img = cap.read()[1]
            img = cv2.cvtColor(img, cv2.cv.CV_RGB2GRAY)
            # img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = img[center[0]-ofs:center[0]+ofs, center[1]-ofs:center[1]+ofs]

            # sleep(1000)

            # print img.shape, img.dtype, img.min(), img.max()
            # print prev.shape, prev.dtype, prev.min(), prev.max()

            if i==0: prev = img.copy()
            # flow = empty(img.shape+(2,), img.dtype)
            # print flow.shape
            # flow = cv2.calcOpticalFlowFarneback(prev, img, None, pyr_scale=0.5, levels=5, winsize=7, iterations=3, poly_n=5, poly_sigma=1.1, flags=0)
            flow = cv2.calcOpticalFlowFarneback(prev, img, pyr_scale=0.5, levels=5, winsize=7, iterations=3, poly_n=5, poly_sigma=1.1, flags=0)
            # u, v = fastdeepflow.calc_flow(prev, img)
            prev = img.copy()
            # img = cv2.resize(img, V_SHAPE, interpolation=cv2.INTER_LINEAR)
            # flow = concatenate((u[...,newaxis],v[...,newaxis]), axis=2)
            # print flow.shape, flow.dtype, flow.min(), flow.max()

            # print flow.shape, flow.dtype, flow.min(), flow.max()
            # print 

            # u = scale_minmax(normalize(u),0,255).astype("uint8")
            # v = scale_minmax(normalize(v),0,255).astype("uint8")
            # ang = arctan2(v, u) + pi
            # v = sqrt(u*u+v*v)
            # hsv = zeros((ofs*2, ofs*2, 3), uint8)
            # hsv[...,0] = ang*(180/pi/2)
            # hsv[...,1] = 255
            # hsv[...,2] = minimum(v*100, 255)
            # flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

            # flow = draw_hsv(flow)
            # img = cv2.resize(img, (300,300), interpolation=cv2.INTER_NEAREST)
            flow = cv2.resize(flow, V_SHAPE, interpolation=cv2.INTER_LINEAR)
            # cv2.imshow("img", flow)
            # cv2.imshow("img2",img)

            # import scipy.misc
            # img = draw_hsv(flow)
            # scipy.misc.toimage(img, cmin=0.0, cmax=255.0).save('%s.png' % (i,))

            # cv2.waitKey(1)
            vid[i,0], vid[i,1]= flow[...,0], flow[...,1]



        # sleep(100000)

        # cap = cv2.VideoCapture(paths["depth"])
        # assert cap.isOpened()
        # for i in range(n_frames):
        #     img = cap.read()[1]
        #     img = cv2.cvtColor(img, cv2.cv.CV_RGB2GRAY)
        #     img = img[center[0]-ofs:center[0]+ofs, center[1]-ofs:center[1]+ofs]
        #     img = cv2.resize(img, V_SHAPE, interpolation=cv2.INTER_LINEAR)
        #     # cv2.imshow("img",img)
        #     # cv2.waitKey(1)
        #     vid[i,-1] = img

        # ZMUV normalisation
        vid = scale_minmax(normalize(vid),0,255).astype("uint8")


        if COMPRESS: savez_compressed(DST+"%s.npz"%vid_id, vid)
        else: save(DST+"%s.npy"%vid_id, vid)

        csvfile.close()
        archive.close()
        os.remove(vid_path)
    except e: print "err:", file_path, e.message()


def scale_minmax(x, new_min=0, new_max=255):
    old_min = x.min()
    return 1.*(x-old_min)*(new_max-new_min)/(x.max()-old_min)+new_min


def normalize(x): 
    # ZMUV normalization
    mu = x.mean()
    sigma = sqrt(x.var() + 1e-9)
    x = (x.astype(float)-mu)/sigma
    return x


if __name__ == '__main__':
    import signal

    def siginthndlr(sig, frame):
        os.killpg(0, signal.SIGKILL)
        sys.exit()

    signal.signal(signal.SIGINT, siginthndlr) #Register SIGINT handler function
    main()