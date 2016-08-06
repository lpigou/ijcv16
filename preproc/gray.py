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


COMPRESS = False
N_PROCS = 4
V_SHAPE = (64, 64)
DST = "data/gray_i64_train/"

def exit(): 
    this_file = os.path.basename(__file__)
    sys.exit("Usage: python preproc/%s [video_size=64] [n_procs=4] [subset=train]"%this_file)


def main():
    global V_SHAPE, N_PROCS, DST
    if not len(sys.argv) in [1,2,3,4]: exit()
    subset = "train"
    if len(sys.argv) > 1: V_SHAPE = (int(sys.argv[1]),)*2
    if len(sys.argv) > 2: N_PROCS = int(sys.argv[2])
    if len(sys.argv) > 3: subset = int(sys.argv[3])
    if not subset in ["train", "test"]: exit()
    DST = "data/gray_i%i_%s/"%(V_SHAPE[0], subset)
    print "size", V_SHAPE, "jobs", N_PROCS, "subset", subset

    # cleanup
    to_delete = glob("data/%s/Sample*/"%subset) # folders
    for p in to_delete: shutil.rmtree(p)
    to_delete = glob("/dev/shm/Sample*.mp4")
    for p in to_delete: os.remove(p)

    # grab sample files
    file_paths = glob("data/%s/*.zip"%subset)
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
    # file_paths.sort(revert=True)
    pool.map(partial(extract_depth, subset=subset), file_paths)
    # extract_depth(file_paths[504-1])
    # extract_depth(file_paths[random.randint(len(file_paths))])


def extract_depth(file_path, subset="train"):
    try:    
        vid_id = file_path.split("/")[-1][-8:-4]
        if os.path.exists(DST+"%s.npz"%vid_id): return
        
        print "Processing", file_path

        archive = zipfile.ZipFile(file_path, 'r')
        file_name = file_path.split("/")[-1][:-4]+'_color.mp4'
        vid_path = "/dev/shm/%s"%file_name
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

        cap = cv2.VideoCapture(vid_path)
        assert cap.isOpened()
        n_frames = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))

        vid = empty((n_frames,)+V_SHAPE, uint8)
        for i in range(n_frames):
            img = cap.read()[1]
            img = cv2.cvtColor(img, cv2.cv.CV_RGB2GRAY)
            img = img[center[0]-ofs:center[0]+ofs, center[1]-ofs:center[1]+ofs]
            img = cv2.resize(img, V_SHAPE, interpolation=cv2.INTER_LINEAR)
            # cv2.imshow("img",img)
            # cv2.waitKey(10)
            vid[i] = img
        # sleep(1000)
        # ZMUV normalisation per sample
        # mu = vid.mean()
        # sigma = sqrt(vid.var() + 1e-15)
        # vid = (vid.astype(float)-mu)/sigma
        # old_min = vid.min()
        # vid = (vid-old_min)*255./(vid.max()-old_min)
        # vid = vid.astype("uint8")

        if COMPRESS: savez_compressed(DST+"%s.npz"%vid_id, vid)
        else: save(DST+"%s.npy"%vid_id, vid)

        csvfile.close()
        archive.close()
        os.remove(vid_path)
    except: "err: "+file_path


if __name__ == '__main__':
    import signal

    def siginthndlr(sig, frame):
        os.killpg(0, signal.SIGKILL)
        sys.exit()

    signal.signal(signal.SIGINT, siginthndlr) #Register SIGINT handler function
    main()