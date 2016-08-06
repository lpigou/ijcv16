from numpy import *
_load = load

JOINTS = ['HipCenter','Spine','ShoulderCenter','Head','ShoulderLeft','ElbowLeft','WristLeft','HandLeft','ShoulderRight','ElbowRight','WristRight','HandRight','HipLeft','KneeLeft','AnkleLeft','FootLeft','HipRight','KneeRight','AnkleRight','FootRight']
JOINT_PARAMS = ["WorldCoord","Orient","PixelCoord"]

def load(subset, selections=None):
    return _load("data/skeleton_raw_%s.npy"%subset)
    

def select(skeletons, selections):
    """ Example: select(skeletons,  ['HandRight',
                                    ["HipCenter",("WorldCoord","PixelCoord")],
                                    ["WristRight","Orient"] ]) """
    if not selections: return skeletons
    inds, new_inds = [], []
    params = ones((len(selections),3),bool)
    for i,s in enumerate(selections):
        if isinstance(s,str): inds.append(JOINTS.index(s))
        else: 
            inds.append(JOINTS.index(s[0]))
            params[i] = [False]*3
            p = s[1]
            if isinstance(p,str): p = [p]
            for j in p: params[i,JOINT_PARAMS.index(j)] = True

    for i,ind in enumerate(inds):
        _i, _ind = i*9, ind*9
        if params[i,0]: new_inds.extend(range(_ind,_ind+3))
        _i, _ind = _i+3, _ind+3
        if params[i,1]: new_inds.extend(range(_ind,_ind+4))
        _i, _ind = _i+4, _ind+4
        if params[i,2]: new_inds.extend(range(_ind,_ind+2))
        _i, _ind = _i+2, _ind+2

    return skeletons[:,new_inds]

if __name__ == '__main__':
    skel = load("train")

    print skel.shape

    lens = [s.shape[0] for s in skel]
    print min(lens), max(lens)