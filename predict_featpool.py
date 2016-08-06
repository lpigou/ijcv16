import sys

import numpy as np
import theano
import theano.tensor as T
import time
import os
import string
import importlib
from glob import glob
import scipy.stats

import utils
import lasagne as nn
import split
import load


batch_size = 10
cut_off = 16
floatX = theano.config.floatX


if not (2 <= len(sys.argv) <= 4):
    sys.exit("Usage: %s <metadata_path> [subset=test] [batch_size=10]"%os.path.basename(__file__))

metadata_path = sys.argv[1]
if len(sys.argv) >= 3:
    subset = sys.argv[2]
    if len(sys.argv) >= 4:
        batch_size = int(sys.argv[3])
else:
    print "no subset specified, predicting for subset 'test'"
    subset = "test"

print "Load parameters"
metadata = np.load(metadata_path)


config_name = metadata['configuration']

cfg = importlib.import_module("models.%s" % config_name)
cfg.batch_size = batch_size
cfg.cut_off = cut_off

filename = os.path.splitext(os.path.basename(metadata_path))[0]
target_path = "predictions/%s--%s/" % (subset, filename)
csv_path = target_path + "csv/"
if not os.path.exists("predictions/"): os.mkdir("predictions")
if not os.path.exists(target_path): os.mkdir(target_path)
if not os.path.exists(csv_path): os.mkdir(csv_path)
target_path += "preds.pkl"
print target_path

if not os.path.exists(target_path):

    print "Build model"
    model = cfg.build_model(cfg.batch_size)
    l_in, l_out = model.input, model.out

    all_layers = nn.layers.get_all_layers(l_out)
    num_params = nn.layers.count_params(l_out)
    print "  number of parameters: %d" % num_params
    print "  layer output shapes:"
    for layer in all_layers:
        name = string.ljust(layer.__class__.__name__, 32)
        print "    %s %s" % (name, layer.get_output_shape(),)

    x_shared = nn.utils.shared_empty(dim=len(l_in.get_output_shape())) 
    m_shared = nn.utils.shared_empty(dim=2)

    print "Compiling"
    idx_ofs = T.lscalar('idx_ofs') # offset in sequence
    ofs_slice = slice(idx_ofs, idx_ofs+cfg.seq_length)

    mask_batch = T.matrix("mask_batch")
    output_pred = model.out.get_output(deterministic=True, mask=mask_batch)

    compute_output = theano.function([idx_ofs], output_pred,
        on_unused_input="ignore",
        givens={ model.input.input_var:  x_shared[:, ofs_slice],
                 mask_batch:             m_shared[:, ofs_slice] })
    print "\t...done"
    param_values = metadata['param_values']
    nn.layers.set_all_param_values(l_out, param_values)



if subset == "test":

    if not os.path.exists(target_path):
        n_samples = len(split.test_idxs) 
        # n_samples = 10
        data_path = cfg.data_loader.data_path.replace("train", "test")
        n_batches = n_samples // cfg.batch_size

        test_idxs = split.test_idxs[:n_samples]

        zeros_x = np.zeros((cfg.seq_length//2, cfg.n_channels)+cfg.im_shape,dtype="float32")

        def load_batch(b):
            global x_shared, m_shared

            smpls = []
            for idx in range(cfg.batch_size):
                path = data_path+str(split.test_idxs[cfg.batch_size*b+idx]).zfill(4)+".npy"
                smpl = np.load(path)
                smpl = smpl[:,cfg.channels] / 255.
                smpl = np.concatenate((zeros_x[:-1], smpl, zeros_x), axis=0)
                # print smpl.shape

                # resize if necessary
                if cfg.im_shape != smpl.shape[-2:]:
                    new_smpl = np.empty(smpl.shape[:-2]+cfg.im_shape, dtype=smpl.dtype)
                    for i in range(smpl.shape[0]):
                        for j in range(smpl.shape[1]):
                            new_smpl[i,j] = load.resize_im(smpl[i,j], cfg.im_shape)
                    smpl = new_smpl

                smpls.append(smpl.copy())

            max_len = np.max([len(s) for s in smpls])
            max_len += cfg.seq_length - 1
            valid_len = int(np.ceil(max_len / float(cfg.seq_length) )) * cfg.seq_length
            m = np.zeros((cfg.batch_size, valid_len), dtype="uint8")
            x = np.zeros((cfg.batch_size, valid_len, cfg.n_channels)+cfg.im_shape, dtype=floatX)

            for idx, smpl in enumerate(smpls):
                x[idx, :len(smpl)] = smpl
                m[idx, :len(smpl)] = 1

            x_shared.set_value(x, borrow=True)
            m_shared.set_value(m, borrow=True)

        t0 = time.time()

        print "Predicting"

        preds = []
        for b in range(n_batches):
            load_batch(b)
            val_seq_len =  x_shared.get_value(borrow=True).shape[1]
            n_offsets = val_seq_len - cfg.seq_length
            pred_batch = []
            for ofs in range(n_offsets):
                pred = compute_output(ofs)
                pred_batch.append(pred)
            pred_batch = np.array(pred_batch).swapaxes(0,1)

            msk = m_shared.get_value(borrow=True)
            new_preds_batch = []
            for i in range(len(pred_batch)):
                l = len(msk[i][msk[i]==0.])
                new_preds_batch.append(pred_batch[i][:-l])
            preds.extend(new_preds_batch)
            print "%i%%"%int(np.round((b+1)/float(n_batches)*100.))

            utils.save_pkl(preds, target_path)

    else: 
        preds = utils.load_pkl(target_path)
        assert len(preds) == len(split.test_idxs)

    print "Preparing csv files"
    pred_files = glob(csv_path+"Sample*")
    if len(pred_files) != len(split.test_idxs):
        for i, pred in enumerate(preds):

            pred = np.argmax(pred, axis=-1)

            # window = 10
            # new_pred = pred.copy()
            # for j in range(len(pred)-window):
            #     new_pred[j+window//2] = scipy.stats.mode(pred[j:j+window])[0][0]
            # pred = new_pred

            s = ""
            start = 0
            prev = 0
            for j, p in enumerate(pred):
                if j != 0:
                    if p != prev:
                        if prev != 0: s+= "%i,%i,%i\n"%(prev, start+1, j)
                        start = j
                prev = p

            file = open(csv_path+"Sample"+str(split.test_idxs[i]).zfill(4)+"_prediction.csv", "w")
            file.write(s[:-1])
            file.close()
            print "%i%%"%int(np.round((i+1)/float(len(preds))*100.))
    from sklearn import metrics
    labels = np.load("data/labels_raw_test.npy")
    # labelsn = labels.copy()
    for i, lbl in enumerate(labels):
        labels[i] = lbl[:-1]
        # labels[i] = lbl[1:]

    labels[:] = map(lambda y: utils.one_hot(y,21), labels)



    # print labels[0].shape, preds[0].shape
    roc, prec, rec, acc = [], [],[], []
    l, p = np.vstack(labels), np.vstack(preds)
    # ln = np.vstack(labelsn)

    # print np.mean(l==p)
    y_pred = np.argmax(p,1)
    y_true = np.argmax(l,1)
    # print l.shape
    # print y.shape
    acc = np.mean(y_true==y_pred)
    rec = metrics.recall_score(y_true, y_pred)
    prec = metrics.precision_score(y_true, y_pred)
    # print "acc sklearn", metrics.accuracy_score(y_true, y_pred)
    # print acc, rec, prec

    tps = np.zeros((20,))
    rec = np.empty((20,))
    prec = np.empty((20,))
    for i, p in enumerate(y_pred):
        if p == 0: continue
        if p == y_true[i]:
            tps[p-1] += 1

    for i in range(20):
        # print (y_true==i+1).shape
        rec[i] = tps[i] / float((y_true==i+1).sum())
        prec[i] = tps[i] / float((y_pred==i+1).sum())

    print "rec", rec
    print rec.mean()
    print "prec", prec
    print prec.mean()

    votes = []
    prev,total, correct = 0,0,0
    for i, p in enumerate(y_true):
        if prev==0 and p == 0:  continue
        if p == 0: 
            vote = int(scipy.stats.mode(votes)[0][0])
            if vote == prev: correct += 1
            total += 1
            votes = []
        else:
            votes.append(y_pred[i])
        prev = p
    assert prev==0
    print "acc", correct/float(total), correct, total
    
    print "Evaluating"
    score = utils.jaccard_index_chalearn(csv_path, "gt/")
    print score

elif subset == "valid":



    ld = cfg.data_loader

    # n_samples=len(split.valid_idxs)
    n_samples=100
    labels = np.load(ld.labels_path)
    labels[:] = map(lambda y: utils.one_hot(y,ld.n_classes), labels)
    labels = labels[-len(split.valid_idxs):]

    # inds = np.random.permutation(len(split.valid_idxs))[:n_samples]
    # labels = labels[inds]
    # valid_idxs = split.valid_idxs[inds]

    labels = labels[:n_samples]
    valid_idxs = split.valid_idxs[:n_samples]

    n_batches = n_samples // cfg.batch_size

    if not os.path.exists(target_path):


        def load_batch(b):
            global x_shared, m_shared

            max_len = np.max([len(i) for i in labels[b*cfg.batch_size:(b+1)*cfg.batch_size]])
            eff_len = cfg.seq_length - 2*cfg.cut_off
            valid_len = cfg.seq_length + int(np.ceil((max_len-cfg.seq_length) / \
                float(eff_len))*eff_len)
            m = np.zeros((cfg.batch_size, valid_len), dtype="uint8")
            x = np.zeros((cfg.batch_size, valid_len, ld.n_channels)+cfg.im_shape, dtype=floatX)

            for idx in range(cfg.batch_size):

                path = ld.data_path+str(valid_idxs[cfg.batch_size*b+idx]).zfill(4)+".npy"
                smpl = np.load(path)
                smpl = smpl[:,cfg.channels] / 255.

                # resize if necessary
                if cfg.im_shape != smpl.shape[-2:]:
                    new_smpl = np.empty(smpl.shape[:-2]+cfg.im_shape, dtype=smpl.dtype)
                    for i in range(smpl.shape[0]):
                        for j in range(smpl.shape[1]):
                            new_smpl[i,j] = ld.resize_im(smpl[i,j], cfg.im_shape)
                    smpl = new_smpl

                x[idx, :len(smpl)] = smpl
                m[idx, :len(smpl)] = 1

            x_shared.set_value(x, borrow=True)
            m_shared.set_value(m, borrow=True)

        t0 = time.time()

        print "Predicting"

        def cut_off_pred(pred, ofs_idx, n_offsets):
            if ofs_idx == 0: return pred[:,:-cfg.cut_off]
            elif ofs_idx == n_offsets-1: return pred[:,cfg.cut_off:]
            else: return pred[:,cfg.cut_off:-cfg.cut_off]

        preds = []
        # valid_gen = cfg.data_loader.create_valid_generator(n_samples=n_samples)
        for b in range(n_batches):
            load_batch(b)
            val_seq_len =  x_shared.get_value(borrow=True).shape[1]
            n_offsets = (val_seq_len - cfg.seq_length) // eff_len + 1
            pred_batch = []
            for ofs in range(n_offsets):
                pred = compute_output(ofs)
                pred = cut_off_pred(pred, ofs, n_offsets)            
                pred_batch.append(pred)
            pred_batch = np.hstack(pred_batch)

            msk = m_shared.get_value(borrow=True)
            new_preds_batch = []
            for i in range(len(pred_batch)):
                l = len(msk[i][msk[i]==1.])
                new_preds_batch.append(pred_batch[i][:l])
            preds.extend(new_preds_batch)
            print "%i%%"%int(np.round((b+1)/float(n_batches)*100.))

        utils.save_pkl(preds, target_path)
    else:
        preds = utils.load_pkl(target_path)
        assert len(preds) == len(split.valid_idxs)

    print "Calculating ji"
    overlaps = [utils.jaccard_index(preds[i], labels[i]) for i in range(n_samples)]
    # for i,o in enumerate(overlaps): print "%i %.3f"%(split.valid_idxs[i], o)
    overlap = np.mean(overlaps)
    print overlap, time.time()-t0