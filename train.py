################################################################################
#                                                                          INIT
################################################################################
print"v1"
import numpy as np
import theano
import theano.tensor as T
import lasagne as nn
from time import time, strftime, localtime
from subprocess import Popen
import sys
import os
import importlib
import warnings
import string
import cPickle
import platform

import utils
import split



warnings.filterwarnings('ignore', '.*topo.*')

if len(sys.argv) < 2:
    print "Usage: %s <config_path>"%os.path.basename(__file__)
    cfg_path = "models/debug_local.py"
else: cfg_path = sys.argv[1]

cfg_name = cfg_path.split("/")[-1][:-3]
print "Model:", cfg_name
cfg = importlib.import_module("models.%s" % cfg_name)

expid = "%s-%s-%s" % (cfg_name, platform.node(), strftime("%Y%m%d-%H%M%S", localtime()))
print "expid:", expid


################################################################################
#                                                               BUILD & COMPILE
################################################################################
print "Building"
model = cfg.build_model()

all_layers = nn.layers.get_all_layers(model.out)
num_params = nn.layers.count_params(model.out)
print "Number of parameters: %d" % num_params
print "Layer output shapes:"
for layer in all_layers:
    name = string.ljust(layer.__class__.__name__, 32)
    print "  %s %s" % (name, layer.output_shape,)

x = nn.utils.shared_empty(dim=len(model.input.output_shape))
y = nn.utils.shared_empty(dim=len(model.out_reshape.output_shape))
idx = T.lscalar('idx')
targets_batch = y[idx*cfg.batch_size:(idx+1)*cfg.batch_size]
targets_batch = targets_batch.reshape(
                    (cfg.batch_size*cfg.seq_length, cfg.n_classes))


# obj = nn.objectives.Objective(model.out, loss_function=log_loss)

def log_loss(y, t, eps=10e-8):
    y = T.clip(y, eps, 1 - eps)
    return -T.sum(t * T.log(y)) / y.shape[0].astype(utils.floatX)

#train_loss = nn.objectives.categorical_crossentropy(T.clip( nn.layers.get_output(model.out), 1e-15, 1 - 1e-15),
#                                                   targets_batch)

train_loss = log_loss(nn.layers.get_output(model.out), targets_batch)
train_loss = train_loss.mean();

givens = {
    # obj.target_var: targets_batch,
    model.input.input_var: x[idx*cfg.batch_size:(idx+1)*cfg.batch_size]
}

all_params = nn.layers.get_all_params(model.out, trainable=True)



learning_rate = theano.shared(utils.cast_floatX(cfg.learning_rate))

using_micro = False
if hasattr(cfg, 'build_updates_with_micro'):
    using_micro = True
    updates, micro_updates = cfg.build_updates_with_micro(train_loss, all_params, learning_rate)
else:
    if hasattr(cfg, 'build_updates'):
        updates = cfg.build_updates(train_loss, all_params, learning_rate)
    else:
        updates = nn.updates.adam( train_loss, all_params, learning_rate)


mask = nn.utils.shared_empty(dim=2)
mask_batch = T.matrix("mask_batch")
# l_mask = nn.layers.InputLayer(shape=(cfg.batch_size, cfg.seq_length), input_var=mask_batch)

batch_slice = slice(idx*cfg.batch_size, (idx+1)*cfg.batch_size)
idx_ofs = T.lscalar('idx_ofs') # offset in sequence
eff_len = cfg.seq_length - 2*cfg.cut_off # effective length
ofs_slice = slice(idx_ofs*eff_len, idx_ofs*eff_len+cfg.seq_length)

# model_det = cfg.build_model(mask=l_mask)
# nn.layers.helper.set_all_param_values(model_det.out, all_params, trainable=True)
output_pred = nn.layers.get_output(model.out_reshape, deterministic=True, mask=mask_batch)

print "Compiling"
if using_micro:
    apply_updates = theano.function([], updates=updates, on_unused_input='ignore')
    iter_train = theano.function([idx], train_loss, givens=givens, updates=micro_updates)
else:
    iter_train = theano.function([idx], train_loss, givens=givens, updates=updates)


compute_output = theano.function([idx, idx_ofs], output_pred, 
    on_unused_input="ignore",
    givens={ model.input.input_var:  x[batch_slice, ofs_slice],
             mask_batch:             mask[batch_slice, ofs_slice] })

################################################################################
#                                                                         TRAIN
################################################################################

n_batches = cfg.chunk_size // cfg.batch_size
n_valid_samples = len(split.valid_idxs)//cfg.batch_size*cfg.batch_size
n_chunks = np.uint32(cfg.n_chunks) # this handles the cfg.n_chunks = -1
n_updates = 0

print "Loading"
cfg.data_loader.initialize(x, y, mask)
train_gen = cfg.data_loader.create_train_generator()
labels_valid = cfg.data_loader.labels_shm.get()[split.valid_idxs-1]

grad_drop = False
if hasattr(cfg, "grad_drop_updates"):
    print "grad_drop"
    grad_drop = True

slow_start = False
if hasattr(cfg, "slow_start"):
    print "slow_start"
    slow_start = True
    slow_v = utils.cast_floatX(cfg.slow_start["lr"])
    learning_rate.set_value(np.float32(slow_v))

def train(e):
    global n_updates
    t0 = time()
    train_gen.next()
    data_overhead = (time()-t0)*1000

    losses = []
    for b in xrange(n_batches):
        loss = iter_train(b)
        if np.isnan(loss): raise RuntimeError("NaN DETECTED.")
        losses.append(loss)
        # if e==0: print "Training loss: %.3f"%loss
        if using_micro:
            if not (b+1)*cfg.batch_size % cfg.mini_batch_size:
                apply_updates()
                n_updates += 1
        else:
            n_updates += 1

        if grad_drop:
            if cfg.grad_drop_updates >= n_updates:
                cfg.p_shared.set_value(np.float32(n_updates / float(cfg.grad_drop_updates) * 0.5))
        if slow_start:
            if cfg.slow_start["epochs"] >= n_updates:
                learning_rate.set_value(
                    np.float32(slow_v + n_updates / float(cfg.slow_start["epochs"]) * (cfg.learning_rate - slow_v)))

    return np.mean(losses), data_overhead

def cut_off_pred(pred, ofs_idx, n_offsets):
    if ofs_idx == 0: return pred[:,:-cfg.cut_off]
    elif ofs_idx == n_offsets-1: return pred[:,cfg.cut_off:]
    else: return pred[:,cfg.cut_off:-cfg.cut_off]

def validate():
    preds = []
    valid_gen = cfg.data_loader.create_valid_generator(n_samples=n_valid_samples)
    for v in valid_gen:
        val_seq_len =  x.get_value(borrow=True).shape[1]
        n_offsets = (val_seq_len - cfg.seq_length) // eff_len + 1
        pred_batch = []
        for ofs in range(n_offsets):
            pred = compute_output(0, ofs)
            pred = cut_off_pred(pred, ofs, n_offsets)            
            pred_batch.append(pred)
        pred_batch = np.hstack(pred_batch)

        msk = mask.get_value(borrow=True)
        new_preds_batch = []
        for i in range(len(pred_batch)):
            l = len(msk[i][msk[i]==1.])
            new_preds_batch.append(pred_batch[i][:l])
        preds.extend(new_preds_batch)

    overlap = np.mean([utils.jaccard_index(preds[i], labels_valid[i]) 
                        for i in range(n_valid_samples)])
    return overlap

train_losses, valid_losses = [],[]
import shutil
def save():
    global mv_process

    metadata_tmp_path = "/var/tmp/%s.pkl"%expid
    if not os.path.exists("metadata/"): os.mkdir("metadata")
    metadata_target_path = "metadata/%s.pkl"%expid
    print "\tSaving metadata, parameters in ", metadata_target_path

    with open(metadata_tmp_path, 'w') as f:
        cPickle.dump({
            'configuration': cfg_name,
            'experiment_id': expid,
            'losses_train': train_losses,
            'losses_eval_valid': valid_losses,
            'param_values': nn.layers.get_all_param_values(model.out), 
            'learning_rate': learning_rate.get_value(),
            'n_updates': n_updates
        }, f, cPickle.HIGHEST_PROTOCOL)

    try:
        shutil.move(metadata_tmp_path, metadata_target_path)
    except Exception as e:
        print e

print "Training"
valid_loss = 0
data_overhead = []
best_valid_loss = 0
start_time = time()
n_upd_prev_val, n_upd_prev_rep = 0, 0

def secondsToStr(t):return "%dh%02dm" % reduce(lambda ll,b : divmod(ll[0],b) + ll[1:],[(t*1000,),1000,60,60])[:2]

#valid_loss = validate()

for e in xrange(n_chunks): 
    t0 = time()
    train_loss, data_overhead = train(e)
    train_time = (time()-t0)*1000. / cfg.chunk_size
    train_losses.append(train_loss)

    if n_updates - n_upd_prev_val > cfg.validate_every: 
        n_upd_prev_val = n_updates
        print "\tValidating"
        valid_loss = validate()
        valid_losses.append(valid_loss)
        if valid_loss > best_valid_loss: 
            best_valid_loss = valid_loss 
            save()
        print "\tValidation Jaccard Index:", valid_loss
        print "\tBest validation Jaccard Index:", best_valid_loss
        print "\tTime since start: %s" % secondsToStr(time()-start_time)

    if n_updates - n_upd_prev_rep > cfg.report_every:
        n_upd_prev_rep = n_updates
        lr = learning_rate.get_value()
        
        if grad_drop:
            print "%i time %.2ems train %.3f lr %.2e do %.2fms dr %.2f" \
                % (n_updates, train_time, np.mean(train_losses[-cfg.report_every:]), lr, np.mean(data_overhead), cfg.p_shared.get_value())
        else:
            print "%i time %.2ems train %.3f lr %.2e do %.2fms" \
                % (n_updates, train_time, np.mean(train_losses[-cfg.report_every:]), lr, np.mean(data_overhead))
        data_overhead = []
