from functools import partial
import theano
import theano.tensor as T
import numpy as np
import lasagne as nn
from lasagne.layers import Gate, RecurrentLayer, Conv2DLayer, MaxPool2DLayer, ReshapeLayer, DimshuffleLayer, DropoutLayer, LSTMLayer
from lasagne.layers.dnn import Conv3DDNNLayer, MaxPool3DDNNLayer
# from nervana_theano import layers

import utils
from load import DataLoader
import rnn
import lstm

import sys
sys.setrecursionlimit(1500)


# SHAPE
############
seq_length = 64 # lenght of the sequence
cut_off = 16
batch_size = 8 # micro-batch !!
mini_batch_size = 32
n_classes = 21
im_shape = (64,64)
channels = slice(0,4)

# TRAINING 
############
chunk_size = 128 # samples to keep in memory
n_chunks = -1 # -1 for infinite number of chunks
report_every = 10 # updates between each report print
validate_every = 2000 # updates between each report print
save_every = 2000
p_shared = theano.shared(np.float32(0.))
grad_drop_updates = 1000

# SGD
############
slow_start = dict(epochs=200, lr=1e-6)
learning_rate = 1e-3*mini_batch_size/32.
learning_rate_decay = 3e-5
learning_rate_limit = 1e-6
momentum = 0.9
truncate_steps = -1

# DATA
###########

aug_params = {
    'scale':    (1/1.1, 1.1),
    't_scale':  (1/1.2, 1.2),
    'shear':    (-2, 2),
    'rot':      (-2, 2),
    'trans_x':  (-10, 10),
    'trans_y':  (-5, 5),
}

data_loader = DataLoader( 
    seq_length=seq_length,
    chunk_size=chunk_size,
    batch_size=batch_size,
    cut_off=cut_off,
    channels=channels,
    im_shape=im_shape,
    n_jobs=4,
    n_classes=n_classes,
    aug_params=aug_params,
    block=True,
    n_mem_slots=128,)

n_channels = data_loader.n_channels

# INIT RNN
###########

W_gate = nn.init.Normal(0.01)
b_gate = nn.init.Constant(0.)
default_gate = Gate(W_in=W_gate, W_hid=W_gate, W_cell=W_gate, b=b_gate)
cell_gate = Gate(W_in=W_gate, W_hid=W_gate, W_cell=None, b=b_gate, nonlinearity=nn.nonlinearities.tanh)
activ_gate = nn.nonlinearities.sigmoid
leakiness = 0.3

lstm_layer = partial(lstm.LSTMLayer,
    ingate=default_gate,
    forgetgate=default_gate,
    cell=cell_gate,
    outgate=default_gate,
    nonlinearity=nn.nonlinearities.tanh,
    cell_init=nn.init.Constant(0.),
    hid_init=nn.init.Constant(0.),
    backwards=False,
    learn_init=False,
    peepholes=True,
    gradient_steps=truncate_steps,
    grad_clipping=0,
    unroll_scan=True,
    precompute_input=True,
    mask_input=None,
    only_return_final=False)

rnn_layer = partial(RecurrentLayer,
    W_in_to_hid=nn.init.Orthogonal("relu"),
    W_hid_to_hid=nn.init.Orthogonal("relu"),
    b=nn.init.Constant(0.1),
    nonlinearity=nn.nonlinearities.LeakyRectify(leakiness),
    hid_init=nn.init.Constant(0.),
    backwards=False,
    learn_init=False,
    gradient_steps=truncate_steps,
    unroll_scan=True)

# lstm_layer = partial(lstm.LSTMLayer,
#     W_in_to_ingate=W_gate,
#     W_hid_to_ingate=W_gate,
#     W_cell_to_ingate=W_gate,
#     b_ingate=b_gate,
#     nonlinearity_ingate=activ_gate,
#     W_in_to_forgetgate=W_gate,
#     W_hid_to_forgetgate=W_gate,
#     W_cell_to_forgetgate=W_gate,
#     b_forgetgate=b_gate,
#     nonlinearity_forgetgate=activ_gate,
#     W_in_to_cell=W_gate,
#     W_hid_to_cell=W_gate,
#     b_cell=b_gate,
#     nonlinearity_cell=nn.nonlinearities.tanh,
#     W_in_to_outgate=W_gate,
#     W_hid_to_outgate=W_gate,
#     W_cell_to_outgate=W_gate,
#     b_outgate=b_gate,
#     nonlinearity_outgate=activ_gate,
#     nonlinearity_out=nn.nonlinearities.tanh,
#     cell_init=nn.init.Constant(0.),
#     hid_init=nn.init.Constant(0.),
#     backwards=False,
#     learn_init=False,
#     peepholes=True,
#     gradient_steps=truncate_steps)
#
# rnn_layer = partial(rnn.RecurrentLayer,
#     W_in_to_hid=nn.init.Orthogonal("relu"),
#     W_hid_to_hid=nn.init.Orthogonal("relu"),
#     b=nn.init.Constant(0.1),
#     nonlinearity=nn.nonlinearities.LeakyRectify(leakiness),
#     hid_init=nn.init.Constant(0.),
#     backwards=False,
#     learn_init=False,
#     gradient_steps=truncate_steps)

conv_layer = partial(Conv2DLayer,
    filter_size=(3,3),
    stride=(1, 1),
    pad="same",
    untie_biases=False,
    W=nn.init.Orthogonal("relu"),
    b=nn.init.Constant(0.1),
    nonlinearity=nn.nonlinearities.LeakyRectify(leakiness))

conv3d_layer = partial(Conv3DDNNLayer,
    filter_size=(3,3,3),
    stride=(1, 1,1),
    pad="same",
    untie_biases=False,
    W=nn.init.Orthogonal("relu"),
    b=nn.init.Constant(0.1),
    nonlinearity=nn.nonlinearities.LeakyRectify(leakiness))

conv1d_layer_base = partial(nn.layers.Conv1DLayer,
    filter_size = 3,
    pad="same",
    W=nn.init.Orthogonal("relu"),
    b=nn.init.Constant(0.1), 
    nonlinearity=nn.nonlinearities.LeakyRectify(leakiness))

def conv1d_layer(l, num_filters=16, filter_size=3):
    s = seq_length
    b = batch_size
    bs,c,h,w = l.output_shape

    l = nn.layers.ReshapeLayer(l, (b,s,c,h,w) )

    l = nn.layers.DimshuffleLayer(l, (0,3,4,2,1))
    l = nn.layers.ReshapeLayer(l, (-1, c, s))

    l = conv1d_layer_base(l, num_filters=num_filters, filter_size=filter_size)

    l = nn.layers.ReshapeLayer(l, (b,h,w,num_filters,s))
    l = nn.layers.DimshuffleLayer(l, (0,4,3,1,2))
    l = nn.layers.ReshapeLayer(l, (b*s,num_filters,h,w))

    return l

maxpool = MaxPool2DLayer

dense_layer = partial(nn.layers.DenseLayer,
    W=nn.init.Orthogonal("relu"),
    b=nn.init.Constant(0.1), 
    nonlinearity=nn.nonlinearities.LeakyRectify(leakiness))


def build_model(batch_size=batch_size, seq_length=seq_length):
    l_in = nn.layers.InputLayer(shape=(batch_size, seq_length, n_channels)+im_shape)
    l = l_in

    l = nn.layers.ReshapeLayer(l, (batch_size*seq_length,n_channels)+im_shape)


    n = 16
    l = conv_layer(l, num_filters=n, filter_size=(3,3))
    l = conv1d_layer(l, num_filters=n, filter_size=3)

    l = conv1d_layer(l, num_filters=n, filter_size=3)
    l = conv_layer(l, num_filters=n, filter_size=(3,3))

    l = maxpool(l, pool_size=(2,2))


    n = 32
    l = conv_layer(l, num_filters=n, filter_size=(3,3))
    l = conv1d_layer(l, num_filters=n, filter_size=3)

    l = conv_layer(l, num_filters=n, filter_size=(3,3))
    l = conv1d_layer(l, num_filters=n, filter_size=3)

    l = maxpool(l, pool_size=(2,2))

    n = 64
    l = conv_layer(l, num_filters=n, filter_size=(3,3))
    l = conv1d_layer(l, num_filters=n, filter_size=3)

    l = conv_layer(l, num_filters=n, filter_size=(3,3))
    l = conv1d_layer(l, num_filters=n, filter_size=3)

    l = maxpool(l, pool_size=(2,2))

    n = 128
    l = conv_layer(l, num_filters=n, filter_size=(3,3))
    l = conv1d_layer(l, num_filters=n, filter_size=3)

    l = conv_layer(l, num_filters=n, filter_size=(3,3))
    l = conv1d_layer(l, num_filters=n, filter_size=3)

    l = maxpool(l, pool_size=(2,2))

    l = nn.layers.ReshapeLayer(l, (batch_size, seq_length, -1))

    l = nn.layers.DropoutLayer(l, p=p_shared)

    n_states = 512
    l_fwd = lstm_layer(l, n_states)
    l_bck = lstm_layer(l, n_states, backwards=True)
    l = nn.layers.ElemwiseSumLayer([l_fwd, l_bck])

    l = nn.layers.DropoutLayer(l, p=p_shared)

    l = nn.layers.ReshapeLayer(l, (batch_size*seq_length, n_states))

    l_out = nn.layers.DenseLayer(l, num_units=n_classes, 
                            nonlinearity=nn.nonlinearities.softmax)
    l_out_reshape = nn.layers.ReshapeLayer(l_out, (batch_size, seq_length, n_classes))

    return utils.struct(
        input=l_in, 
        out=l_out, 
        out_reshape=l_out_reshape)


# def build_updates_with_micro(loss, all_params, learning_rate,  beta1=0.1, beta2=0.001,
#                     epsilon=1e-8):
#     """ Adam update rule by Kingma and Ba, ICLR 2015. """
#     all_grads = theano.grad(loss, all_params)
#     updates, micro_updates = [], []
#
#     t = theano.shared(1) # timestep, for bias correction
#     for param_i, grad_i in zip(all_params, all_grads):
#         zeros = np.zeros(param_i.get_value(borrow=True).shape, dtype=theano.config.floatX)
#         mparam_i = theano.shared(zeros) # 1st moment
#         vparam_i = theano.shared(zeros.copy()) # 2nd moment
#         sum_grad_i = theano.shared(zeros.copy())
#
#         micro_updates.append((sum_grad_i, sum_grad_i+grad_i))
#
#         grad = sum_grad_i / np.float32(mini_batch_size//batch_size)
#         m = beta1 * grad + (1 - beta1) * mparam_i # new value for 1st moment estimate
#         v = beta2 * T.sqr(grad) + (1 - beta2) * vparam_i # new value for 2nd moment estimate
#
#         m_unbiased = m / (1 - (1 - beta1) ** t.astype(theano.config.floatX))
#         v_unbiased = v / (1 - (1 - beta2) ** t.astype(theano.config.floatX))
#         w = param_i - learning_rate * m_unbiased / (T.sqrt(v_unbiased) + epsilon) # new parameter values
#
#         updates.append((mparam_i, m))
#         updates.append((vparam_i, v))
#         updates.append((param_i, w))
#         updates.append((sum_grad_i, zeros.copy()))
#     updates.append((learning_rate, learning_rate * (1-learning_rate_decay)))
#     updates.append((t, t + 1))
#
#     return updates, micro_updates


