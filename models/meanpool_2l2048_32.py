from functools import partial
import theano
import theano.tensor as T
import numpy as np
import lasagne as nn
from lasagne.layers import dnn


import rnn
import lstm
import utils
from load_featpool import DataLoader


# SHAPE
############
seq_length = 32 # lenght of the sequence
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

# SGD
############
slow_start = dict(epochs=10, lr=1e-4)
learning_rate = 1e-3
learning_rate_decay = 1e-4
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
    data_path="rgbd_i64_train",
    labels_path="labels_raw.npy",
    n_jobs=4,
    n_classes=n_classes,
    aug_params=aug_params,
    block=True,
    n_mem_slots=128)

n_channels = data_loader.n_channels

# transfer = "metadata/pret_2l_2048-geit-20150514-203238.pkl"

# INIT RNN
###########

W_gate = nn.init.Normal(0.01)
b_gate = nn.init.Constant(0.)
activ_gate = nn.nonlinearities.sigmoid
leakiness = 0.3

lstm_layer = partial(lstm.LSTMLayer, 
    W_in_to_ingate=W_gate,
    W_hid_to_ingate=W_gate,
    W_cell_to_ingate=W_gate,
    b_ingate=b_gate,
    nonlinearity_ingate=activ_gate,
    W_in_to_forgetgate=W_gate,
    W_hid_to_forgetgate=W_gate,
    W_cell_to_forgetgate=W_gate,
    b_forgetgate=b_gate,
    nonlinearity_forgetgate=activ_gate,
    W_in_to_cell=W_gate,
    W_hid_to_cell=W_gate,
    b_cell=b_gate,
    nonlinearity_cell=nn.nonlinearities.tanh,
    W_in_to_outgate=W_gate,
    W_hid_to_outgate=W_gate,
    W_cell_to_outgate=W_gate,
    b_outgate=b_gate,
    nonlinearity_outgate=activ_gate,
    nonlinearity_out=nn.nonlinearities.tanh,
    cell_init=nn.init.Constant(0.),
    hid_init=nn.init.Constant(0.),
    backwards=False,
    learn_init=False,
    peepholes=True,
    gradient_steps=truncate_steps)

rnn_layer = partial(rnn.RecurrentLayer, 
    W_in_to_hid=nn.init.Orthogonal("relu"),
    W_hid_to_hid=nn.init.Orthogonal("relu"),
    b=nn.init.Constant(0.1),
    nonlinearity=nn.nonlinearities.LeakyRectify(leakiness),
    hid_init=nn.init.Constant(0.), 
    backwards=False,
    learn_init=False, 
    gradient_steps=truncate_steps)

conv_layer = partial(dnn.Conv2DDNNLayer,
    strides=(1, 1),
    border_mode="same", 
    untie_biases=False, 
    W=nn.init.Orthogonal("relu"),
    b=nn.init.Constant(0.1), 
    nonlinearity=nn.nonlinearities.LeakyRectify(leakiness))

max_pooling = dnn.MaxPool2DDNNLayer

dense_layer = partial(nn.layers.DenseLayer,
    W=nn.init.Orthogonal("relu"),
    b=nn.init.Constant(0.1), 
    nonlinearity=nn.nonlinearities.LeakyRectify(leakiness))

def build_model(batch_size=batch_size, seq_length=seq_length):
    l_in = nn.layers.InputLayer(shape=(batch_size, seq_length, n_channels)+im_shape)
    l = l_in

    l = nn.layers.ReshapeLayer(l, (batch_size*seq_length,n_channels)+im_shape)

    l = conv_layer(l, num_filters=16, filter_size=(3,3))
    l = conv_layer(l, num_filters=16, filter_size=(3,3))
    l = max_pooling(l, ds=(2,2))
    l = conv_layer(l, num_filters=32, filter_size=(3,3))
    l = conv_layer(l, num_filters=32, filter_size=(3,3))
    l = max_pooling(l, ds=(2,2))
    l = conv_layer(l, num_filters=64, filter_size=(3,3))
    l = conv_layer(l, num_filters=64, filter_size=(3,3))
    l = max_pooling(l, ds=(2,2))
    l = conv_layer(l, num_filters=128, filter_size=(3,3))
    l = conv_layer(l, num_filters=128, filter_size=(3,3))
    l = max_pooling(l, ds=(2,2))

    l = nn.layers.ReshapeLayer(l, (batch_size, seq_length, -1))

    l = nn.layers.FeaturePoolLayer(l, ds=seq_length,  pool_function=T.mean, axis=1)

    l = nn.layers.DropoutLayer(l, p=0.5)

    l = dense_layer(l, num_units=2048)

    l = nn.layers.DropoutLayer(l, p=0.5)

    l = dense_layer(l, num_units=2048)

    l = nn.layers.DropoutLayer(l, p=0.5)

    l_out = nn.layers.DenseLayer(l, num_units=n_classes, 
                            nonlinearity=nn.nonlinearities.softmax)

    return utils.struct(
        input=l_in, 
        out=l_out)


def build_updates_with_micro(loss, all_params, learning_rate,  beta1=0.1, beta2=0.001, 
                    epsilon=1e-8):
    """ Adam update rule by Kingma and Ba, ICLR 2015. """
    all_grads = theano.grad(loss, all_params)
    updates, micro_updates = [], []
    
    t = theano.shared(1) # timestep, for bias correction
    for param_i, grad_i in zip(all_params, all_grads):
        zeros = np.zeros(param_i.get_value(borrow=True).shape, dtype=theano.config.floatX)
        mparam_i = theano.shared(zeros) # 1st moment
        vparam_i = theano.shared(zeros.copy()) # 2nd moment
        sum_grad_i = theano.shared(zeros.copy())

        micro_updates.append((sum_grad_i, sum_grad_i+grad_i))

        grad = sum_grad_i / np.float32(mini_batch_size//batch_size)
        m = beta1 * grad + (1 - beta1) * mparam_i # new value for 1st moment estimate
        v = beta2 * T.sqr(grad) + (1 - beta2) * vparam_i # new value for 2nd moment estimate
        
        m_unbiased = m / (1 - (1 - beta1) ** t.astype(theano.config.floatX))
        v_unbiased = v / (1 - (1 - beta2) ** t.astype(theano.config.floatX))
        w = param_i - learning_rate * m_unbiased / (T.sqrt(v_unbiased) + epsilon) # new parameter values

        updates.append((mparam_i, m))
        updates.append((vparam_i, v))
        updates.append((param_i, w))
        updates.append((sum_grad_i, zeros.copy()))
    updates.append((learning_rate, learning_rate * (1-learning_rate_decay)))
    updates.append((t, t + 1))

    return updates, micro_updates

