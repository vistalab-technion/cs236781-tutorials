'''
This is a PyTorch implementation of 'Domain-Adversarial Training of 
Neural Networks' by Yaroslav Ganin et al. (2016).

The DANN model uses the adversarial learning paradigm to force a 
classifier to only learn features that exist in both domains. This
enables a classifier trained on the source domain to generalize to 
the target domain.

This is achieved with the 'gradient reversal' layer to form
a domain invariant feature embedding which can be used with the 
same CNN.

This example uses MNIST as source dataset and USPS or MNIST-M 
as target datasets.

Author: Daniel Bartolomé Rojas (d.bartolome.r@gmail.com)
'''
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable, Function

from evaluation import eval_clf
from models import DANN
from data_handling import batch_generator, data_loader
   
# parameters
learning_rate = 1e-2          # learning rate
num_epochs = 30               # number of epochs to train models
num_src_epochs = 15           # number of epochs to pre-train source only model
batch_size = 50               # size of image sample per epoch 
source_data = 'mnist'         # mnist / mnist_m / usps
target_data = 'usps'          # mnist / mnist_m / usps
input_ch = 1                  # 1 for usps, 3 for mnistm

# instantiate the models
f_ext, d_clf, c_clf = DANN(input_ch)

# set loss functions
d_crit = nn.BCELoss()         # binary crossentropy
c_crit = nn.MSELoss()         # mean squared error

# set optimizers
d_optimizer = optim.SGD(d_clf.parameters(), lr=learning_rate, momentum=0.9)
c_optimizer = optim.SGD(c_clf.parameters(), lr=learning_rate, momentum=0.9)
f_optimizer = optim.SGD(f_ext.parameters(), lr=learning_rate, momentum=0.9)

# load source domain dataset
(Xs_train, ys_train), (Xs_test, ys_test) = data_loader(source_data)

# same lengths as USPS dataset
(Xs_train, ys_train), (Xs_test, ys_test) = (Xs_train[:7291], ys_train[:7291]), (Xs_test[:2007], ys_test[:2007])
# # concat MNIST images as channels to match number of MNIST-M channels
# Xs_train = np.concatenate([Xs_train, Xs_train, Xs_train], axis=1)
# Xs_test = np.concatenate([Xs_test, Xs_test, Xs_test], axis=1)

# load target domain dataset
(Xt_train, yt_train), (Xt_test, yt_test) = data_loader(target_data)

# init necessary objects
num_steps = num_epochs * (Xs_train.shape[0] / batch_size)
yd = Variable(torch.from_numpy(np.hstack([np.repeat(1, int(batch_size / 2)), np.repeat(0, int(batch_size / 2))]).reshape(50, 1)))
j = 0

# pre-train source only model
print('\nPre-training source-only model..')
for i in range(num_src_epochs):
    source_gen = batch_generator(int(batch_size / 2), Xs_train, ys_train)
    
    # iterate over batches
    for (xs, ys, _) in source_gen:
        
        # exit when batch size mismatch
        if len(xs) != batch_size / 2:
            continue
        
        # reset gradients
        f_ext.zero_grad()
        c_clf.zero_grad()
        
        # calculate class_classifier predictions
        c_out = c_clf(f_ext(xs).view(int(batch_size / 2), -1))
        
        # optimize feature_extractor and class_classifier with output
        f_c_loss = c_crit(c_out, ys.float())
        f_c_loss.backward(retain_graph = True)
        c_optimizer.step()
        f_optimizer.step()
        
        # print batch statistics
        print('\rEpoch {}       - loss: {}'.format(i+1, format(f_c_loss.data[0], '.4f')), end='')
    
    # print epoch statistics    
    s_acc = eval_clf(c_clf, f_ext, Xs_test, ys_test, 1000)
    print(' - val_acc: {}'.format(format(s_acc, '.4f')))

# print target accuracy with source model
t_acc = eval_clf(c_clf, f_ext, Xt_test, yt_test, 1000)
print('\nTarget accuracy with source model: {}\n'.format(format(t_acc, '.4f')))

# train DANN model
print('Training DANN model..')
for i in range(num_epochs):
    source_gen = batch_generator(int(batch_size / 2), Xs_train, ys_train)
    target_gen = batch_generator(int(batch_size / 2), Xt_train, None)

    # iterate over batches
    for (xs, ys, _) in source_gen:
        
        # update lambda and learning rate as suggested in the paper
        p = float(j) / num_steps
        lambd = round(2. / (1. + np.exp(-10. * p)) - 1, 3)
        lr = 0.01 / (1. + 10 * p)**0.75
        d_clf.set_lambda(lambd)
        d_optimizer.lr = lr
        c_optimizer.lr = lr
        f_optimizer.lr = lr
        j += 1
        
        # get next target batch
        xt, _ = next(target_gen)

        # exit when batch size mismatch
        if len(xs) + len(xt) != batch_size:
            continue
        
        # concatenate source and target batch
        x = torch.cat([xs, xt], 0)
        
        # 1) train feature_extractor and class_classifier on source batch
        # reset gradients
        f_ext.zero_grad()
        c_clf.zero_grad()
        
        # calculate class_classifier predictions on batch xs
        c_out = c_clf(f_ext(xs).view(int(batch_size / 2), -1))
        
        # optimize feature_extractor and class_classifier on output
        f_c_loss = c_crit(c_out, ys.float())
        f_c_loss.backward(retain_graph = True)
        c_optimizer.step()
        f_optimizer.step()
        
        # 2) train feature_extractor and domain_classifier on full batch x
        # reset gradients
        f_ext.zero_grad()
        d_clf.zero_grad()
        
        # calculate domain_classifier predictions on batch x
        d_out = d_clf(f_ext(x).view(batch_size, -1))
        
        # use normal gradients to optimize domain_classifier
        f_d_loss = d_crit(d_out, yd.float())
        f_d_loss.backward(retain_graph = True)
        d_optimizer.step()
        f_optimizer.step()
        
        # print batch statistics
        print('\rEpoch         - d_loss: {} - c_loss: {}'.format(format(f_d_loss.data[0], '.4f'),
                            format(f_c_loss.data[0], '.4f')), end='')           
    
    # print epoch statistics
    t_acc = eval_clf(c_clf, f_ext, Xt_test, yt_test, 1000)
    s_acc = eval_clf(c_clf, f_ext, Xs_test, ys_test, 1000)
    print(' - target_acc: {} - source_acc: {}'.format(format(t_acc, '.4f'), format(s_acc, '.4f')))
