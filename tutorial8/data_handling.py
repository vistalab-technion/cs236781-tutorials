import torch
from torch.autograd import Variable

import numpy as np
from sklearn.model_selection import KFold

from data_sources import load_mnistm, load_mnist, load_usps

# generates torch tensor batches from input data
def batch_generator(batch_size, data, labels):
    size = data.shape[0]
    idx_array = np.arange(size)
    n_batch = int(np.ceil(size / float(batch_size)))
    batches = [(int(i * batch_size), int(min(size, (i + 1) * batch_size))) for i in range(n_batch)]
    for batch_index, (start, end) in enumerate(batches):
        print('\rBatch {}/{}'.format(batch_index+1, n_batch), end='')
        batch_ids = idx_array[start:end]
        if labels is not None:
            yield Variable(torch.from_numpy(data[batch_ids])), Variable(torch.from_numpy(labels[batch_ids])), batch_ids
        else:
            yield Variable(torch.from_numpy(data[batch_ids])), batch_ids

# function that loads and pre-processes {mnist_m, mnist, usps} dataset
def data_loader(dataset):

    # helper function to normalize images between -1 and 1
    def normalize_img(x):
        return -1 + (x - np.min(x)) * 2 / (np.max(x) - np.min(x))

    # helper function to one-hot-encode target variables
    def one_hot_encode(y, n_class):
        return np.eye(np.array(n_class))[y]

    # load and process MNIST-M dataset
    if dataset == 'mnist_m':
        print('Loading {} data..'.format(dataset.upper()))
        
        # load mnist_m data
        (x_train, y_train), (x_test, y_test) = load_mnistm()
        
        # preprocess (reshape, float32, normalize)
        x_train = x_train.transpose(0, 3, 1, 2).astype('float32') / 255
        x_test = x_test.transpose(0, 3, 1, 2).astype('float32') / 255
        
        # one hot encode target
        y_train = one_hot_encode(y_train, 10)
        y_test = one_hot_encode(y_test, 10)
 
        # return images
        return (x_train, y_train), (x_test, y_test)

    # load and process USPS dataset
    if dataset == 'usps':
        print('Loading {} data..'.format(dataset.upper()))
        
        # load usps data
        (x_train, y_train), (x_test, y_test) = load_usps()
            
        # one hot encode target
        y_train = one_hot_encode(y_train, 10)
        y_test = one_hot_encode(y_test, 10)

        # parse data
        x_train = [np.delete(np.array(p.strip().split(' ')).astype('float32'), 0) for p in x_train]
        x_test = [np.delete(np.array(p.strip().split(' ')).astype('float32'), 0) for p in x_test]
        
        # reshape
        x_train = np.array(x_train).reshape(len(y_train), 1, 16, 16)
        x_test = np.array(x_test).reshape(len(y_test), 1, 16, 16)
        
        # zero-pad usps images to shape (1 x 28 x 28; compatible with mnist)
        x_train = np.array([np.pad(img, ((0,0),(6,6),(6,6)), 'constant', constant_values=-1) for img in x_train])
        x_test = np.array([np.pad(img, ((0,0),(6,6),(6,6)), 'constant', constant_values=-1) for img in x_test])

        # normalize image values between -1 and 1
        x_train = np.array([normalize_img(img) for img in x_train])
        x_test = np.array([normalize_img(img) for img in x_test])
        
        return (x_train, y_train), (x_test, y_test)

    # load and process MNIST dataset
    elif dataset == 'mnist':
        print('Loading {} data..'.format(dataset.upper()))
        
        # load mnist data
        (x_train, y_train), (x_test, y_test) = load_mnist()
        
        # preprocess (reshape, float32)
        x_train = x_train.reshape(x_train.shape[0], 1, 28, 28).astype('float32') / 255
        x_test = x_test.reshape(x_test.shape[0], 1, 28, 28).astype('float32') / 255

        # preprocess (normalize)
        x_train = np.array([normalize_img(img) for img in x_train])
        x_test = np.array([normalize_img(img) for img in x_test])
        
        # one hot encode target
        y_train = one_hot_encode(y_train, 10)
        y_test = one_hot_encode(y_test, 10)
        
        # return data
        return (x_train, y_train), (x_test, y_test)
        
        # if target_data == 'usps':
            
        #     # return same amount of MNIST samples as USPS images
        #     return (x_train[:7291], y_train[:7291]), (x_test[:2007], y_test[:2007])
        # else:
            
        #     # concat MNIST images as channels to match number of MNIST-M channels
        #     x_train = np.concatenate([x_train, x_train, x_train], axis=1)
        #     x_test = np.concatenate([x_test, x_test, x_test], axis=1)
        #     return (x_train, y_train), (x_test, y_test)
