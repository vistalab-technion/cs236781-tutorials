import sys
import gzip
import numpy as np
from six.moves import cPickle
from keras.utils.data_utils import get_file

# downloads mnist-m from github repository
def load_mnistm(path='keras_mnistm.pkl.gz'):
    path = get_file(path, origin='https://github.com/VanushVaswani/keras_mnistm/releases/download/1.0/keras_mnistm.pkl.gz')
    with gzip.open(path, 'rb') if path.endswith('.gz') else open(path, 'rb') as f:
        data = cPickle.load(f) if sys.version_info < (3,) else cPickle.load(f, encoding='bytes')
    (_, y_train), (__, y_test) = load_mnist()
    return (data[b'train'], y_train), (data[b'test'], y_test) 

# downloads mnist from aws s3 bucket
def load_mnist(path='mnist.npz'):
    path = get_file(path, origin='https://s3.amazonaws.com/img-datasets/mnist.npz')
    with np.load(path) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']
    return (x_train, y_train), (x_test, y_test)

# loads local copy of usps
def load_usps(path='../data/usps/'):
    with open(path+'zip.train') as f:
        x_train = f.readlines()
    with open(path+'zip.test') as f:
        x_test = f.readlines()
    y_train = np.array([int(digit[0]) for digit in x_train])
    y_test = np.array([int(digit[0]) for digit in x_test])
    return (x_train, y_train), (x_test, y_test)
    