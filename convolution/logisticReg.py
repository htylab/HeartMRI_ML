
__docformat__ = 'restructedtext en'

import pickle as cPickle
import gzip
import os
import sys
import timeit

import numpy as np
import pickle
import theano
import theano.tensor as T

fx = theano.config.floatX

class LogisticRegression(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, n_in, n_out, W = None, b = None):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        :type W: tensor of size
        :param n_out: number of output units, the dimension of the space in

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
        """

        if W is None:

            # initialize with 0 the weights W as a matrix of shape (batch_size, n_in, n_out)
            self.W = theano.shared(
                value=np.zeros(
                    (n_in, n_out),
                    dtype=theano.config.floatX
                ),
                name='W',
                borrow=True
            )

        else:
            self.W = W

        if b is None:

            # initialize the biases b as a vector of n_out 0s
            self.b = theano.shared(
                value=np.zeros(
                    (n_out,),
                    dtype=theano.config.floatX
                ),
                name='b',
                borrow=True
            )

        else:
            self.b = b

        # output
        self.output = T.nnet.sigmoid( T.dot(input, self.W) + self.b )       # batch_size x 1024
        self.thresh = T.round(self.output)
        # parameters of the model
        self.params = [self.W, self.b]                                      # W: 1024 x 8100, b: 1024 x 1

        # keep track of model input
        self.input = input


def load_data():
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    #############
    # LOAD DATA #
    #############


    print('... loading data')

    train_set_x = np.load('../data/SBXtrainImage64')
    train_set_y = np.load('../data/SBXtrainBinaryMask32')

    train_set_x = np.asarray(train_set_x, dtype='float64')
    dim = train_set_x.shape
    train_set_x = np.reshape(train_set_x, (dim[0], (dim[1]*dim[2])) )
    shared_x = theano.shared(train_set_x.astype(fx), borrow=True)                      # convert to 260 x 4096

    train_set_y = np.asarray(train_set_y, dtype='float64')
    dim = train_set_y.shape
    train_set_y = np.reshape(train_set_y, (dim[0], (dim[1]*dim[2])) )
    shared_y = theano.shared(train_set_y.astype(fx), borrow=True)                      # convert to 260 x 1024

    rval = [(shared_x, shared_y)]

    return rval

if __name__ == '__main__':
    load_data()
