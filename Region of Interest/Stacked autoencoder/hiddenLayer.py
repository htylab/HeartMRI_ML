import theano.tensor as T
import numpy as np
from theano.tensor.shared_randomstreams import RandomStreams
import theano

class HiddenLayer(object):

    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=theano.tensor.nnet.sigmoid):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input = input

        if W is None:
            W_values = np.asarray(
                rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b
        self.params = [self.W, self.b]

        lin_output = T.dot(input, self.W) + self.b
        self.output = activation(lin_output)



def iterate_epochs(n_epochs, numbatches, train_fn, model_class):
    '''_
        runs through training epochs for a given theano function
    '''
    print '######------------TRAINING MODEL %s, ' % model_class
    for epoch in xrange(n_epochs):
        for nindex in range(numbatches):
            c = train_fn(nindex) #compute cost
            print 'Training epoch %d,' % epoch
            print 'Batch id %s,' % nindex
            print 'Cost %s,' % c

            print '----------------------------'
            print '----------------------------'
