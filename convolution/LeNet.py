
import os
import sys
import timeit
import logging
import numpy
import pickle
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv2d
from logisticReg import LogisticRegression, load_data
import sys
sys.path.insert(0, '/home/odyss/Desktop/dsb/AML/DataScienceBowl/')
from LoadData import crop_resize


fx = theano.config.floatX

logging.basicConfig(filename='logistic.log', filemode='w', level=logging.INFO)

#matplotlib.use('Agg')



class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(6, 6), W = None, b = None):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape           20 x 1 x 64 x 64

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)                  100 x 1 x 11 x 11

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)                     20 x 1 x 64 x 64

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)

        :type W: tensor with size of filter_shape
        :param W: filter weights

        :type b: tensor of length (filter_shape[0],)
        :param b: bias term of each convolution
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input

        # load filter weights

        if W is None:

            # if W are not provided, generated them randomly

            fan_in = numpy.prod(filter_shape[1:])
            #    pooling size each unit in the lower layer receives a gradient from:
            # "num output feature maps * filter height * filter width" /
            #
            fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /
                       numpy.prod(poolsize))
            # initialize weights with random weights
            W_bound = numpy.sqrt(6. / (fan_in + fan_out))
            self.W = theano.shared(
                numpy.asarray(
                    rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                    dtype=theano.config.floatX
                ),
                borrow=True
            )
        else:
            self.W = W

        # load bias
        if b is None:

            # if b are not provided, generate them randomly

            b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
            self.b = theano.shared(value=b_values, borrow=True)
        else:
            self.b = b

        # convolve input feature maps with filters
        conv_out = conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=image_shape
        )
        # copv_out should be batch_size x 100 x 54 x 54

        # apply sigmoid before pooling
        conv_out = T.nnet.sigmoid( conv_out + self.b.dimshuffle('x', 0, 'x', 'x') )

        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
            #mode='average_exc_pad'
        )
        # pooled_out should be batch_size x 100 x 9 x 9

        self.output = pooled_out

        # store parameters of this layer
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input


def fine_tuning(learning_rate = 0.1, n_epochs = 1000, nkerns = 100, batch_size = 260,
                logistic_params_path = None, CNN_inputFilters_path = None, CNN_inputBias_path = None):
    """ Demonstrates lenet on MNIST dataset

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type nkerns: int
    :param nkerns: number of convolution layer filters (kernels)

    :type batch_size: int
    :param batch_size: size of batch in which the data are passed to the model
    """

    ######################
    #   INITIALIZATIONS  #
    ######################

    # load Auto-encoder pre-trained bias
    if CNN_inputBias_path is None:
        b_CNN_input = None
    else:
        b_temp = numpy.load(CNN_inputBias_path)
        b_CNN_input = theano.shared(
            value=b_temp.astype(fx),       # b is 100 x 1, is ok
            name='b_CNN_input',
            borrow = True
        )

    # load Auto-encoder pre-trained filter weights
    if CNN_inputFilters_path is None:
        W_CNN_input = None
    else:
        W = numpy.load(CNN_inputFilters_path)
        W_4D_tensor = numpy.reshape(W, (100,1,11,11))
        W_CNN_input = theano.shared(
            value=W_4D_tensor.astype(fx),    # W is 100 x 11 x 11 should convert to 100 x 1 x 11 x 11
            name='W_CNN_input',
            borrow = True
        )

    # load logistic layer pre-training parameters
    if logistic_params_path is None:
        W_logistic = None
        b_logistic = None
    else:
        with open(logistic_params_path) as f:
            params = pickle.load(f)
        W_logistic, b_logistic = params[0]

    rng = numpy.random.RandomState(23455)

    # load data set
    datasets = load_data()
    train_set_x, train_set_y = datasets[0]
    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_train_batches /= batch_size                                           # 13

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    # start-snippet-1
    x = T.matrix('x', dtype=fx)   # the data is presented as rasterized images
    y = T.matrix('y', dtype=fx)  # the labels are presented as 1D vector of
                        # [int] labels

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    #print('... building the model')

    # Reshape matrix of images of shape (batch_size, 64 * 64)
    # to a 4D tensor of shape (batch_size, 1, 64, 64)
    layer0_input = x.reshape((batch_size, 1, 64, 64))

    # Construct convolutional & pooling layer:
    # filtering reduces the image size to (64-11+1 , 64-11+1) = (54, 54)
    # maxpooling reduces this further to (54/6, 54/6) = (9, 9)
    # 4D output tensor is thus of shape (batch_size, 100, 9, 9)
    layer0 = LeNetConvPoolLayer(
        rng = rng,
        input = layer0_input,
        filter_shape = (nkerns, 1, 11, 11),
        image_shape = (batch_size, 1, 64, 64),                # batch_size x 100 x 11 x 11
        poolsize = (6, 6),
        W = W_CNN_input,
        b = b_CNN_input
    )

    # flatten out the input of the logistic layer
    layer0_output = layer0.output.flatten(2)                # batch_size x 8,100

    # classify the values of the fully-connected sigmoidal layer
    layer3 = LogisticRegression(
        input = layer0_output,
        n_in = 8100,
        n_out = 1024,
        W = W_logistic,
        b = b_logistic
    )

    layer3_output = layer3.output                           # batch_size x 1024 tensor

    # compute cost
    #cost = 0.5 * T.mean((layer3_output - y) ** 2)
    # regularization parameter
    l = 0.0001
    # calculate norms for cost
    l2_squared = (layer0.W ** 2).sum() + (layer3.W ** 2).sum()
    cost = 0.5 * T.mean((layer3_output - y) ** 2) + 0.5 * l * l2_squared

    # create a list of all model parameters to be fit by gradient descent
    params = layer3.params + layer0.params

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    # updates. loop over all parameters and gradients
    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]

    # theano function to evaluate model
    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    ###############
    # TRAIN MODEL #
    ###############
    print('... fine tuning')

    epoch = 0
    #epsilon = 0.0000005
    #last_loss = 0

    logging.debug('%-10s%-10s%-10s' %('Epoch','Batch','Cost'))
    while (epoch < n_epochs):
        epoch += 1
        for minibatch_index in xrange(n_train_batches):

            cost_ij = train_model(minibatch_index)
            #print '\nepoch = %s' % epoch
            #print 'batch = %s' % minibatch_index
            print 'epoch = %s batch = %s cost = %s' % (epoch,minibatch_index,cost_ij)
            logging.debug('%-10s %-10s %-10s' % (epoch, minibatch_index, cost_ij))
            #if cost_ij - last_loss <= epsilon:
                #print 'converged: %.2f' % (cost_ij - last_loss)
            #    logging.debug('Converged %s'%(cost_ij - last_loss))
            #    return

            #last_loss = cost_ij

    print('Optimization complete.')

    with open('../data/fine_tune_paramsXnew.pickle', 'w') as f:
        pickle.dump([params], f)



def predict(inputimages, nkerns = 100, batch_size = 260, fine_tuned_params_path = None):

    ######################
    #   INITIALIZATIONS  #
    ######################
    dim = len(inputimages)
    new_images = numpy.zeros((dim,64,64))

    for i in xrange(dim):
        new_images[i, :, :] = crop_resize(inputimages[i], newsize=(64, 64))
        new_images[i, :, :] = numpy.true_divide(new_images[i], 255)

    if fine_tuned_params_path is None:
        b_CNN_input = None
        W_CNN_input = None
        W_logistic = None
        b_logistic = None
    else:
        with open(fine_tuned_params_path) as f:
            params = pickle.load(f)

        # load pre-trained parameters
        W_logistic, b_logistic, W_CNN_input, b_CNN_input = params[0]
        W_logistic = numpy.asarray(W_logistic,dtype='float32')
        b_logistic = numpy.asarray(b_logistic,dtype='float32')
        W_CNN_input = numpy.asarray(W_CNN_input,dtype='float32')
        b_CNN_input= numpy.asarray(b_CNN_input,dtype='float32')

        W_logistic = theano.shared(W_logistic.astype(fx), borrow=True)
        b_logistic = theano.shared(b_logistic.astype(fx), borrow=True)
        W_CNN_input = theano.shared(W_CNN_input.astype(fx), borrow=True)
        b_CNN_input= theano.shared(b_CNN_input.astype(fx), borrow=True)

    rng = numpy.random.RandomState(23455)

    # manipulate data

    train_set_x = numpy.asarray(new_images, dtype='float32')
    dim = train_set_x.shape
    train_set_x = numpy.reshape(train_set_x, (dim[0], (dim[1]*dim[2])))
    train_set_x = theano.shared(train_set_x.astype(fx), borrow=True)                      # convert to 260 x 4096

    n_batches = train_set_x.get_value(borrow=True).shape[0]
    n_batches /= batch_size

    ###############
    # BUILD MODEL #
    ###############

    # build model
    #print('... building the model')
    index = T.lscalar()
    x = T.matrix('x', dtype=fx)

    # Convolution + Pooling Layer
    layer0_input = x.reshape((batch_size, 1, 64, 64))
    layer0 = LeNetConvPoolLayer(
        rng = rng,
        input = layer0_input,
        filter_shape = (nkerns, 1, 11, 11),
        image_shape = (batch_size, 1, 64, 64),
        poolsize = (6, 6),
        W = W_CNN_input,
        b = b_CNN_input
    )
    layer0_output = layer0.output.flatten(2)

    # Logistic Regression Layer
    layer3 = LogisticRegression(
        input = layer0_output,
        n_in = 8100,
        n_out = 1024,
        W = W_logistic,
        b = b_logistic
    )
    predict_model = theano.function(
        inputs = [index],
        outputs=layer3.thresh,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size]
        }
    )

    preds = [predict_model(minibatch_index) for minibatch_index in xrange(n_batches)]
    images = [numpy.reshape(preds[i],(32,32)) for i in xrange(n_batches)]

    '''
    with open('/Users/Peadar/Documents/KagglePythonProjects/AML/DataScienceBowl/data/CNN_output.pickle', 'wb') as f:
        pickle.dump(images, f)
    '''

    return images


if __name__ == '__main__':
    logging.basicConfig(filename='lenet.log', filemode='w', level=logging.DEBUG)
    # Fine Tuning of CNN + output layer
    fine_tuning(n_epochs=1000,batch_size=260,logistic_params_path = '../data/logistic_paramsXnew.pickle',CNN_inputFilters_path = '../data/CNN_inputFilters',CNN_inputBias_path = '../data/CNN_inputBias')
    # call to predict outcome after fine tuning
    predict(batch_size=1, fine_tuned_params_path = '../data/fine_tune_paramsXnew.pickle')

