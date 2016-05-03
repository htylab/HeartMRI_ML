import theano.tensor as T
import numpy as np
import timeit
from theano.tensor.shared_randomstreams import RandomStreams
import theano
import hiddenLayer as HL


class LogisticRegression(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, masks, n_in, n_out):
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

        """
        rng = np.random.RandomState(123)
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        initial_W = np.asarray(
                rng.uniform(  # uniform initialization of W
                    low = -4 * np.sqrt(6. / (n_in + n_out)),
                    high = 4 * np.sqrt(6. / (n_in + n_out)),
                    size = (n_in, n_out) # n_hidden x n_visible matrix
                ),
                dtype = theano.config.floatX)

        self.W = theano.shared(value=initial_W, name='W', borrow=True)

        # initialize the biases b as a vector of n_out 0s

        initial_b = np.asarray(
                rng.uniform(  # uniform initialization of b
                    low = -4 * np.sqrt(6. / (n_in + n_out)),
                    high = 4 * np.sqrt(6. / (n_in + n_out)),
                    size = (n_out,)
                ),
                dtype = theano.config.floatX)
        self.b = theano.shared(value=initial_b, name='b', borrow=True)

        # symbolic variables for inputs
        if masks is None:
            self.Y = T.dmatrix(name='masks')
        else:
            self.Y = masks
        if input is None:
            self.X = T.dmatrix(name='input')
        else:
            self.X = input

        # push through sigmoid layer to get prediction prob
        self.p_y_given_x = T.nnet.sigmoid(T.dot(self.X, self.W) + self.b)

        # set to 1 if greater than 0.5, 0 otherwise
        self.y_pred = T.round(self.p_y_given_x)

        # parameters of the model
        self.params = [self.W, self.b]


    def get_cost_updates(self, learning_rate, lam, rho=0, beta=0):
        """
        :type scalar
        :param learning_rate: rate which weighs the gradient step

        :type scalar
        :param lam: regularization parameter for the cost function

        :type pair (cost, update)
        :return: compute cost and update for one training step of the autoencoder
        """

        # Compute the cost
        l2_squared = (self.W ** 2).sum()
        cost = 0.5*T.mean((self.p_y_given_x-self.Y ) ** 2) + (0.5*lam*l2_squared)

        # Compute updates
        gparams = T.grad(cost, self.params)
        updates = [(param, param - learning_rate * gparam)
            for param, gparam in zip(self.params, gparams)]

        return cost, updates


def train_logreg(train_data, train_masks, numbatches,
                 n_epochs, model_class, **args):

    """
    trains auto-encoder model for initialising weights for the CNN layer of the model, taking as input
    random mini batches from the training images.
    :param training data
    :param n_epochs: number of training iterations
    :param model_class: class of model to train
    :param **args: any named inputs required by the cost function


    RETURNS: final array of weights from trained model
    """

    traindim = train_data.shape
    batch_size = traindim[0]/numbatches

    X = T.matrix('X')
    Y = T.matrix('Y')
    index = T.lscalar()

    train_data = theano.shared(train_data)
    train_masks = theano.shared(train_masks)


    model_object = model_class(
            input=X,
            masks=Y,
            n_in=100,
            n_out=1024)

    cost, updates = model_object.get_cost_updates(**args)

    train_model = theano.function(inputs=[index], outputs=cost, updates=updates,
                                  givens={X: train_data[index * batch_size:(index + 1) * batch_size],
                                          Y: train_masks[index * batch_size:(index + 1) * batch_size]})

    # go through training epochs
    HL.iterate_epochs(n_epochs, numbatches, train_model, model_class)

    weights = model_object.W.get_value()
    bias = model_object.b.get_value

    return weights, bias

if __name__ == "__main__":

    # load required inputs and call training method (random data used until CNN is working)

    trainMask = np.random.rand(4000, 32, 32)
    train = np.random.rand(4000, 100)
    train = np.array(train, dtype='float64')

    dim = trainMask.shape
    trainMask = np.reshape(trainMask, (dim[0], (dim[1]*dim[2])))
    trainMask = np.array(trainMask, dtype='float64')
    numbatches = 1
    batchdim = train.shape[0]/numbatches

    weights, bias = train_logreg(train_data=train, train_masks=trainMask,
                                            numbatches=numbatches, n_epochs=10000,
                                            model_class=LogisticRegression, datadim=batchdim,
                                            learning_rate=10, lam=0.0001)

