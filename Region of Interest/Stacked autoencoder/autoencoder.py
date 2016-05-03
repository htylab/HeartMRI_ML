import theano.tensor as T
import numpy as np
from theano.tensor.shared_randomstreams import RandomStreams
import theano
import hiddenLayer as HL

class AutoEncoder(object):
    def __init__(
        self,
        numpy_rng,
        theano_rng = None,
        input = None,
        n_visible = 121,
        n_hidden = 100,
        Whid = None,
        Wvis = None,
        bhid = None,
        bvis = None
    ):
        """
        :type numpy_rng: numpy.random.RandomState
        :param numpy_rng: number random generator used to generate weights

        :type theano_rng: theano.tensor.shared_randomstreams.RandomStreams
        :param theano_rng: Theano random generator; if None is given one is
                     generated based on a seed drawn from `rng`

        :type input: theano.tensor.TensorType
        :param input: a symbolic description of the input or None for
                      expects a (num_observations x n_visible) dmatrix

        :type n_visible: int
        :param n_visible: number of visible units

        :type n_hidden: int
        :param n_hidden:  number of hidden units

        :type Whid: theano.shared
        :param Whid: Theano shared variable for the mapping from the input layer to
                     the hidden layer ( hid = f(Whid * input + bhid) )
                     val: theano.tensor.TensorType (100 x 121 numpy array)

        :type Wvis: theano.shared
        :param Wvis: Theano shared variable for the mapping from the hidden layer to
                     the output layer ( y = f(Whid * hid + bhid) )
                     val: theano.tensor.TensorType (121 x 100 numpy array)

        :type bhid: theano.shared
        :param bhid: Theano shared variable for the mapping from the input layer to
                     the hidden layer ( hid = f(Whid * input * bhid)

        :type bvis: theano.shared
        :param bvis: Theano shared variable for the mapping from the input layer to
                     the hidden layer ( output = f(Wvis * hid * bvis)
        """
        # initialisation

        self.n_visible = n_visible
        self.n_hidden = n_hidden

        # if no random number generator was passed to the class instance
        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2**30)) # random generator with
                                                                 # seed

        # if no weights for the mapping from input layer to hidden layer were passed
        if not Whid:
            # Whid is initialized with initial_W which is uniformly sampled
            initial_Whid = np.asarray(
                numpy_rng.uniform(  # uniform initialization of Whid
                    low = -4 * np.sqrt(6. / (n_hidden + n_visible)),
                    high = 4 * np.sqrt(6. / (n_hidden + n_visible)),
                    size = (n_visible, n_hidden) # n_hidden x n_visible matrix
                ),
                dtype = theano.config.floatX # theano.config.floatX enables GPU
            )
            Whid = theano.shared(value=initial_Whid, name='Whid', borrow=True)

        # if no weights for the mapping from hidden layer to output layer were passed
        if not Wvis:
            # Wvis is initialized with initial_W which is uniformly sampled
            initial_Wvis = np.asarray(
                numpy_rng.uniform(  # uniform initialization of Whid
                    low = -4 * np.sqrt(6. / (n_hidden + n_visible)),
                    high = 4 * np.sqrt(6. / (n_hidden + n_visible)),
                    size = ( n_hidden, n_visible) # n_visible x n_hiden matrix
                ),
                dtype = theano.config.floatX # theano.config.floatX enables GPU
            )
            Wvis = theano.shared(value=initial_Wvis, name='Wvis', borrow=True)

        # if no bias for mapping from input to hidden layer is passed
        if not bhid:
            bhid = theano.shared(
                value=np.zeros(
                    n_hidden,                  # size
                    dtype=theano.config.floatX # GPU
                ),
                name='bhid',
                borrow=True                    # I dont think borrow=True necessary here
            )

        # if no bias for mapping from hidden to output layer is passed
        if not bvis:
            bvis = theano.shared(
                value=np.zeros(
                    n_visible,                 # size
                    dtype=theano.config.floatX # GPU
                ),
                name='bvis',
                borrow = True                  # I dont think borrow=True necessary here
            )

        self.Whid = Whid   # the weight matrix from input to hidden (val is 121 x 100 array)
        self.Wvis = Wvis   # the weight matrix from hidden to output (val is 121 x 100 array)
        self.bhid = bhid   # the bias vector for the hidden layer (100 x 1 array)
        self.bvis = bvis   # the bias vector for the visible layer (121 x 1 array)
        self.theano_rng = theano_rng

        # self.X is the megamatrix (number of rows = number of observations, number of cols = unrolled image size)
        # self.Xbatch is the minibatch matrix (number of rows = number of minibatches, number of cols = unrolled mini-batch size)
        if input is None:
            self.X = T.dmatrix(name='input')
        else:
            self.X = input


        # these are the parameters we are optimizing
        self.params = [self.Whid, self.bhid, self.Wvis, self.bvis]

    def get_hidden_values(self, input):
        """
        :return: Compute the values of the hidden layer
        """

        return T.nnet.sigmoid(T.dot(input, self.Whid) + self.bhid)

    def get_output(self, hidden):
        """
        :return: Compute the reconstructed input given the hidden layer
        """
        return T.nnet.sigmoid(T.dot(hidden, self.Wvis) + self.bvis)

    # pass every minibatch through the autoencoder and calculate the y's
    def get_cost_updates(self, learning_rate, lam = 0.0001, beta=3, rho = 0.1):
        """
        :type scalar
        :param learning_rate: rate which weighs the gradient step

        :type scalar
        :param lam: regularization parameter for the cost function

        :type pair (cost, update)
        :return: compute cost and update for one training step of the autoencoder
        """

        # y holds all the minibatch-processed vectors

        h = self.get_hidden_values(self.X)
        y = self.get_output(h)

        # Compute the cost
        l2_squared = (self.Wvis ** 2).sum() + (self.Whid ** 2).sum()
        KL = T.abs_(rho - T.mean(h))               # True KL?? How to deal with distribution...T.log(T.true_div(rho,rho_hat))
        cost = 0.5*T.mean((y - self.X) ** 2)+0.5*lam*l2_squared # + beta*KL


        # Compute updates
        gparams = T.grad(cost, self.params)
        updates = [(param, param - learning_rate * gparam)
            for param, gparam in zip(self.params, gparams)]

        return cost, updates



def train_ac(train_data, numbatches, n_epochs, model_class, **args):

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
    index = T.lscalar()

    train_data = theano.shared(train_data)
    rng = np.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    model_object = model_class(numpy_rng=rng,
            theano_rng=theano_rng,
            input=X,
            n_visible=121,
            n_hidden=100)

    cost, updates = model_object.get_cost_updates(**args)

    train_model = theano.function(inputs=[index], outputs=cost, updates=updates,
                                   givens={X: train_data[index * batch_size:(index + 1) * batch_size]})

    # go through training epochs
    HL.iterate_epochs(n_epochs, numbatches, train_model, model_class)

    W_hid = model_object.Whid.get_value()
    b_hid = model_object.bhid.get_value()

    return W_hid, b_hid

if __name__ == "__main__":

    # load sunny data and collapse to correct dim

    train = np.load('/Users/Peadar/Documents/KagglePythonProjects/AML/DataScienceBowl/data/SBXtrainImage_batch')
    dim = train.shape
    train = np.reshape(train, (dim[0], (dim[1]*dim[2])))
    train = np.array(train, dtype='float64')
    numbatches = 5
    batchdim = train[0]/numbatches

    W_hid, b_hid = train_ac(train_data=train, numbatches=numbatches, n_epochs=10000,
                               model_class = AutoEncoder, learning_rate=10, lam=10^4,
                            beta=3, rho = 0.1)

    W_hid = np.array(W_hid)
    b_hid = np.array(b_hid)

    # configure final output as inputs for CNN, represented by 100 11*11 filters:

    W_hid = np.transpose(W_hid)
    W_hid = np.reshape(W_hid, (W_hid.shape[0], dim[1], dim[2]))

    W_two = W_hid.dump('/Users/Peadar/Documents/KagglePythonProjects/AML/DataScienceBowl/data/CNN_inputFilters')
    b_hid = b_hid .dump('/Users/Peadar/Documents/KagglePythonProjects/AML/DataScienceBowl/data/CNN_inputBias')

