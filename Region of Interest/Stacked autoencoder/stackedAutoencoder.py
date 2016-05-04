import theano.tensor as T
import numpy as np
from theano.tensor.shared_randomstreams import RandomStreams
import theano
import hiddenLayer as HL
import autoencoder as AC
import logisticRegression as LR
from scipy import misc
#import matplotlib.pyplot as plt
import pickle
import os
import pdb
fx = theano.config.floatX

class SA(object):
    """Stacked denoising auto-encoder class (SdA)
    """

    def __init__(
        self,
        inputs,
        masks,
        numpy_rng,
        theano_rng=None,
        n_ins=4096,
        hidden_layers_sizes=[200, 100, 100, 200],
        n_outs=4096):

        self.sigmoid_layers = []
        self.HL_output = []
        self.AutoEncoder_layers = []
        self.params = []
        self.n_layers = len(hidden_layers_sizes)

        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        if masks is None:
            self.Y = T.dmatrix(name='masks')
        else:
            self.Y = masks
        if input is None:
            self.X = T.dmatrix(name='input')

        else:
            self.X = inputs

        for i in xrange(self.n_layers):

            # construct the sigmoidal layer
            # the size of the input
            if i == 0:
                input_size = n_ins
            else:
                input_size = hidden_layers_sizes[i - 1]

            # the input to this layer
            if i == 0:
                layer_input = self.X
            else:
                layer_input = self.sigmoid_layers[-1].output

            sigmoid_layer = HL.HiddenLayer(rng=numpy_rng,
                                        input=layer_input,
                                        n_in=input_size,
                                        n_out=hidden_layers_sizes[i],
                                        activation=theano.tensor.nnet.sigmoid)

            # add the layer to our list of layers
            self.sigmoid_layers.append(sigmoid_layer)
            self.params.extend(sigmoid_layer.params)


            # Construct an autoencoder that shared weights with this layer and append to list
            AutoEncoder_layer = AC.AutoEncoder(numpy_rng=numpy_rng,
                          theano_rng=theano_rng,
                          input=layer_input,
                          n_visible=input_size,
                          n_hidden=hidden_layers_sizes[i],
                          Whid=sigmoid_layer.W,
                          bhid=sigmoid_layer.b)
            self.AutoEncoder_layers.append(AutoEncoder_layer)

        # We now need to add a logistic layer on top of the MLP
        self.logLayer = LR.LogisticRegression(
            input=self.sigmoid_layers[-1].output,
            masks=self.Y,
            n_in=hidden_layers_sizes[-1],
            n_out=n_outs
        )

        self.params.extend(self.logLayer.params)

    def get_cost_updates(self, learning_rate, lam):
        """
        :type scalar
        :param learning_rate: rate which weighs the gradient step
        :type scalar
        :param lam: regularization parameter for the cost function
        :type pair (cost, update)
        :return: compute cost and update for one training step of the autoencoder
        """

        # Compute the cost
        l2_squared= 0
        for i in xrange(self.n_layers):
            l2_squared += (self.sigmoid_layers[i].W** 2).sum()
        l2_squared = l2_squared + (self.logLayer.W** 2).sum()
        cost = 0.5*T.mean((self.Y - self.logLayer.p_y_given_x) ** 2)+ (0.5*lam*l2_squared)

        # Compute updates
        gparams = T.grad(cost, self.params)
        updates = [(param, param - learning_rate * gparam)
            for param, gparam in zip(self.params, gparams)]

        return cost, updates




def pretrain_sa(train_data, train_masks, numbatches, n_epochs, model_class, **args):
    '''_
        Pretrains stacked autoencoder
    '''


    X = T.matrix('X')
    Y = T.matrix('Y')
    index = T.lscalar('index')

    traindim = train_data.shape
    batch_size = traindim[0]/numbatches

    train_data = theano.shared(train_data)
    train_masks = theano.shared(train_masks)

    rng = np.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    model_object = model_class(
        inputs=X,
        masks=Y,
        numpy_rng=rng,
        theano_rng=theano_rng,
        n_ins=4096,
        hidden_layers_sizes=[200, 100,100, 200],
        n_outs=4096)

    for autoE in model_object.AutoEncoder_layers:
        # get the cost and the updates list
        cost, updates = autoE.get_cost_updates(**args)
        # compile the theano function
        train_model = theano.function(
            inputs=[index],
            outputs=cost,
            updates=updates,
            givens={X: train_data[index * batch_size:(index + 1) * batch_size]})
        HL.iterate_epochs(n_epochs, numbatches, train_model, autoE)

    logReg = model_object.logLayer
    logcost, logupdates = logReg.get_cost_updates(**args)
    train_model = theano.function(
            inputs=[index],
            outputs=logcost,
            updates=logupdates,
            givens={X: train_data[index * batch_size:(index + 1) * batch_size],
                    Y: train_masks[index * batch_size:(index + 1) * batch_size]})

    HL.iterate_epochs(n_epochs, numbatches, train_model, logReg)


    return model_object



def finetune_sa(train_data, train_masks, numbatches, n_epochs, pretrainedSA, **args):

    '''
        Fine tunes stacked autoencoder
    '''
    finetunedSA = pretrainedSA

    traindim = train_data.shape
    batch_size = traindim[0]/numbatches


    index = T.lscalar()



    train_data = theano.shared(train_data)
    train_masks = theano.shared(train_masks)

    cost, updates = finetunedSA.get_cost_updates(**args)

    train_model = theano.function(inputs=[index], outputs=cost, updates=updates,
                                  givens={finetunedSA.X: train_data[index * batch_size:(index + 1) * batch_size],
                                          finetunedSA.Y: train_masks[index * batch_size:(index + 1) * batch_size]})

    HL.iterate_epochs(n_epochs, numbatches, train_model, finetunedSA)

    return finetunedSA



def predict_sa(images, trained_SA_path = '/data/SA_Xmodel'):

    with open(trained_SA_path) as f:
         SA_inst = pickle.load(f)

    images = np.asarray(images)
    dim = images.shape
    dim0 = dim[0]
    eg_image = images[1]
    image_dim = eg_image.shape

    images = np.reshape(images, (dim0, (image_dim[0]*image_dim[1])))

    mask_predictions = []

    predict_model = theano.function(
            inputs = [SA_inst.X],
            outputs= SA_inst.logLayer.y_pred)

    for i in range(0, dim[0]):
        current_image = np.reshape(images[i,:], ((image_dim[0]*image_dim[1]), 1))
        pred = predict_model(np.transpose(current_image).astype(fx))
        mask_predictions.append(pred)

    mask_predictions = np.reshape(mask_predictions, (dim0, image_dim[0], image_dim[1]))
    images = np.reshape(images, (dim0, image_dim[0], image_dim[1]))
    masks = np.array(mask_predictions)

#    for i in range(0,10):
#        plt.subplot(1,2,1)
#        plt.imshow(masks[i,:,:])
#        plt.subplot(1,2,2)
#        plt.imshow(images[i,:,:])
#        plt.show()

    return masks


def crop_ROI(images, roi, roi_dim, newsize):

    # images is a list of arrays
    dim = len(images)
    image_roi = []

    for i in range(0, dim):

        # prep image files including up sampling roi to 64 x 64
        image = images[i]
        region = roi[i]
        region = misc.imresize(region, (256, 256))

        # get roi co-ords for cropping; using centre
        if np.max(region) >1:
           region = np.true_divide(region, 255)

        rows, cols = np.where(region == 1)
        cen_x, cen_y = (np.round(np.median(cols)), np.round(np.median(rows)))

        ind_y_lower = cen_y - (roi_dim[1]/2)
        ind_y_upper = cen_y + (roi_dim[1]/2)
        if ind_y_lower < 0:
            ind_y_upper -= ind_y_lower
            ind_y_lower = 0

        ind_x_lower = cen_x - (roi_dim[1]/2)
        ind_x_upper = cen_x + (roi_dim[1]/2)
        if ind_x_lower < 0:
            ind_x_upper -= ind_x_lower
            ind_x_lower = 0

        ind_y_lower = int(ind_y_lower)
        ind_y_upper = int(ind_y_upper)
        ind_x_lower = int(ind_x_lower)
        ind_x_upper = int(ind_x_upper)

        # execute  cropping on the image to produce ROI

        image = image[ind_y_lower:ind_y_upper, ind_x_lower:ind_x_upper]

        image = misc.imresize(image, newsize)

        if np.max(image) >1:
           image = np.true_divide(image, 255)

        image_roi.append(image)

    image_roi = np.array(image_roi)

    return image_roi


if __name__ == "__main__":


    # load required inputs and call training method (random data used until CNN is working)
    roi = np.load('/data/SBXtrainBinaryMask32')
    train = np.load('/data/SBXtrainImage256')
    mask = np.load('/data/SBXtrainMask256')

    dimimages = roi.shape
    numimages = dimimages[0]



    with open('/data/CNN_output.pickle', 'rb') as f:
        roi_pred = pickle.load(f)
        roi_pred = np.asarray(roi_pred)
        thres = 0.5
        roi_pred[roi_pred >= thres ] = 1
        roi_pred[roi_pred < thres ] = 0

    train_roi =crop_ROI(images=train, roi=roi, roi_dim=(100,100), newsize=(64, 64))

    mask_roi =crop_ROI(images=mask, roi=roi, roi_dim=(100,100), newsize=(64, 64))


    dim = mask_roi.shape

    mask_roi = np.reshape(mask_roi, (dim[0], (dim[1]*dim[2])))
    mask_roi = np.array(mask_roi, dtype=fx)

    train_roi = np.reshape(train_roi, (dim[0], (dim[1]*dim[2])))
    train_roi= np.array(train_roi, dtype=fx)

    numbatches = 1
    batchdim = train.shape[0]/numbatches

        #Just test the output



    pretrainedSA = pretrain_sa(train_data=train_roi, train_masks=mask_roi, numbatches =numbatches,
                               n_epochs=1, model_class=SA,
                                            learning_rate=100, lam=0.0001, beta=3, rho = 0.1)

    finetunedSA = finetune_sa(train_data =train_roi, train_masks=mask_roi, numbatches =numbatches,
                               n_epochs=1, pretrainedSA=pretrainedSA,
                                            learning_rate=100, lam=0.0001)


    with open('/data/SA_PreModel', 'wb') as f:
        pickle.dump(pretrainedSA, f)

    with open('/data/SA_Model', 'wb') as g:
        pickle.dump(finetunedSA, g)

    train_roi = np.reshape(train_roi, (dim[0], dim[1],dim[2]))
    mask_roi = np.reshape(mask_roi, (dim[0], dim[1],dim[2]))
    mask_predictions = predict_sa(train_roi, trained_SA_path = '/data/SA_Model')


    with open('/data/SA_Pred', 'wb') as f:
        pickle.dump(mask_predictions, f)
