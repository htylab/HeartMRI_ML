from __future__ import print_function
import os
import numpy as np
import dicom
from scipy import misc


# TODO reshape data to collapse into array of pixels (TBC depending on how input must look for model
# TODO add method for storing resulting numpy arrays as theano shared variables
# TODO add method for expanding training data set, as described in 'DSBC Challenge - Model outline'



def load_dcm_data(directory, preprocess, **args):

    """
    loads data from DCM files into numpy arrays

    :param directory: top level folder where image .dcm files are stored
    :param preprocess: function handle which preprocessing method to apply to images
    :param **args: any named arguments and their values required by preprocess method

    RETURNS: imagedata: is a numpy array numimages * imageheight * imagewidth for processed images
             imageids: identifier code for image containing info on study id
    """

    print('Source directory {0}...'.format(directory))

    imagedata = []
    imageids= []

    for root, _, files in os.walk(directory):
        total = 1

        for f in files:
            image_path = os.path.join(root, f)

            if not image_path.endswith('.dcm') or root.find("sax") == -1: #sax files are of interest initially as per data notes on kaggle
                continue

            current_image = dicom.read_file(image_path)
            current_image = current_image.pixel_array.astype(float)#/np.max(current_image.pixel_array) #between 0 and 1
            current_image = preprocess(current_image, **args) #applies selected prepocess routine
            imageid = "%s-%s" % (f.rsplit('-')[1], f.rsplit('-')[2][0:4])
            print('Image id loaded... {0}'.format(imageid))

            imagedata.append(current_image)
            imageids.append(imageid)

            total += 1

    imagedata = np.array(imagedata)

    print('Number of images {0}'.format(imagedata.shape[0]))

    return imagedata, imageids


def crop_resize(image, crop=False, newsize=()):

    """
    Crops (optional) and resizes images

    :param image: numpy array to be processed
    :param crop: true/false whether to crop image from centre
    :param newsize: optional tuples for new image size

    RETURNS: cropped and resized image
    """
    if crop:
        edge = min(image.shape[:2]) #cropping from centre by using half the short edge length
        y = int((image.shape[0] - edge) / 2)
        x = int((image.shape[1] - edge) / 2)
        image = image[y: y + edge, x: x + edge]

    image = misc.imresize(image, newsize)  #using scipy resize function with tuple for new size

    return image


def get_vol_labels(filename):

    """
    creates mapping between training images and their diastole and systole
    volumes given in 'train.csv'. These act as targets in supervised learning problem.

    :param filename: name of csv file containing the targets
    """
    file = open(filename)
    targets= file.readlines()
    targetmapping = []

    i = 0
    for item in targets:
        if i > 0:
            id, diastole, systole = item.replace('\n', '').split(',')
            targetmapping.append([float(id), float(diastole), float(systole)])
        i = i + 1

    return targetmapping

# Load data and store to numpy files for re-use

if __name__ == "__main__":

    data, ids = load_dcm_data('data/train', crop_resize, newsize = (64,48))
    target = get_vol_labels('data/train.csv')
    # numpy pickled files will appear in data folder of directory
    #np.save('data/trainIn.npy', data)
    #np.save('data/trainOut.npy', target)

    data.dump('data/trainImage')
    target.dump('data/trainVols')








