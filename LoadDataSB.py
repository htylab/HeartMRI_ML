from __future__ import print_function
import os
import fnmatch
import re
import numpy as np
import dicom
import cv2
from LoadData import crop_resize
import random
import matplotlib.pyplot as plt

# TODO add method for storing resulting numpy arrays as theano shared variables

# Declare the top level directories that hold the image and contour files within the sunnybrook data

sb_root = "/Users/Peadar/Documents/KagglePythonProjects/SunnybrookData"
image_dir = os.path.join(sb_root, "challenge_training")
contour_dir = os.path.join(sb_root,"Sunnybrook Cardiac MR Database ContoursPart3","TrainingDataContours")


# Two methods required, 1 for getting meta data about the relationship between the contour files and the image files,
# and the other to load images and masks into numpy array prior to saving to binary numpy files for easy re-use

def get_mapping(contour_dir):
    """
    gets the path, case and corresponding image mapping for each contour
    :param contour_dir: top level directory path where contours live

    RETURNS: arrays for contour path, contour series id and corresponding image id
    #NOTE: Consider to include CLASS for contour to clean up implementation in future.

    """
    c_path = []
    c_series = []
    c_imgid = []

    for root, _, files in os.walk(contour_dir):

         #Only interested in type 'i' images - representing endocardium. See sunnybrook 'README Contour Format ... . txt'.

        for f in fnmatch.filter(files, 'IM-0001-*-icontour-manual.txt'):
            path = os.path.join(root, f)
            c_path.append(path)

            # use regular expressions to match strings

            lookup = re.search(r"/([^/]*)/contours-manual/IRCCI-expert/IM-(\d{4})-(\d{4})-icontour-manual.txt", path)

            c_series.append(lookup.group(1)) # Contour series ID
            #c_imgid.append("%s-%s" % (lookup.group(2), lookup.group(3)))  # Contour corresponding image ID based on parsed regular expression
            c_imgid.append(lookup.group(3))

    print("Number of contours: {:d}".format(len(c_path)))

    return c_path, c_series, c_imgid


def load_contours_dcm(c_path, c_series, c_imgid,
                      image_dir, training_dic, preprocess, **args):

    """
    loads data from DCM files into numpy arrays, and creates mask given contours
    :param c_path, c_series, c_imgid: array results from get_mapping
    :param image_dir: top level directory path where training images live

    RETURNS: numpy arrays for image data and corresponding masks built from contour co-ordinates
    """

    print('Source directory {0}...'.format(image_dir))

    imagedata = []
    contourdata= []

    for i in range(0, len(c_series)):

        file = "IM-%s-%s.dcm" % (training_dic[c_series[i]], c_imgid[i])  #Builds image path based on ID returned from contour
        path = os.path.join(image_dir, c_series[i], file)
        current_image = dicom.read_file(path)
        current_image = current_image.pixel_array.astype(float)

        contour = np.loadtxt(c_path[i], delimiter=" ").astype(np.int)
        mask = np.zeros_like(current_image, dtype="uint8")
        cv2.fillPoly(mask, [contour], 1) #Need to install dependencies for cv2

        #further preprocess image by passed method preprocess
        current_image = preprocess(current_image, **args)
        mask = preprocess(mask, **args)
        current_image = np.true_divide(current_image,255)


        #collect data
        imagedata.append(current_image)
        contourdata.append(mask)


    imagedata = np.array(imagedata)
    contourdata = np.array(contourdata)

    print('Number of images {0}'.format(imagedata.shape[0]))

    return imagedata, contourdata


    # create 10000 minibatches

def get_image_batch(imagedata, batchsize, numbatches):

    """
    generates numbatches number of random batchsize ([heigth * width]) image patches for passing
    to auto encoder layer form the downsampled images in imagedata

    RETURNS: array [numbatches * batchsize[0] * batchsize[1]

    """
    imagedata_batch = []

    for i in range(0, numbatches):

        # get random indices required for patch size

        imageid = random.randint(0, (imagedata.shape[0] - 1))
        patch_w = random.randint(0, (imagedata.shape[1] - 12))
        patch_h= random.randint(0, (imagedata.shape[2] - 12))
        patch_w_r = patch_w + batchsize[0]-1
        patch_h_r = patch_h + batchsize[1]-1

        # extract current patch from imagedata
        current_batch = np.array(imagedata[imageid, patch_w:patch_w_r+1, patch_h:patch_h_r+1])

        # collect data
        imagedata_batch.append(current_batch)

    imagedata_batch = np.array(imagedata_batch)

    return imagedata_batch


def get_binary_masks(contourdata, mask_region, preprocess, **args):

    # find indices in countour image that corresponds to contour
    # workout the centre and the min and max

    dim = contourdata.shape
    mask_binary = []
    for i in range(0, dim[0]):

        mask = contourdata[i,:,:]
        rows, cols = np.where(mask==1)

        cen_x, cen_y = (np.median(cols), np.median(rows))

        mask[cen_y - (mask_region[1]/2):cen_y + (mask_region[1]/2),
            cen_x - (mask_region[0]/2):cen_x + (mask_region[0]/2)] = 1

        # mask[cen_x, cen_y]=1  Illustrate centre if required

        mask_binary.append(preprocess(mask,**args))

    mask_binary = np.array(mask_binary)

    return mask_binary




# There is a manual process to map contours to images, as IDs don't match exactly. Resulting in Dic:

SAX_SERIES = {

    "SC-HF-I-01": "0004",
    "SC-HF-I-02": "0106",
    "SC-HF-I-04": "0116",
    "SC-HF-I-40": "0134",
    "SC-HF-NI-03": "0379",
    "SC-HF-NI-04": "0501",
    "SC-HF-NI-34": "0446",
    "SC-HF-NI-36": "0474",
    "SC-HYP-01": "0550",
    "SC-HYP-03": "0650",
    "SC-HYP-38": "0734",
    "SC-HYP-40": "0755",
    "SC-N-02": "0898",
    "SC-N-03": "0915",
    "SC-N-40": "0944",
}


# Load data and store to numpy files for re-use



if __name__ == "__main__":

    c_path, c_series, c_imgid = get_mapping(contour_dir)
    #imagedata, contourdata = load_contours_dcm(c_path, c_series, c_imgid,
                                              # image_dir, SAX_SERIES, crop_resize,
                                             #  newsize=(256, 256))

    imagedata = np.load('/Users/Peadar/Documents/KagglePythonProjects/AML/DataScienceBowl/data/SBXtrainImage256')
    contourdata = np.load('/Users/Peadar/Documents/KagglePythonProjects/AML/DataScienceBowl/data/SBXtrainMask256')

    masks_binary = get_binary_masks(contourdata, mask_region = (100,100),
                                    preprocess=crop_resize, newsize=(32, 32))

    #print(masks_binary.shape)

    # numpy pickled files will appear in data folder of directory
    # use numpy.load to access
    #imagedata.dump('data/SBtrainImage256')
   # contourdata.dump('data/SBtrainMask256')
    masks_binary.dump('data/SBXtrainBinaryMask32')
    #imagedata_batch.dump('data/SBtrainImage_batch11from64')






















