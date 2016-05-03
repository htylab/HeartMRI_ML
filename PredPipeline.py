
import theano.tensor as T
import numpy as np
from numpy import genfromtxt
from theano.tensor.shared_randomstreams import RandomStreams
import theano
from scipy import misc
import matplotlib.pyplot as plt
import pickle
import os
import re
import dicom
from LoadData import crop_resize
from sklearn import linear_model
import sys
sys.path.insert(0, '/Users/Peadar/Documents/KagglePythonProjects/AML/DataScienceBowl/convolution/')
import LeNet
from LeNet import predict as CNNpred
sys.path.insert(0, '/Users/Peadar/Documents/KagglePythonProjects/AML/DataScienceBowl/Region of Interest/Stacked autoencoder/')
from stackedAutoencoder import predict_sa as SApred
from stackedAutoencoder import crop_ROI
from stackedAutoencoder import SA

sys.path.insert(0, '/Users/Peadar/Documents/KagglePythonProjects/AML/DataScienceBowl/Active Contour/')
import active_contour as AC




class Patient(object):

    # static variable
    patient_count = 0

    def __init__(self, train_path, study):
        # intervening directories
        while True:
            subs = next(os.walk(train_path))[1]
            if len(subs) == 1:
                train_path = os.path.join(train_path, subs[0])
            else:
                break
        # now subs contains 'sax5', 'sax6', ...

        # list of height indices
        slices = []

        for sub in subs:
            m = re.match('sax_(\d+)', sub)
            if m is not None:
                slices.append(int(m.group(1)))

        # slices is a list containing [17,10,5,9,...,57] corresponding to sax

        slices_map = {}
        first = True
        times = []

        for cslice in slices:
            cslice_files = next(os.walk(os.path.join(train_path, 'sax_%d' % cslice)))[2]
            # cslice_files contains a list ['IM-4557-0021.dcm', 'IM-4557-0026.dcm',...]
            offset = None
            for cslice_file in cslice_files:
                m = re.match('IM-(\d{4,})-(\d{4})\.dcm', cslice_file)
                if m is not None:
                    if first:
                        times.append(int(m.group(2)))
                    if offset is None:
                        offset = int(m.group(1))
            first = False
            slices_map[cslice] = offset

        # some instance variables
        self.directory = train_path
        self.time = sorted(times)
        self.slices = sorted(slices)
        self.slices_map = slices_map
        Patient.patient_count += 1
        self.name = study
        self.images = []
        self.predROIs = []
        self.imagesROIs = []
        self.predSAContours = []
        self.predACContours = []

    # returns the name of a file of a specific slice on specific time
    def _filename(self, cslice, time):
        return os.path.join(self.directory, 'sax_%d' % cslice,
        'IM-%04d-%04d.dcm' % (self.slices_map[cslice], time))

    # read one single dicom file

    def _read_dicom_image(self, filename):
        d = dicom.read_file(filename)
        img = d.pixel_array.astype('float')
        img = crop_resize(img, newsize=(64,64))  # PH Added preprocessing
        img = np.true_divide(img,255)

        return img

    # loads all patient's images. saves at instance variable self.images
    # images: [slice_height x time x 64 x 64]
    # also calculates slice thickness and area_multiplier (one for all images)
    def _read_all_dicom_images(self):

        #Computing distance between...
        f1 = self._filename(self.slices[0], self.time[0])
        d1 = dicom.read_file(f1)
        (x, y) = d1.PixelSpacing
        (x, y) = (float(x), float(y))
        f2 = self._filename(self.slices[1], self.time[0])
        d2 = dicom.read_file(f2)

        # try a couple of things to measure distance between slices
        try:
            dist = np.abs(d2.SliceLocation - d1.SliceLocation)
        except AttributeError:
            try:
                dist = d1.SliceThickness
            except AttributeError:
                dist = 8  # better than nothing...

        self.images = np.array([[self._read_dicom_image(self._filename(d, i))
                                 for i in self.time]
                                for d in self.slices])
        # sets dist equal to the two first slices distance (subtracting first
        # from second) or equal to first slice's thickness. I THINK the second
        # logic is better
        #
        # maybe set dist = d1.SliceThickness
        self.dist = dist
        self.area_multiplier = x * y



    def predictContours(self):

        '''
        Method for pushing images loaded by _read_all_dicom_images through full learned network
        to predict the resulting contour
           STEPS: 1) Feedforward through CNN, followed by through SA, followed by calling AC model, giving final result

        :return:
        '''

        #images are slice * time * height * width
        self.predROIs = np.array([CNNpred(inputimages = self.images[s,:], batch_size=1,
                                          fine_tuned_params_path = '/Users/Peadar/Documents/KagglePythonProjects/AML/DataScienceBowl/data/fine_tune_paramsXnew.pickle')
                                  for s in range(0, len(self.slices))])


        self.imagesROIs = np.array([crop_ROI(images=self.images[s,:], roi=self.predROIs[s,:],
                                             roi_dim=(100,100), newsize=(64, 64))
                              for s in range(0, len(self.slices))])

        self.predSAContours = np.array([SApred(self.imagesROIs[s,:],
                                               trained_SA_path ='/Users/Peadar/Documents/KagglePythonProjects/AML/DataScienceBowl/data/SA_XPHmodel')
                                        for s in range(0, len(self.slices))])

        self.predACContours = np.array([[AC.evolve_contour(lv = self.predSAContours[s,t], roi=self.imagesROIs[s,t],alpha3=0)
                                         for t in range(0, len(self.time))] for s in range(0, len(self.slices))])



def calc_areas(images):
    # images are circular binary masks
    (slice_locations, times, _, _) = images.shape
    areas = [{} for t in range(times)]
    for t in range(times):
        for h in range(slice_locations):
            areas[t][h] = np.count_nonzero(images[h][t])
    return areas

def calc_volume(areas, mult, dist):
    slices = np.array(sorted(areas.keys()))
    slices_new = [areas[h]*mult for h in slices]
    vol = 0
    for h in slices[:-1]:
        a, b = slices_new[h], slices_new[h+1]
        dvol = (dist/3.0) * (a + np.sqrt(a*b) + b)
        vol += dvol
    return vol / 1000.0


def calc_volarea(patient):
    # compute the areas of all images. areas:

    areas = calc_areas(patient.predACContours)
    volumes = [calc_volume(area, patient.area_multiplier, patient.dist) for area in areas]
    # find edv and esv
    edv = max(volumes)
    esv = min(volumes)
    # calculate ejection fraction
    ef = (edv - esv) / edv

    # save as instance variables
    patient.edv = edv
    patient.esv = esv
    patient.ef= ef

def regress_vol(resultspath = '/Users/Peadar/Documents/KagglePythonProjects/AML/DataScienceBowl/results.csv'):

    """"
    Method for regressing final kaggle predictions to the provided patient training volumes
    and then updating the predictions based on the learned linear_model the stored learned model is extracted
    so it can be used subsequently at test time when the volumes are unkown
    """
    results = genfromtxt(resultspath, delimiter=',') # get volume predictions and labels from file
    dim = results.shape

    # get X and Y arguments for subsequent regression and resize in order for them to work with fit function
    edv_pred = np.reshape(results[:,3],(dim[0],1))
    esv_pred = np.reshape(results[:,4],(dim[0],1))
    edv_label = np.reshape(results[:,1],(dim[0],1))
    esv_label = np.reshape(results[:,2],(dim[0],1))
    patient = np.reshape(results[:,0],(dim[0],1))

    edv_regr = linear_model.LinearRegression()# create regression object
    edv_regr.fit(edv_pred, edv_label) # fit edv model with data

    edv_regvol = edv_regr.predict(edv_pred)# regress edv volumes

    # repeat for esv
    esv_regr = linear_model.LinearRegression()
    esv_regr.fit(esv_pred, esv_label)
    esv_regvol = esv_regr.predict(esv_pred)

    # concatenate results
    regressed_results = np.concatenate((patient,esv_label, edv_label, edv_regvol, esv_regvol), axis=1)

    # store results in csv file ready for processing

    np.savetxt(
        'regressed_results', # file name
        regressed_results, # array to save
        fmt='%.2f', # format
        delimiter=',', # column delimiter
        newline='\n')  # new line character

    # pickle dump regression models built using the training data, so they can be loaded and used at test time

    with open('/Users/Peadar/Documents/KagglePythonProjects/AML/DataScienceBowl/data/regressionModel_esvVol', 'wb') as f:
        pickle.dump(esv_regvol , f)

    with open('/Users/Peadar/Documents/KagglePythonProjects/AML/DataScienceBowl/data/regressionModel_edvVol', 'wb') as fi:
        pickle.dump(edv_regvol , fi)


if __name__ == "__main__":

    # contains 'train', 'validate', etc
    data_path = '../DataScienceBowl/data'

    labels = np.loadtxt(os.path.join(data_path, 'train.csv'), delimiter=',', skiprows=1)
    label_dict = {}
    for label in labels:
        label_dict[label[0]] = (label[2],label[1])


    # contains '1' (maybe patient '1')
    train_path = os.path.join(data_path, 'train')
    # contains 'sax5', 'sax6', ...
    studies = next(os.walk(train_path))[1]

    results_csv = open('results.csv', 'w')

    for study in studies:
        patient = Patient(os.path.join(train_path, study), study)
        patient._read_all_dicom_images()
        patient.predictContours()
        print 'Processing patient %s...' % patient.name
        try:
            calc_volarea(patient)
            (edv, esv) = label_dict[int(patient.name)]
            results_csv.write('%s,%f,%f,%f,%f\n' % (patient.name, edv, esv, patient.edv, patient.esv))
        except Exception as e:
            print '***ERROR***: Exception %s thrown by patient %s' % (str(e), patient.name)
    results_csv.close()

    regress_vol(resultspath = '/Users/Peadar/Documents/KagglePythonProjects/AML/DataScienceBowl/results.csv') # regress final volumes
