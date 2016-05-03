
import theano.tensor as T
import numpy as np
from theano.tensor.shared_randomstreams import RandomStreams
import theano
from scipy import misc
import matplotlib.pyplot as plt
import pickle
import os
import re
import dicom
from LoadData import crop_resize
import sys
sys.path.insert(0, '/home/odyss/Desktop/dsb/AML/DataScienceBowl/convolution/')
import LeNet
from LeNet import predict as CNNpred
sys.path.insert(0, '/home/odyss/Desktop/dsb/AML/DataScienceBowl/Region of Interest/Stacked autoencoder/')
from stackedAutoencoder import predict_sa as SApred
from stackedAutoencoder import crop_ROI
from stackedAutoencoder import SA
from sklearn import linear_model
import time
import pdb

sys.path.insert(0, '/home/odyss/Desktop/dsb/AML/DataScienceBowl/Active Contour/')
import active_contour as AC
from numpy import genfromtxt




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

        self.big_images = []      # collect the slice IDs for big images
        self.small_images = []    # collect the slice IDs for small images
        big_small_threshold = -71 # this is the slice location threshold. all images with slice location > -71 are big
                                  # all <= -71 are small

        for cslice in slices:
            cslice_files = next(os.walk(os.path.join(train_path, 'sax_%d' % cslice)))[2]
            # cslice_files contains a list ['IM-4557-0021.dcm', 'IM-4557-0026.dcm',...]
            offset = None
            for cslice_file in cslice_files:
                #m = re.match('IM-(\d{4,})-(\d{4})\.dcm', cslice_file)
                m = re.match('IM-(\d*)-(\d*)\.dcm', cslice_file)
                if m is not None:
                    if first:
                        times.append(int(m.group(2)))
                    if offset is None:
                        #offset = int(m.group(1))
                        offset = m.group(1)
            first = False
            slices_map[cslice] = offset

        # some instance variables
        self.directory = train_path
        self.time = sorted(times)
        self.slices = sorted(slices)
        self.slices_map = slices_map
        Patient.patient_count += 1
        self.name = study
        self.study = study
        self.images = []
        self.predROIs = []
        self.imagesROIs = []
        self.predSAContours = []
        self.predACContours = []

    # returns the name of a file of a specific slice on specific time
    def _filename(self, cslice, time):

        return os.path.join(self.directory, 'sax_%d' % cslice,
        'IM-%s-%04d.dcm' % (self.slices_map[cslice], time))

    # read one single dicom file

    def _read_dicom_image(self, filename):

        d = dicom.read_file(filename)
        sl = d.SliceLocation
        img = d.pixel_array.astype('float')

        return img

    # loads all patient's images. saves at instance variable self.images
    # images: [slice_height x time x 64 x 64]
    # also calculates slice thickness and area_multiplier (one for all images)
    def _read_all_dicom_images(self):
        """
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

        """

        self.big_slices = []
        self.small_slices = []

        for dirs in os.walk(os.path.join(train_path, study)):

            saxno = re.search('sax_(\d+)', dirs[0])

            if saxno is not None:

                dicom_file_list = dirs[2]
                if re.search('\.dcm', dicom_file_list[0]):
                    file_path = os.path.join(dirs[0], dicom_file_list[0])

                else:
                    file_path = os.path.join(dirs[0], dicom_file_list[1])
                dicom_file = dicom.read_file(file_path)
                if dicom_file.SliceLocation > -71:
                    self.big_slices.append(int(saxno.group(1)))
                else:
                    self.small_slices.append(int(saxno.group(1)))
        """
        self.images = np.array([[self._read_dicom_image(self._filename(d, i))
                                 for i in self.time]
                                for d in self.slices])
        """

        self.big_images = [[self._read_dicom_image(self._filename(d, i)) for i in self.time] for d in self.big_slices]


        self.small_images =[[self._read_dicom_image(self._filename(d, i)) for i in self.time] for d in self.small_slices]

        # sets dist equal to the two first slices distance (subtracting first
        # from second) or equal to first slice's thickness. I THINK the second
        # logic is better
        #
        # maybe set dist = d1.SliceThickness
        #self.dist = dist
        #self.area_multiplier = x * y



    def predictContours(self):

        '''
        Method for pushing images loaded by _read_all_dicom_images through full learned network
        to predict the resulting contour
           STEPS: 1) Feedforward through CNN, followed by through SA, followed by calling AC model, giving final result

        :return:
        '''

        #images are slice * time * height * width
        #############################################
        # PREDICT ROI MASKS
        # IMPORTANT: Change fine_tuned_params_path file to fine_tune_paramsX_big.pickle and .._small.pickle respectively

        # self.big_images and self.small_images are both lists of lists of arrays
        b = len(self.big_images)
        s = len(self.small_images)
        # CNN should return list of arrays
        self.pred_big_ROIs = [CNNpred(inputimages = self.big_images[s], batch_size=1,
                                      fine_tuned_params_path = '/home/odyss/Desktop/dsb/AML/DataScienceBowl/data/fine_tune_paramsXlarge.pickle')
                              for s in range(0, len(self.big_slices))]
        self.pred_small_ROIs = [CNNpred(inputimages = self.small_images[s], batch_size=1,
                                        fine_tuned_params_path = '/home/odyss/Desktop/dsb/AML/DataScienceBowl/data/fine_tune_paramsXsmall.pickle')
                                for s in range(0, len(self.small_slices))]

        """
        self.predROIs = np.array([CNNpred(inputimages = self.images[s,:], batch_size=1,
                                          fine_tuned_params_path = '/Users/mh/AML/DataScienceBowl/data/fine_tune_paramsXnew.pickle')
                                  for s in range(0, len(self.slices))])
        """
        #############################################
        # CROP IMAGES TO ROIs
        # crop ROI should return list of arrays
        self.imagesROIs_big = [crop_ROI(images=self.big_images[s], roi=self.pred_big_ROIs[s],
                                        roi_dim=(100,100), newsize=(64, 64))
                                        for s in range(0, len(self.big_slices))]
        self.imagesROIs_small = [crop_ROI(images=self.small_images[s], roi=self.pred_small_ROIs[s],
                                          roi_dim=(100,100), newsize=(64, 64))
                                        for s in range(0, len(self.small_slices))]

        """
        self.imagesROIs = np.array([crop_ROI(images=self.images[s,:], roi=self.predROIs[s,:],
                                             roi_dim=(100,100), newsize=(64, 64))
                              for s in range(0, len(self.slices))])
        """
        #############################################
        # PREDICT CONTOUR USING SA
        # IMPORTANT: Change trained_SA_path to ../SA_Xmodel_big and ../SA_Xmodel_small, respectively


        self.predSAbigContours = np.array([SApred(self.imagesROIs_big[s],
                                               trained_SA_path ='/home/odyss/Desktop/dsb/AML/DataScienceBowl/data/SA_RLUmodel_large')
                                        for s in range(0, len(self.big_slices))])
        self.predSAsmallContours = np.array([SApred(self.imagesROIs_small[s],
                                               trained_SA_path ='/home/odyss/Desktop/dsb/AML/DataScienceBowl/data/SA_RLUmodel_small')
                                        for s in range(0, len(self.small_slices))])

        """
        self.predSAContours = np.array([SApred(self.imagesROIs[s,:],
                                               trained_SA_path ='/Users/mh/AML/DataScienceBowl/data/SA_Xmodel')
                                        for s in range(0, len(self.slices))])
        """
        #############################################
        # ACTIVE CONTOUR

        self.predACContours_big = np.array([[AC.evolve_contour(lv = self.predSAbigContours[s][t], roi=self.imagesROIs_big[s][t]
                                                               , alpha1=2, alpha2=1.5, alpha3=0.002)
                                         for t in range(0, len(self.time))] for s in range(0, len(self.big_slices))])
        self.predACContours_small = np.array([[AC.evolve_contour(lv = self.predSAsmallContours[s][t], roi=self.imagesROIs_small[s][t]
                                                                 , alpha1=2, alpha2=1.5, alpha3=0.007)
                                         for t in range(0, len(self.time))] for s in range(0, len(self.small_slices))])

        """
        self.predACContours = np.array([[AC.evolve_contour(lv = self.predSAContours[s,t], roi=self.imagesROIs[s,t])
                                         for t in range(0, len(self.time))] for s in range(0, len(self.slices))])
        """

    def calc_areas(self):
        """
        if (self.predACContours_small is None) | (self.predACContours_big is None):
            print("First pass images through Active Contour to get a prediciton. No predictions for lv found.")
            return
        elif (len(np.shape(self.predACContours_small) != 4)) | (len(np.shape(self.predACContours_big) != 4)):
            print("The lv predictions must be in the shape (num of slices, num of time steps, height, width")
            return
        """

        # Specify number of slices and time steps among the group of small and big images
        b = len(self.predACContours_big)
        s = len(self.predACContours_small)

        if b > 0:
            [l_big, timesb, _, _] = np.shape(self.predACContours_big)
        else:
            l_big = 0
            timesb = 0

        if s > 0:
            [l_small, timess, _, _] = np.shape(self.predACContours_small)
        else:
            l_small = 0
            timess = 0

        if timesb == 0:
            times = timess
        elif timesb != 0:
            times = timesb
        else:
            times = 0

        # Create time dictionaries that map time to the total area at that time for
        # ... small images
        areas_small = [(t, 0) for t in range(times)]
        areas_small = dict(areas_small)
        # ... big images
        areas_big = [(t, 0) for t in range(times)]
        areas_big = dict(areas_big)

        # Compute areas at each time step for both small and big slices
        for t in range(times):
            # at time step t, compute the areas of slices in...
            for hb in range(l_big):
                # ... big images
                areas_big[t] += np.count_nonzero(self.predACContours_big[hb][t])
            for hs in range(l_small):
                # ... small images
                areas_small[t] += np.count_nonzero(self.predACContours_small[hs][t])

        # Find the indices in the time dictionaries that correspond to smallest area...
        # ... among small images

        if len(areas_small) > 0:
            min_area_small_idx = min(areas_small, key=areas_small.get)
            max_area_small_idx = max(areas_small, key=areas_small.get)
        # ... among big images
        if len(areas_big) > 0:
            min_area_big_idx = min(areas_big, key=areas_big.get)
            max_area_big_idx = max(areas_big, key=areas_big.get)

        # Find the indices in the time dictionaries that correspond to smallest area...
        # ... among small images

        # ... among big images


        # Add smallest areas among small and big images together -> total area of slices at end systole
        #self.end_systole_area = areas_big[min_area_big_idx] + areas_small[min_area_small_idx]
        # Add biggest areas among small and big images together -> total area of slices at end diastole
        #self.end_diastole_area = areas_big[max_area_big_idx] + areas_small[max_area_small_idx]

        areas_total = []
        for t in range(times):
            areas_total.append(areas_small[t] + areas_big[t])

        self.end_systole_area = np.min(areas_total)
        self.end_diastole_area = np.max(areas_total)

        return [self.end_systole_area, self.end_diastole_area]


# the single pipeline area and volume cals are depreciated due to added complexity of two way
"""
def calc_areas(images):
    # images are circular binary masks
    (slice_locations, times, _, _) = images.shape
    areas = [{}, t for t in range(times)]
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
"""
"""
def calc_volarea(patient):
    # compute the areas of all images. areas:
    areas_big = calc_areas(patient.predACContours_big)

    areas_big = calc_areas(patient.predACContours_big)
    areas_small = calc_areas(patient.predACContours_small)
    volumes = [calc_volume(area, patient.area_multiplier, patient.dist) for area in areas]
    volumes_big = [calc_volume(area, patient.area_multiplier, patient.dist) for area in areas_big]
    volumes_small = [calc_volume(area, patient.area_multiplier, patient.dist) for area in areas_small]
    # find edv and esv
    big_edv = max(volumes_big)
    small_edv = max(volumes_small)
    big_esv = min(volumes_big)
    small_esv = min(volumes_small)
    edv = max(volumes)
    esv = min(volumes)
    # calculate ejection fraction
    big_ef = (big_edv - big_esv) / big_edv
    small_ef = (small_edv - small_esv) / small_edv
    # save as instance variables
    patient.big_edv = big_edv
    patient.small_esv = small_esv
    patient.big_ef= big_ef
    patient.small_ef= small_ef

    patient.edArea = big_edv
    patient.esArea = areas_small

 """


def regress_vol(resultspath = '/home/odyss/Desktop/dsb/AML/DataScienceBowl/data/results.csv'):

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

    edv_regr = linear_model.LinearRegression() # create regression object
    edv_regr.fit(edv_pred, edv_label) # fit edv model with data

    edv_regvol = edv_regr.predict(edv_pred) # regress edv volumes

    # repeat for esv
    esv_regr = linear_model.LinearRegression()
    esv_regr.fit(esv_pred, esv_label)
    esv_regvol = esv_regr.predict(esv_pred)

    # concatenate results
    regressed_results = np.concatenate((patient,edv_label, esv_label, edv_regvol, esv_regvol), axis=1)

    # store results in csv file ready for processing
    np.savetxt(
        'regressed_results', # file name
        regressed_results, # array to save
        fmt='%.2f', # format
        delimiter=',', # column delimiter
        newline='\n')  # new line character

    # pickle dump regression models built using the training data, so they can be loaded and used at test time
    with open('/home/odyss/Desktop/dsb/AML/DataScienceBowl/data/regressionModel_esvVol', 'wb') as f:
        pickle.dump(esv_regvol , f)

    with open('/home/odyss/Desktop/dsb/AML/DataScienceBowl/data/regressionModel_edvVol', 'wb') as fi:
        pickle.dump(edv_regvol , fi)





if __name__ == "__main__":


    # contains 'train', 'validate', etc
    data_path = '/home/odyss/Desktop/dsb/AML/DataScienceBowl/data/'

    labels = np.loadtxt(os.path.join(data_path, 'train.csv'), delimiter=',', skiprows=1)
    label_dict = {}
    for label in labels:
        label_dict[label[0]] = (label[2],label[1])


    # contains '1' (maybe patient '1')
    train_path = os.path.join(data_path, 'train')
    # contains 'sax5', 'sax6', ...
    studies = next(os.walk(train_path))[1]

    results_csv = open('/home/odyss/Desktop/dsb/AML/DataScienceBowl/data/results_401_500.csv', 'w')

    for study in studies:

        print "study:" + study
        patient = Patient(os.path.join(train_path, study), study)
        print 'Processing patient %s...' % patient.name

        patient._read_all_dicom_images()

        print 'Predicting contours...'

        patient.predictContours()

        print 'predict Contour done'

        print 'Calculating volume...'

        try:
            [ESA, EDA] = patient.calc_areas()
            # IMPORTANT: ESA and EDA must be passed into the volume regression
            (edv, esv) = label_dict[int(patient.name)]
            # results_csv.write('%s,%f,%f,%f,%f\n' % (patient.name, edv, esv, patient.edv, patient.esv))
            results_csv.write('%s,%f,%f,%f,%f\n' % (patient.name, edv, esv, EDA, ESA))
        except Exception as e:
            print '***ERROR***: Exception %s thrown by patient %s' % (str(e), patient.name)
        print 'Done'

    results_csv.close()

    #regress_vol(resultspath = '/home/odyss/Desktop/dsb/AML/DataScienceBowl/data/results_final.csv') # regress final volumes
