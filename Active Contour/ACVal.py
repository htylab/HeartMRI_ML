from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import active_contour as AC
import sys
sys.path.insert(0, '/Users/Peadar/Documents/KagglePythonProjects/AML/DataScienceBowl/Region of Interest/Stacked autoencoder/')
import stackedAutoencoder
from stackedAutoencoder import crop_ROI
import pdb


"""
    This module runs validation for best parameter selection against the active contour model using predictions from the stacked autoencoder.
    It uses the validation function from within the active_contour module
"""

# ADD YOUR DATA'S LOCAL DATA PATH
LOCALDATAPATH = '/Users/Peadar/Documents/KagglePythonProjects/AML/DataScienceBowl/data/'

# load required data

# binary region of interest square (output from CNN)
roi_large = np.load(LOCALDATAPATH + 'SBXtrainBinaryMask32_large')
roi_small = np.load(LOCALDATAPATH + 'SBXtrainBinaryMask32_small')

# original training image
train_large = np.load(LOCALDATAPATH + 'SBXtrainImage256_large')
train_small = np.load(LOCALDATAPATH + 'SBXtrainImage256_small')

# original training binary contour
mask_large = np.load(LOCALDATAPATH + 'SBXtrainMask256_large')
mask_small = np.load(LOCALDATAPATH + 'SBXtrainMask256_small')

# predictions from SA based on the above data
preds_Large = np.load(LOCALDATAPATH + 'SA_predictions_large')
preds_Small = np.load(LOCALDATAPATH + 'SA_predictions_small')

# crop original image and contour data to match the region of the predictions
train_roi_large =crop_ROI(images=train_large, roi=roi_large, roi_dim=(100,100), newsize=(64, 64))
train_roi_small =crop_ROI(images=train_small, roi=roi_small, roi_dim=(100,100), newsize=(64, 64))

mask_roi_large =crop_ROI(images=mask_large, roi=roi_large, roi_dim=(100,100), newsize=(64, 64))
mask_roi_small =crop_ROI(images=mask_small, roi=roi_small, roi_dim=(100,100), newsize=(64, 64))


show_preds_large = [train_roi_large[i,:,:] + preds_Large[i,:,:] for i in range(np.shape(preds_Large)[0])]
show_preds_small = [train_roi_small[i,:,:] + preds_Small[i,:,:] for i in range(np.shape(preds_Small)[0])]

#pdb.set_trace()

"""
    Trial parameter ranges: alpha1{1, 1.5, 2}, alpha 2{1.5,2,2.5},  alpha 3 = {0, ..., 0.01} steps 0.001

"""

#params are [alpha1 set, alpha 2 set, alpha3 set]
# pred IDs small: 0, 1, 60, 70, 100, 200, 340
# pred IDs LARGE: 0, 5, 100, 250, 340

preds_Small = preds_Small[[0, 1, 60, 70, 100, 200, 340],:,:]
preds_Large = preds_Large[[0, 5, 100, 250, 340],:,:]

train_roi_large = train_roi_large[[0, 5, 100, 250, 340],:,:]
train_roi_small= train_roi_small[[0, 1, 60, 70, 100, 200, 340],:,:]

mask_roi_large = mask_roi_large [[0, 5, 100, 250, 340],:,:]
mask_roi_small = mask_roi_small [[0, 1, 60, 70, 100, 200, 340],:,:]

params_large = [[1, 1.5, 2], [1.5, 2, 2.5], [0, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01]]
params_small = [[1, 1.5, 2], [1.5, 2, 2.5], [0, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01]]

# run the validation function to find best set of parameters from the combinations available in the above list

best_params_Large = AC.ac_val(preds_Large, train_roi_large, mask_roi_large, params_large)
best_params_Small = AC.ac_val(preds_Small, train_roi_small, mask_roi_small, params_small)

best_params_Large.dump(LOCALDATAPATH + 'AC_params_Large')
best_params_Small.dump(LOCALDATAPATH + 'AC_params_Small')