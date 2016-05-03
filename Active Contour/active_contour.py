######################################################################
# This routine is an implementation of the active contour model
# following Chan, T. and Vese, L.(2001): Active Contours without Edges
######################################################################
from __future__ import division  # prevents pesky non-float division
import numpy as np
import skfmm
from matplotlib import pyplot as plt
import copy
import pdb  # debugger
import time
import itertools
import sklearn as sk
from sklearn.metrics import f1_score

def evolve_contour(lv, roi, deltaT=0.1, alpha1=1, alpha2=1, alpha3=0.1, eps=1 / np.pi, eta=1e-5, n_reinit=10, n_max = 100):
    """
    evolve_contour performs an active contour algorithm on a level set curve,
    specifically on the zero level of a signed distance function. The zero-level
    is represented by discrete points on a grid.
    Each point is evolved in the direction where some energy function decreases most
    :param lv: 64 x 64 binary array predicted
    :param roi: 64 x 64 region of interest (gray scale) actual image
    :param deltaT: the step size in time
    :param alpha1, alpha2, alpha3: parameters for energy components
    :param eps: parameter for the function approximating the delta_0 function
    :return: evolved level set function
    """
    try:
        # error handling: inputs must be 64 x 64
        # if np.shape(lv) != (64, 64) | np.shape(roi)!= (64, 64):
        #    raise TypeError("lv and roi must be 64 x 64")
        if np.unique(lv).size > 2:
            raise TypeError("lv must be binary")

        # Initialize phi as a signed distance function which looks like a ice cream cone. It
        # computes for each point in the 64 x 64 region of interest its distance to
        # the closest contour point in the binary image LV.

        phi = copy.deepcopy(lv)
        phi[phi == 1] = -1
        phi[phi == 0] = 1
        phi = 10 * skfmm.distance(phi)  # now phi is a signed distance function which is negative inside the contour and

        # positive outside
        # we will store the initialization of phi again in an extra variable because
        # we have to recall it in every evolution step
        phi0 = copy.deepcopy(phi)
        # START ENERGY OPTIMIZATION
        convergence = False
        cIter = 1
        # create imshow object. in each iteration update that figure

        while not convergence:

            # 1. compute all finite differences
            # this will be done by the divergence function, so forget about this step
            # if cIter == 521:
            #    pdb.set_trace()
            # 2. compute averages inside and outside contour
            # 2a. average outside
            c1 = np.sum(roi * heavyside(phi)) / (phi[phi >= 0].size + 0.00000001)
            c2 = np.sum(roi * heavyside(-phi)) / (phi[phi < 0].size + 0.00000001)


            """
            # 2. compute averages inside and outside contour
            # 2a. determine which pixels are inside the contour and which are outside
            # a pixel at (i,j) is inside the contour if phi(i,j) < 0
            # Create a mask which is True if phi<=0 and False else
            phi_mask = copy.deepcopy(-phi) # now phi is 1 inside contour and -1 outside
            phi_mask[phi_mask < 0] = 0     # now phi is 1 inside contour and 0 outside
            phi_mask = np.ma.make_mask(phi_mask) # create mask

            # 2b. Compute average roi pixel value inside the contour (i.e. where phi <= 0)
            # Wherever mask == True -> apply_mask_to_roi = 1, wherever mask == False -> apply_mask_to_roi = 0
            avg_roi_inside = np.mean(roi[phi_mask])
            avg_roi_outside = np.mean(roi[-phi_mask])
            c1 = avg_roi_outside
            c2 = avg_roi_inside
            """

            # pdb.set_trace()

            # 3. Compute divergence
            old_phi = copy.deepcopy(phi)
            div = get_div(phi)

            # 4. Evolve contour
            # pdb.set_trace()
            #force = delta_eps(phi, eps) * (alpha1 * div + (alpha2 * np.power(roi - c2, 2))
            #                                             - (alpha2 * np.power(roi - c1, 2))
            #                                                    - (2 * alpha3 * (phi - phi0)))
            # pdb.set_trace()
            force = alpha1 * div + alpha2 * np.power(roi - c2, 2) - alpha2 * np.power(roi - c1, 2) - 2 * alpha3 * (phi - phi0)

            phi += deltaT * force

            # 6. stop if phi has converged or maximum number of iterations has been reached
            if (np.linalg.norm(phi - old_phi, 'fro') < eta) | (cIter == n_max):
                convergence = True
                #print("converged")
                # Draw final level set function
                contour = copy.deepcopy(phi)
                contour[contour >= 0] = 0
                contour[contour < 0] = 1

            elif cIter % n_reinit == 0:  # Check if we have to reinitialize phi
                # reinitialize by figuring out where phi is neg. and where pos -> define intermediate contour as points
                # where phi is non-positive and reinitialize as signed distance mapping
                phiSmallerZero = phi <= 0  # is an array of bools with True where phi>=0 and False else
                intermediate_contour = phiSmallerZero.astype(int)  # is a binary array that has 1 where phi<=0 and
                # and 0 else
                intermediate_contour[intermediate_contour == 1] = -1  # skfmm.distance wants the inner part of the contour
                # to be -1...
                intermediate_contour[intermediate_contour == 0] = 1  # ...and the outer part +1
                phi = skfmm.distance(intermediate_contour)  # reinitialize phi as signed distance function
                cIter += 1
                #print(cIter)
            else:
                cIter += 1
                #print("cIter", cIter)
                # draw contour: update the pixels and then draw the figure
                # pdb.set_trace()
                if cIter % n_max == 0:
                    contour = copy.deepcopy(phi)
                    contour[contour > 0] = 0
                    contour[contour < 0] = 1
                    contour = contour + roi
        return contour

    except:
        print "Returning LV as evolution failed."
        return lv




def hessian(x):
    """
    :param x: numpy array
       - x : ndarray
    :return: hessian of x
           hessian[0,0] = deriv_yy
           hessian[0,1] = deriv_yx
           hessian[1,0] = deriv_xy
           hessian[1,1] = deriv_xx
    """
    # compute the hessian and lay it out in a format which is sensible
    x_grad = np.gradient(x)
    h = np.empty((x.ndim, x.ndim) + x.shape, dtype=x.dtype)
    for k, grad_k in enumerate(x_grad):
        # iterate over dimensions
        # apply gradient again to every component of the first derivative.
        tmp_grad = np.gradient(grad_k)
        for l, grad_kl in enumerate(tmp_grad):
            h[k, l, :, :] = grad_kl
    return h


def get_div(x):
    """
    :param x: numpy array
    :return: divergence of x
    """
    # compute matrices of derivatives in x- and y- direction
    g = np.gradient(x)
    dy = g[0]
    dx = g[1]
    # compute matrix of norm of gradient
    gnorm = np.sqrt(np.power(dy, 2) + np.power(dx, 2))
    # compute hessian
    h = hessian(x)
    dyy = h[0, 0]
    dyx = h[0, 1]
    dxx = h[1, 1]
    # compute divergence
    divisor = np.power((np.power(dx, 2) + np.power(dy, 2)), 1.5)
    divisor[divisor == 0] = 0.00000001
    dividend = dxx * np.power(dy, 2) - 2 * dx * dy * dyx + dyy * np.power(dx, 2)
    div = dividend / divisor

    # Chen and Vese conduct additionally the following steps
    div = div * np.power(np.power(dx, 2) + np.power(dy, 2), 0.5) / (np.max(div * np.power(np.power(dx, 2) + np.power(dy, 2), 0.5)))
    div = div / np.max(div)
    return div


def delta_eps(x, eps):
    return eps / (np.pi * (np.power(eps, 2) + np.power(x, 2)))

def heavyside(phi):
    """
    Compute smoothed Heavyside function of numpy array
    :param phi: 64 x 64 numpy array (the level set function phi from above)
    :return: smoothed Heavyside function: H = 1 where phi > Epsilon
                                          H = smooths down from 1 to 0, -Epsilon <= phi <= Epsilon
                                          H = 0 where phi < -Epsilon
    """
    # pdb.set_trace()

    Epsilon = 1e-5
    H = copy.deepcopy(phi)
    H[H > Epsilon] = 1
    H[H < -Epsilon] = 0

    # create mask: True where -Epsilon < phi < Epsilon, False else
    if np.any(phi[np.abs(phi) < Epsilon]):
        m2 = copy.deepcopy(phi)
        m2[m2 > Epsilon] = 0
        m2[m2 < -Epsilon] = 0
        m2[m2 != 0] = 1
        mask = np.ma.make_mask(m2)
        H[mask] = 1/2 * (1 + H[mask]/Epsilon + 1/np.pi * np.sin(np.pi * H[mask]/Epsilon))

    return H



def ac_val(contour_preds, roi_images, contour_labels, trial_params):

    """
    Trial parameter ranges: alpha1{1, 1.5, 2}, alpha 2{1.5,2,2.5},  alpha 3 = {0, ..., 0.01} steps 0.001
    This method gets all combinations of trail parameter across 3 tuning variables alpha 1-3.
    It then passes each contour_pred image through the AC for each set of parameter combinations,
    recording the f1 score in the process.

    The parameters with the best f1_score are selected as the best for use at test time for the AC during prediction.

    """

    dim = roi_images.shape # get dimension of image data to loop through
    combs_params = np.array(list(itertools.product(*trial_params))) #g ets all combinations of params for each validation step

    # set placeholders for the prediction image storage and the prediction error storage
    error = []

    # collapse the true LV contour for use when computing the f1 score vrs the collapsed predictions
    contour_labels_col = np.reshape(contour_labels, (dim[0],(dim[1]* dim[2])))

    # loop through all the parameter combinations
    for p in range(0, len(combs_params)):
        pred_ACs = []
        print '##########------Paramaeter combo %d --------- ################,' % p
        # loop through each image and get a prediction by evolving the contour
        for c in range(0, dim[0]):

            print 'Processing image %d.....,' % c

            current_pred = evolve_contour(lv = contour_preds[c], roi = roi_images[c], deltaT=0.1,
                                 alpha1=combs_params[p,0], alpha2=combs_params[p,1], alpha3=combs_params[p,2],
                                 eps=1 / np.pi, eta=1e-5, n_reinit=10, n_max = 1000)

            # collect ACs
            pred_ACs.append(current_pred)

        # collapse the new predictions for use in f1 score calculation

        pred_ACs_col = np.reshape(pred_ACs, (dim[0],(dim[1]* dim[2])))

        # calcualte error
        error.append(f1_score(contour_labels_col, pred_ACs_col)) # F1 = 2 * (precision * recall) / (precision + recall), # check if average or not

    min_cost = np.where(error == np.min(error)) # gets index of best parameters based on minimum error

    best_params = combs_params[min_cost] # singleton array of best parameters selected

    return best_params


