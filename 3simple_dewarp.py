###########################

# This is the script to run for making a warping/astrometric solution to LMIRCam, using data taken
# in Nov and Dev 2016

# from: https://github.com/mwanakijiji/dewarp.git

# parent find_dewarp_soln.py created by E.S., Nov 2016
# child apply_dewarp_soln.py made by E.S., Feb 2017
# edits by E.S., May 2019

# edited for use with AB Aur data Sarah Betti, May 2020

import numpy as np
from astrom_lmircam_soln import dewarp
from astropy.io import fits
import glob
import os

###################### o ########################

# SET THE DEWARP COEFFICIENTS

# dewarp coefficients

## SX (LEFT) UT 2017 Nov 08 -- SINGLE EYE: USE FOR K BAND DATA
#Kx = [[ -1.97674665e+01,   2.26890756e-02,  -1.06483884e-05,   1.33174251e-09],
#      [  1.04269459e+00,  -2.68457747e-05,   2.08519317e-08,  -4.74786541e-12],
#      [ -3.30919802e-05,   9.48855944e-09,  -1.00804780e-11,   3.45894384e-15],
#      [  1.00196745e-08,  -2.58289058e-12,   2.58960909e-15,  -8.74827083e-19]]
#
#Ky = [[  1.05428609e+01,   9.91877631e-01,  -1.30947328e-06,   5.98620664e-09],
#      [ -2.65330464e-02,  -6.14857421e-06,  -1.56796197e-08,   6.61213303e-12],
#      [  1.50777505e-05,  -8.14931285e-09,   2.28968428e-11,  -9.37645995e-15],
#      [ -1.54162134e-09,   5.25556977e-12,  -7.46189515e-15,   3.04540450e-18]]



# DX (RIGHT) 2017 Nov 08 -- SINGLE EYE: USE FOR SINGLE L BAND DATA 
Kx = [[ -1.34669677e+01,   2.25398365e-02,  -7.39846082e-06,  -8.00559920e-11],
      [  1.03267422e+00,  -1.10283816e-05,   5.30280579e-09,  -1.18715846e-12],
      [ -2.60199694e-05,  -3.04570646e-09,   1.12558669e-12,   1.40993647e-15],
      [  8.14712290e-09,   9.36542070e-13,  -4.20847687e-16,  -3.46570596e-19]]

Ky = [[  1.43440109e+01,   9.90752231e-01,  -3.52171557e-06,   7.17391873e-09],
      [ -2.43926351e-02,  -1.76691374e-05,   5.69247088e-09,  -2.86064608e-12],
      [  1.06635297e-05,   8.63408955e-09,  -2.66504801e-12,   1.47775242e-15],
      [ -1.10183664e-10,  -1.67574602e-13,   2.66154718e-16,  -1.13635710e-19]]

# double UT 2016 Nov 22  -- BOTH EYES: USE FOR L BAND DATA
KxD = [[ -4.74621436e+00,   9.99369200e-03,  -4.69741638e-06,   4.11937105e-11],
      [  1.01486148e+00,  -2.84066638e-05,   2.10787962e-08,  -3.90558311e-12],
      [ -1.61139243e-05,   2.24876212e-08,  -2.29864156e-11,   6.59792237e-15],
      [  8.88888428e-09,  -1.03720381e-11,   1.05406782e-14,  -3.06854175e-18]]

KyD = [[  9.19683947e+00,   9.84613002e-01,  -1.28813904e-06,   6.26844974e-09],
      [ -7.28218373e-03,  -1.06359740e-05,   2.43203662e-09,  -1.17977589e-12],
      [  9.48872623e-06,   1.03410741e-08,  -2.38036199e-12,   1.17914143e-15],
      [  3.56510910e-10,  -5.62885797e-13,  -5.67614656e-17,  -4.21794191e-20]]
    
    
###################### o ########################
# CHANGEABLE PARAMETERS 
datadir = '/Users/sbetti/Documents/dissertation_datasets/LMIRCam_Feb13/140213'

# GET ALL FILES 
all_fils = glob.glob(datadir + '/3bkg_sub_1x1/*/*/*.fits')

# random image: this is just to get the shape; image content doesn't matter
hdul = fits.open(datadir + '/1raw/lm_140213_00000.fits')
imagePinholes = hdul[0].data.copy()
imagePinholes = imagePinholes[0,:,:]
# map the coordinates that define the entire image plane
# (transposition due to a coefficient definition change btwn Python and IDL)
dewarp_coords = dewarp.make_dewarp_coordinates(imagePinholes.shape, np.array(Kx).T, np.array(Ky).T) 

###################### o ########################

## START DEWARPING 

# loop through all files and dewarp
for imnum, fil in enumerate(all_fils):
    # get object name and filter
    sep = fil.split('/')
    name = sep[-3]
    filt = sep[-2]
    filname = (sep[-1]).split('final_')[-1]
    print(imnum+1,'/', len(all_fils), filname)
    
    if os.path.exists(datadir + f'/4distortion_correction1x1/{name}/{filt}/distcorr_{filname}'):
        pass
    else:
        if not os.path.exists(datadir + f'/4distortion_correction1x1/{name}/{filt}'):
            os.makedirs(datadir + f'/4distortion_correction1x1/{name}/{filt}')
        
        # grab the pre-dewarp image and header
        imageAsterism, header = fits.getdata(fil, header=True)
        orig_shape = imageAsterism.shape[1]
        
        if imageAsterism.shape[1] < 1000:
            pad_image = np.zeros((len(imageAsterism), 1024, 1024))
            pad_image[:,0:imageAsterism.shape[1],0:imageAsterism.shape[2]] =  imageAsterism
            imageAsterism = pad_image
        print(np.shape(imageAsterism))

        new_image = np.zeros_like(imageAsterism)
        for n in np.arange(len(imageAsterism)):
            imageAsterism_ind = imageAsterism[n,:,:]
            if '(U-D)' in header['NOD']: 
                dewarp_coords = dewarp.make_dewarp_coordinates(imagePinholes.shape, np.array(KxD).T, np.array(KyD).T) 
            else:
                dewarp_coords = dewarp.make_dewarp_coordinates(imagePinholes.shape, np.array(Kx).T, np.array(Ky).T) 

            # dewarp the image
            dewarpedAsterism = dewarp.dewarp_with_precomputed_coords(imageAsterism_ind,dewarp_coords,order=3)
            image = np.squeeze(dewarpedAsterism)
            new_image[n,:,:] = np.squeeze(image)

        # write out
        if orig_shape< 1000:
            new_image = new_image[:,0:511,0:511]
        fits.writeto(datadir + f'/4distortion_correction1x1/{name}/{filt}/distcorr_{filname}',np.squeeze(new_image),header,overwrite=True, output_verify='fix')

###################### o ########################








