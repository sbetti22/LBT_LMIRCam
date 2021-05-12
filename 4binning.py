'''
bin the data 2x2

written by: Sarah Betti - May 15, 2020

'''

import numpy as np
from astropy.io import fits
import glob
import datetime

###################### o ########################

def rebin(a, shape):
    sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
    return a.reshape(sh).mean(-1).mean(1)

###################### o ########################
# CHANGEABLE PARAMETERS

# data directory
datadir = '/Users/sbetti/Documents/dissertation_datasets/LMIRCam_Feb13/140213'

# grab all data
all_data = np.sort(glob.glob(datadir + '/4distortion_correction1x1/HD39925/Lprime/distcorr_lm_140213*.fits'))


###################### o ########################
# START BINNING

# loop through data and bin 2x2
for imnum, data in enumerate(all_data):
    sep = data.split('/')
    name = sep[-3]
    filt = sep[-2]
    filname = sep[-1]
    print(imnum+1,'/', len(all_data), name, filt, filname)
    
    a = fits.open(data, memmap=False)[0]
    im = a.data
    hdr = a.header
    
    im_binned = np.zeros((len(im), im.shape[1]//2, im.shape[2]//2))
    for i in np.arange(len(im)):
        frame = im[i,:,:]
        binned_frame = rebin(frame, [im.shape[1]//2, im.shape[2]//2])
        im_binned[i,:,:] = binned_frame
    hdr.append(('COMMENT', f"binned 2x2 (0.022arcsec/pixel) on {datetime.datetime.now().strftime('%Y-%m-%d')}"), end=True)
    
    hdr['PIXSCALE'] = 0.022
    print(f'writing to: {datadir}/5binned2x2/{name}/{filt}/bin_{filname}')
    fits.writeto(datadir + f'/5binned2x2/{name}/{filt}/bin_{filname}', im_binned, header=hdr, overwrite=True)

###################### o ########################







