'''
Takes raw fits files and converts to data cube.  
Pull sout seeing, sky background counts, FWHM, airmass, filter, and observation time from log and fits files and writes to a .csv file

written by: Sarah Betti 

'''

import numpy as np
from numpy import pi, r_
import pandas as pd

import matplotlib.pyplot as plt
from astropy.io import fits

import glob
import datetime
import os

from astropy.stats import sigma_clip
from datetime import timedelta
from photutils import source_properties, detect_sources, deblend_sources

from scipy import optimize
import scipy.optimize as opt

###################### o ########################

def twoD_GaussianScaledAmp(vals, xo, yo, sigma_x, sigma_y, amplitude, offset):
    """Function to fit, returns 2D gaussian function as 1D array"""
    xo = float(xo)
    yo = float(yo)    
    x,y = vals
    g = offset + amplitude*np.exp( - (((x-xo)**2)/(2*sigma_x**2) + ((y-yo)**2)/(2*sigma_y**2)))
    return g.ravel()

def getFWHM_GaussianFitScaledAmp(img):
    """Get FWHM(x,y) of a blob by 2D gaussian fitting
    Parameter:
        img - image as numpy array
    Returns: 
        FWHMs in pixels, along x and y axes.
    """
    
    if len(img) > 0:
        x = np.linspace(0, img.shape[1], img.shape[1])
        y = np.linspace(0, img.shape[0], img.shape[0])
        x, y = np.meshgrid(x, y)
        #Parameters: xpos, ypos, sigmaX, sigmaY, amp, baseline
        initial_guess = (img.shape[1]/2,img.shape[0]/2,10,10,1,0)
        # subtract background and rescale image into [0,1], with floor clipping
        bg = np.percentile(img,5)
        img_scaled = np.clip((img - bg) / (img.max() - bg),0,1)
        popt, pcov = opt.curve_fit(twoD_GaussianScaledAmp, (x, y), 
                                   img_scaled.ravel(), p0=initial_guess,
                                   bounds = ((img.shape[1]*0.4, img.shape[0]*0.4, 1, 1, 0.5, -0.1),
                                         (img.shape[1]*0.6, img.shape[0]*0.6, img.shape[1]/2, img.shape[0]/2, 1.5, 0.5)))
        xcenter, ycenter, sigmaX, sigmaY, amp, offset = popt[0], popt[1], popt[2], popt[3], popt[4], popt[5]
        FWHM_x = np.abs(4*sigmaX*np.sqrt(-0.5*np.log(0.5)))
        FWHM_y = np.abs(4*sigmaY*np.sqrt(-0.5*np.log(0.5)))
        return FWHM_x, FWHM_y, xcenter, ycenter
    else:
        return np.nan, np.nan, 0, 0
    
###################### o ########################
# CHANGEABLE PARAMETERS 

# date of data
date = '140213'
filt = 'L'
# data directory of raw data 
datadir = '/Users/sbetti/Documents/dissertation_datasets/LMIRCam_Feb13/140213/1raw'
# glob all data
data = np.sort(glob.glob(f'{datadir}/*.fits'))

# make directory to saw cubed fits files
savedir = '/Users/sbetti/Documents/dissertation_datasets/LMIRCam_Feb13/140213/2cubes'
if not os.path.exists(savedir):
    os.makedirs(savedir)
    
# open log
log = '/Users/sbetti/Documents/dissertation_datasets/LMIRCam_Feb13/LBTI_13Feb2014.xlsx'

# read log using pandas
df = pd.read_excel(log, sheet_name='LMIRCam', skiprows=4) 
df = df.dropna(subset=['File'])

fils = df['File'].to_list()
nod = df['Nod info'].to_list()
seq = df['Seq Tot'].to_list()

###################### o ########################
# START MAKING CUBES

# set up lists of information that will be saved to .csv file
seeing = []
airmass = []
filt = []
time = []
obj = []
sky_bkg = []
fwhmx = []
fwhmy = []

# run through each row of log and combine all frames which are in the same nod sequence. 
for ind, idx in enumerate(fils):
    # split log file name at the dash
    nums = idx.split('-')
    #find min and max value in nod sequence
    nod_info = nod[ind]
    seq_tot = seq[ind]
    minn = nums[0]
    maxx = nums[1]
    
    # add an extra zero if the number is too small
    if len(minn) <4:
        minn = '0' + minn
    if len(maxx) < 4:
        maxx = '0' + maxx
        
    print(minn, maxx)
    # only get our data! (other data on log that we don't want to loop through)
    ##### CHANGE VALUE TO BE THE LAST ROW OF OUR DATA ### 
    if int(maxx) <= 7640 : 
        # find images between min and max value
        ind_min = np.where(data == f'{datadir}/lm_{date}_0{minn}.fits')[0][0]
        ind_max = np.where(data == f'{datadir}/lm_{date}_0{maxx}.fits')[0][0]

        print('min/max index and nod', ind_min, ind_max, nod_info)
        
        #grab all frames between those values
        subset = data[ind_min:ind_max+1]

        #loop through subset, append into a cube, and grad info from header for csv file
        cube = np.zeros((len(subset),  1024, 1024))
        for i in np.arange(len(subset)):
            #grad seeing, airmass, filters, observation time, object name information
            raw_data = fits.getdata(subset[i])[0,:,:]
            hdr = fits.getheader(subset[i])
            seeing.append(hdr['SEEING'])
            airmass.append(hdr['LBT_AIRM'])
            xf = hdr['TIME-OBS']
            delta = timedelta(hours=int(xf.split(':')[0]), minutes=int(xf.split(':')[1]), seconds=float(xf.split(':')[2]))
            time.append(delta.total_seconds()/3600)
            
            med_val = np.median(raw_data[:,500::])
            sky_bkg.append(med_val)
            
            if filt != 'L':
                filters = np.array([hdr['LMIR_FW3'],  hdr['LMIR_FW4']])
                filt_ind = np.where((filters == 'Kshort') | (filters == 'H2O-Ice2'))
                if len(filt_ind[0]) == 0:
                    actual_filter = 'H2O-Ice1'
                else:
                    actual_filter = filters[filt_ind][0]
            else:
                actual_filter = 'L'
            filt.append(actual_filter)
            obj.append(hdr['OBJNAME'])
            
            # use photutils to detect the source in the file in order to calculate the FWHM.  Might need to change the threshold pixel count (currently 60000) to detect the source.  this only sometimes works...
            segm = detect_sources(raw_data, 60000, npixels=100)
            if segm is not None:
                # find the FWHM of the source
                cat = source_properties(raw_data, segm)
                xind_cen, yind_cen = int(cat[0].xcentroid.value), int(cat[0].ycentroid.value) 
                raw_data = raw_data[yind_cen-50:yind_cen+50, xind_cen-50:xind_cen+50]
                
                FWHMx, FWHMy, xx,yy = getFWHM_GaussianFitScaledAmp(raw_data) 
                if i%20 ==0:
                    print(i, xind_cen, yind_cen,FWHMx, FWHMy*0.011)

                fwhmx.append(FWHMx*0.011)
                fwhmy.append(FWHMy*0.011)
            else:
                # if the source is not found, just append NaN
                fwhmx.append(np.nan)
                fwhmy.append(np.nan)

            #append frame to cube if the cube does not already exist.
            if os.path.exists(savedir + f'/lm_150104_{minn}_{maxx}.fits'):
                print(f'file exists: {i}/{len(subset)}', end='\r')
                pass
            
            else:
                print(f'getting data: {i}/{len(subset)}', end='\r')
                im = fits.getdata(subset[i])
                if np.shape(im)[1] == 1024:
                    cube[i, :, :] = im
                else:
                    cube[i, 0:511, 0:511] = im
                    
        #save cube to file if cube does not already exist
        print()
        if os.path.exists(f'{savedir}/lm_150104_{minn}_{maxx}.fits'):
            pass
        else:
            print('saving cube')
            hdr.append(('COMMENT', f"cube written on: {datetime.datetime.now().strftime('%Y-%m-%d')}"),
                      end=True)
            hdr.append(('COMMENT', f'files between {minn} and {maxx}'), end=True)
            hdr.append(('NOD', f'{nod_info}'), end=True)


            fits.writeto(savedir + f'/lm_150104_{minn}_{maxx}.fits', cube, hdr, output_verify = 'fix',overwrite=True)


#save information to csv files
d = {'seeing':seeing, 'airmass':airmass, 'filter':filt, 'time':time, 'object':obj, 
    'sky_bkg':sky_bkg, 'fwhmx':fwhmx, 'fwhmy':fwhmy}
df = pd.DataFrame(data=d) 
df.to_csv('raw_data_information_14Feb13.csv', index=False)

###################### o ########################