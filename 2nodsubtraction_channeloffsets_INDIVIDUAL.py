'''
reduce individual data cubes and plot the data reduction steps.  

If file is not needed if you are using file #2 to reduce data.

1. background subtract raw data in up/down nod stacks
2. removes outlier pixels
3. removes column variation
4. stacks cubes and saves reduced stacked data.  

written by: Sarah Betti -  May 15, 2020


'''
import numpy as np

import matplotlib.pyplot as plt
import matplotlib

from astropy.io import fits
from astropy.stats import sigma_clip
from astropy.visualization import ZScaleInterval
interval = ZScaleInterval()

import glob
import datetime

import os
import warnings
warnings.filterwarnings('ignore', category=UserWarning, append=True)

###################### o ########################

def find_outlier_pixels(data,tolerance=3,worry_about_edges=True):
    #This function finds the hot or dead pixels in a 2D dataset. 
    #tolerance is the number of standard deviations used to cutoff the hot pixels
    #If you want to ignore the edges and greatly speed up the code, then set
    #worry_about_edges to False.
    #
    #
    #The function returns a list of hot pixels and also an image with with hot pixels removed

    from scipy.ndimage import median_filter
    blurred = median_filter(data, size=3)
    difference = data - blurred
    threshold = 8*np.std(difference)

    #find the hot pixels, but ignore the edges
    hot_pixels = np.nonzero((np.abs(difference[1:-1,1:-1])>threshold) )
    hot_pixels = np.array(hot_pixels) + 1 #because we ignored the first row and first column

    fixed_image = np.copy(data) #This is the image with the hot pixels removed
    for y,x in zip(hot_pixels[0],hot_pixels[1]):
        fixed_image[y,x]=blurred[y,x]

    if worry_about_edges == True:
        height,width = np.shape(data)

        ###Now get the pixels on the edges (but not the corners)###

        #left and right sides
        for index in range(1,height-1):
            #left side:
            med  = np.median(data[index-1:index+2,0:2])
            diff = np.abs(data[index,0] - med)
            if diff>threshold: 
                hot_pixels = np.hstack(( hot_pixels, [[index],[0]]  ))
                fixed_image[index,0] = med

            #right side:
            med  = np.median(data[index-1:index+2,-2:])
            diff = np.abs(data[index,-1] - med)
            if diff>threshold: 
                hot_pixels = np.hstack(( hot_pixels, [[index],[width-1]]  ))
                fixed_image[index,-1] = med

        #Then the top and bottom
        for index in range(1,width-1):
            #bottom:
            med  = np.median(data[0:2,index-1:index+2])
            diff = np.abs(data[0,index] - med)
            if diff>threshold: 
                hot_pixels = np.hstack(( hot_pixels, [[0],[index]]  ))
                fixed_image[0,index] = med

            #top:
            med  = np.median(data[-2:,index-1:index+2])
            diff = np.abs(data[-1,index] - med)
            if diff>threshold: 
                hot_pixels = np.hstack(( hot_pixels, [[height-1],[index]]  ))
                fixed_image[-1,index] = med

        ###Then the corners###

        #bottom left
        med  = np.median(data[0:2,0:2])
        diff = np.abs(data[0,0] - med)
        if diff>threshold: 
            hot_pixels = np.hstack(( hot_pixels, [[0],[0]]  ))
            fixed_image[0,0] = med

        #bottom right
        med  = np.median(data[0:2,-2:])
        diff = np.abs(data[0,-1] - med)
        if diff>threshold: 
            hot_pixels = np.hstack(( hot_pixels, [[0],[width-1]]  ))
            fixed_image[0,-1] = med

        #top left
        med  = np.median(data[-2:,0:2])
        diff = np.abs(data[-1,0] - med)
        if diff>threshold: 
            hot_pixels = np.hstack(( hot_pixels, [[height-1],[0]]  ))
            fixed_image[-1,0] = med

        #top right
        med  = np.median(data[-2:,-2:])
        diff = np.abs(data[-1,-1] - med)
        if diff>threshold: 
            hot_pixels = np.hstack(( hot_pixels, [[height-1],[width-1]]  ))
            fixed_image[-1,-1] = med

    return hot_pixels,fixed_image     



def data_corr(image):
    ''' takes out outliers, noise offsets in 64x1024 channels and row gradiant at row 451.'''
    H, im = find_outlier_pixels(image, worry_about_edges=False)  
    return im

def reduce_data(stack):
    '''goes through the rows and column channels and subtracts off the median'''
    # sigma clips at 2sigma
    filtered_im = sigma_clip(stack, sigma=2, masked=True)
    # finds the median value of in the top and bottom half of the frame and subtracts it off.  
    a = np.array([0,451, 1024])
    for i in np.arange(len(a)-1):
        row = stack[:,a[i]:a[i+1],:]  
        filtered_row = filtered_im.data[:,a[i]:a[i+1],:]  
        med = np.nanmedian(filtered_row, axis=(1,2))
        stack[:,a[i]:a[i+1],:] = (row.T - med).T

    # sigma clips the new image
    filtered_medsubfin = sigma_clip(stack, sigma=2, masked=True)
    # finds the median value of each 64 column channels and subtracts it off.
    a = np.arange(0, np.shape(stack)[1]-1, 64)
    a = np.append(a, np.shape(stack)[1])
    for i in np.arange(len(a)-1):
        col = stack[:,:, a[i]:a[i+1]] 
        filtered_col = filtered_medsubfin.data[:,:, a[i]:a[i+1]]
        med = np.nanmedian(filtered_col, axis=(1,2))
        stack[:,:, a[i]:a[i+1]] = (col.T - med).T
    return stack


    
###################### o ########################
# CHANGEABLE PARAMETERS 
    
datadir = '/Volumes/ARCTURUS/LMIRCam_Jan15/2cubes/AB-Aur/Kshort'
savedir = '/Volumes/ARCTURUS/LMIRCam_Jan15/3bkg_sub_1x1/AB-Aur/Kshort'

image = datadir + '/lm_150104_04564_04683.fits'
nod_image = datadir + '/lm_150104_04684_04803.fits'

# open data
data = fits.getdata(image)
Pdata = np.copy(data[0,:,:])
hdr = fits.getheader(image)
nums = (image.split('150104_')[-1]).split('.')[0]
print(nums)
print('image:', image.split('/')[-1], ' NOD=', hdr['NOD'])

# open nod data
data_nod= fits.getdata(nod_image)
hdr_nod = fits.getheader(nod_image)
nod_name = nod_image.split('/')[-1]
print('nod image:', nod_name, ' NOD=', hdr_nod['NOD'])

###################### o ########################

# FIND MEDIAN OF NOD IMAGES
maskbadpix_nod = np.zeros((int(len(data_nod)/2), int(data_nod.shape[1]), int(data_nod.shape[2])))
for im in np.arange(data_nod.shape[0])[0:int(len(data_nod)/2)]:
    print('bkg image:', im, end='\r')
    maskbadpix_nod[im,:,:] = data_corr(data_nod[im, :, :])
print() 
medsub_fin_nod = reduce_data(maskbadpix_nod)
median_data_nod = np.nanmedian(medsub_fin_nod, axis=0)

# RUN THROUGH BKG SUBTRACTION FOR DATA IMAGES
maskbadpix_data = np.zeros((len(data), int(data.shape[1]), int(data.shape[2])))
for im in np.arange(data.shape[0]):
    print('data image:', im, end='\r')
    # reduce data
    data_reduced = data_corr(data[im,:,:])
    # subtract off median background
    maskbadpix_data[im,:,:] = data_reduced
print()
Pmaskbadpix_data = np.copy(maskbadpix_data[0,:,:])

# NOD SUBTRACTION
medsub_fin_data = reduce_data(maskbadpix_data)
bkg_data_reduced_FINAL = medsub_fin_data - median_data_nod

# MASK AND SUBTRACT OFF GRADIANT 
mask_im = np.copy(bkg_data_reduced_FINAL)
mask_im[:,:,413-150:413+150] = np.nan
a = np.array([0,451, 1024])
for k in np.arange(len(a)-1):
    print('working on: ', a[k], "-", a[k+1], end='\r')
    row = mask_im[:,a[k]:a[k+1],:]  
    row2 = bkg_data_reduced_FINAL[:,a[k]:a[k+1],:]
    med = np.nanmedian(row, axis=(1,2))
    bkg_data_reduced_FINAL[:,a[k]:a[k+1],:] = (row2.T - med).T

print()

# SAVE FILE
hdr.append(('COMMENT', f'nod bkg subtracted with {nod_name}'), end=True)
fits.writeto(savedir + f'/final_lm_150104_{nums}.fits', bkg_data_reduced_FINAL, hdr, overwrite=True)
print('file saved')



###################### o ########################
# PLOT STEPS
fig, ((ax1, ax2, ax3), (ax4, ax5,ax6), (ax7, ax8, ax9)) = plt.subplots(figsize=(9,7), nrows=3, ncols=3) 

# plot 1 frame from raw data
vmin,vmax = interval.get_limits(Pdata) 
ax1.set_title('Raw Target Image', fontsize=10) 
im1 = ax1.imshow(Pdata,vmin=vmin, vmax=vmax,origin='lower', cmap='gist_gray') 
plt.colorbar(im1, ax=ax1)


ax2.set_title('Hot pixels removed', fontsize=10) 
im2=ax2.imshow(Pmaskbadpix_data,vmin=vmin, vmax=vmax,origin='lower',cmap='gist_gray') 
plt.colorbar(im2, ax=ax2)

fil_im = sigma_clip(Pmaskbadpix_data, sigma=3, masked=True)
ax3.set_title('3 sigma outliers', fontsize=10)
current_cmap = matplotlib.cm.gist_gray
current_cmap.set_bad(color='red')
fil_im.data[fil_im.mask == True] = np.nan 
im3=ax3.imshow(fil_im.data, origin='lower',cmap='gist_gray',vmin=vmin, vmax=vmax)
plt.colorbar(im3, ax=ax3)


im = [medsub_fin_data[0,:,:], median_data_nod, bkg_data_reduced_FINAL[0,:,:]]
midax = [ax4, ax5, ax6]
bottomax = [ax7, ax8, ax9]
title = ['Reduced Target Image', 'Median Opposite Nod', 'Background Subtracted\n Target Image']
for i in np.arange(3): 
    vmin,vmax = interval.get_limits(im[i]) 
    if i !=2:
        q = midax[i].imshow(im[i], vmin=vmin,vmax=vmax, origin='lower',cmap='gist_gray')
    else:
        q = midax[i].imshow(im[i], vmin=-500, vmax=500,origin='lower',cmap='gist_gray')
    plt.colorbar(q, ax=midax[i])
    midax[i].set_title(title[i], fontsize=10)
    midax[i].set_xlabel('x', fontsize=8)
    midax[i].set_ylabel('y', fontsize=8)
    midax[i].axvline(50, color='b', linewidth=1.5)
    midax[i].axhline(50, color='g', linewidth=1.5)
    
    bottomax[i].plot(im[i][50,:], 'g', linewidth=0.5, label='x')
    bottomax[i].plot(im[i][:,50], 'b', linewidth=0.5, label='y')
    bottomax[i].axhline(0, color='k')
    bottomax[i].set_xlabel('position', fontsize=8)
    bottomax[i].set_ylabel('counts', fontsize=8)

all_axes = np.array([ax1, ax2, ax3, ax4, ax5, ax6, ax7,ax8,ax9])
for j in np.arange(9):
    all_axes[j].tick_params(which='major', length=5, direction='in', top=True,
                   right=True, left=True,bottom=True, labelsize=8)
    all_axes[j].tick_params(which='minor', length=2.5, direction='in', top=True, right=True, left=True,bottom=True, labelsize=8)
    all_axes[j].minorticks_on()
    
for k in np.arange(3):
    all_axes[k].set_xlabel('x', fontsize=8)
    all_axes[k].set_ylabel('y', fontsize=8)

ax9.legend(loc='upper right', fontsize=8, ncol=2)

plt.tight_layout()
plt.savefig('reduction_steps.png', transparent=True, dpi=150)
plt.show()

###################### o ########################
