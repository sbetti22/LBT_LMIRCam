'''
Main workhorse to reduce the data
1. background subtract raw data in up/down nod stacks
2. removes outlier pixels
3. removes column variation
4. stacks cubes and saves reduced stacked data.  

written by: Sarah Betti -  May 15, 2020

'''

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.stats import sigma_clip

from photutils import make_source_mask
from photutils import source_properties
from photutils import detect_threshold, detect_sources

import glob
import datetime
import os
import warnings
warnings.filterwarnings('ignore', category=UserWarning, append=True)


###################### o ########################

#find outlier pixel function found from stackoverflow
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




#function to run find outlier pixel function
def data_corr(image):
    ''' runs the find_outlier pixels.
    Did more originally, but was taken out, so this function really isn't needed, but i'm too lazy to fix it.'''
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

def determine_nod(method, i, split, hdr):
    ''' 
    determines which cube should be used for nod subtraction.  Normally, the next opposite nod will be used.  
    If it is the last file in the split up data, then the previous opposite nod before will be used. 
    '''
    
    # if both telescopes were used 
    if method == 'two_eyes':
        # if last file grab previous file 
        if i == len(split)-1:
            hdr_nod = fits.getheader(split[i-1])
        # if not last file, grab next file 
        else:
            hdr_nod = fits.getheader(split[i+1])
        # TEST to see if the nod file grabbed is actually the opposite nod.  If not, then keep grabbing files until it is the opposite nod.  
        j = 0
        found_opposite_nod = False
        while found_opposite_nod != True:
            if 'U-D' not in hdr['NOD']:
                if hdr['NOD'] == hdr_nod['NOD']:
                    j += 1
                    if i == len(split)-1:
                        hdr_nod = fits.getheader(split[i-j])
                    else:
                        hdr_nod = fits.getheader(split[i+j])
                else:
                    if 'U-D' in hdr['NOD']:
                        j+= 1
                        hdr_nod = fits.getheader(split[i+j])
                    else:
                        j+= 1
                        found_opposite_nod = True
            else:
                if hdr_nod['NOD'] != 'D':
                    j+= 1 
                    if i == len(split)-1:
                        hdr_nod = fits.getheader(split[i-j])
                        found_opposite_nod = True
                    else:
                        hdr_nod = fits.getheader(split[i+j])
                        found_opposite_nod = True
                else:
                    j +=1 
                    found_opposite_nod = True
                    
        # finally, grab the correct oppoisite nod file.
        if i == len(split)-1:
            data_nod = fits.getdata(split[i-j])
            split_name = split[i-j].split('/')[-1]
        else:
            data_nod = fits.getdata(split[i+j])
            split_name = split[i+j].split('/')[-1]
        print('nod image:', split_name, ' NOD=', hdr_nod['NOD'])

    # if one eye -- same thing as above.    
    else:
        if i == len(split)-1:
            # if it is the last image, then the nod subtraction image will be the one before
            #open up previous data 
            hdr_nod = fits.getheader(split[i-1])
            j = 0
            found_opposite_nod = False
            while found_opposite_nod != True:
                if hdr['NOD'] == hdr_nod['NOD']:
                    print(j)
                    j += 1
                    hdr_nod = fits.getheader(split[i-j])
                else:
                    j += 1
                    found_opposite_nod = True
            data_nod= fits.getdata(split[i-j])
            split_name = split[i-j].split('/')[-1]
            print('nod image:', split_name, ' NOD=', hdr_nod['NOD'])
        else:
            # if it is NOT the last image, then the nod subtraction image will be the one after. 
            hdr_nod = fits.getheader(split[i+1])
            j = 0
            found_opposite_nod = False
            while found_opposite_nod != True:
                if hdr['NOD'] == hdr_nod['NOD']:
                    j += 1
                    hdr_nod = fits.getheader(split[i+j])
                else:
                    j+= 1
                    found_opposite_nod = True
            data_nod= fits.getdata(split[i+j])
            split_name = split[i+j].split('/')[-1]
            print('nod image:', split_name, ' NOD=', hdr_nod['NOD'])
    # return the data for the nod cube and its name.  
    return data_nod, split_name

def data_split(all_data):
    '''
    Separate data by when it was taken, so a nod subtraction is only done of data taken around the same time.  
    '''
    #grab the object numbers from name
    s = np.array([])
    for i in np.arange(len(all_data)):
        nums = (all_data[i].split('{date}_')[-1]).split('.')[0]
        s = np.append(s, nums)

    #separate min and max numbers and min and max numbers of following frame to figure out where there is a break (switching targets)
    sp = []
    for i in np.arange(len(s)-1):
        n = s[i].split('_')
        n1, n2 = int(n[0]), int(n[1])
        nn = s[i+1].split('_')
        nn1, nn2 = int(nn[0]), int(nn[1])
        if n2 +1 != nn1:
            sp.append(i+1)

    # split data into sublists based on when the data was taken together
    all_data_split = np.split(all_data, sp)
    return all_data_split



###################### o ########################
# CHANGEABLE PARAMETERS 

# main directory
datadir =  '/Users/sbetti/Documents/dissertation_datasets/LMIRCam_Feb13/140213'

date = '140213'
target = 'HD39925'
filt = 'Lprime'
eyes = 'two_eyes'

savedir = datadir + f'/3bkg_sub_1x1/{target}/{filt}'

#get all data
all_data = np.sort(glob.glob(datadir + f'/2cubes/unsats/{target}/{filt}/lm_{date}*.fits'))


###################### o ########################
# START REDUCTION 

# split data by file numbers 
all_data_split = data_split(all_data)
print(all_data_split)
# LOOP THROUGH ALL DATA and perform NOD SUBTRACTION and REMOVE CHANNEL OFFSETS 
#loop through split up data files 
for split in all_data_split:
    # loop through data within each split 
    for i in np.arange(len(split)):
        # pull out data
        print(split[i])
        nums = (split[i].split(f'{date}_')[-1]).split('.')[0]
        print(nums)
        # only reduce data if it hasn't been reduced yet.  this allows you to stop and start code without losing your progress. 
        if not os.path.exists(f'{savedir}/final_lm_{date}_{nums}.fits'):
            #open data and header
            data = fits.getdata(split[i])
            hdr = fits.getheader(split[i])
           
            print('image:', split[i].split('/')[-1], ' NOD=', hdr['NOD'])
            
            # determine what images will be subtracted from each other.  
            data_nod, split_name = determine_nod(eyes, i, split, hdr)
            
            # FIND MEDIAN OF NOD IMAGES
            maskbadpix_nod = np.zeros((int(len(data_nod)/2), int(data_nod.shape[1]), int(data_nod.shape[2])))
            for im in np.arange(data_nod.shape[0])[0:int(len(data_nod)/2)]:
                print('bkg image:', im, end='\r')
                maskbadpix_nod[im,:,:] = data_corr(data_nod[im, :, :])
            print() 
            # remove offsets and bad pixels 
            medsub_fin_nod = reduce_data(maskbadpix_nod) 
            # take median
            median_data_nod = np.nanmedian(medsub_fin_nod, axis=0)

            # RUN THROUGH BKG SUBTRACTION FOR ALL DATA IMAGES
            maskbadpix_data = np.zeros((len(data), int(data.shape[1]), int(data.shape[2])))
            for im in np.arange(data.shape[0]):
                print('data image:', im, end='\r')
                # reduce data
                data_reduced = data_corr(data[im,:,:])
                # subtract off median background
                maskbadpix_data[im,:,:] = data_reduced
            print()
            
            # DO NOD SUBTRACTION! 
            medsub_fin_data = reduce_data(maskbadpix_data)
            bkg_data_reduced_FINAL = medsub_fin_data - median_data_nod

            # mask and subtract off gradiant in image again
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
            
            #SAVE FILE
            hdr.append(('COMMENT', f'nod bkg subtracted with {split_name}'), end=True)
            fits.writeto(f'{savedir}/final_lm_{date}_{nums}.fits', bkg_data_reduced_FINAL, hdr, overwrite=True)

        else:
            pass
    
import os
os.system('say "your program has finished"')

###################### o ########################


