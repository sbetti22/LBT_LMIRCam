'''
functions to run pyKLIP easily

written by: Sarah Betti May 2020
'''

import numpy as np

import matplotlib.pyplot as plt
import astropy.io.fits as fits

import glob
import time
import datetime
import pickle

import pyklip.rdi as rdi
import pyklip.parallelized as parallelized
import pyklip.instruments.LMIRCam as LMIRCam


def unstack(rawdir,date, fils, pixscale, savedir, savefilename):
    '''
    fils: list of files to unstack
    pixscale: float - the pixel scale in arseconds (0.011" or 0.022")
    savedir: string - location to save unstacked images. end with '/'
    savefilename: string - name of file used to save image
    
    EX:
    fils = np.sort(glob.glob('/Volumes/ARCTURUS/LMIRCam_Jan15/circsym_tests/circsym_ind_2x2/ABAur-Kshort_stack/circsym*.fits'))
    pixscale = 0.022
    savedir = '/Volumes/ARCTURUS/LMIRCam_Jan15/circsym_tests/circsym_ind_2x2/ABAur-Kshort_single/'
    savefilename = 'circsym_ind_final_lm_150104'
    '''
    
    fils = np.sort(fils)
    for j, stack in enumerate(fils):
        print(f'starting {j+1}/{len(fils)}')
        data = fits.getdata(stack)
        hdr = fits.getheader(stack)
        if '+' in stack:
            extra = '+' + (stack.split('+')[-1]).split('.')[0]
            print(extra)
        else:
            extra = ''
        comm = hdr['COMMENT']
        nod = hdr['NOD']
#        bkg = ([i for i in comm if 'nod bkg subtracted' in i])[0]
#        a = [i for i in comm if 'files between' in i] 
        if '+' in stack:
            a = (stack.split(f'{date}_')[-1]).split('+')
        else:
            a = (stack.split(f'{date}_')[-1]).split('.')
        minval = int(a[0].split('_')[0])
        maxval = int(a[0].split('_')[1])
        print(minval, maxval)
        vals = ['0'+str(i) for i in np.arange(minval, maxval+1)]

        for i in np.arange(len(data)):
            while len(vals[i]) < 5:
                vals[i] = '0' + vals[i]

            header = fits.getheader(rawdir + f'/lm_{date}_' + vals[i] + '.fits')
            header['NOD'] = nod
            header['PIXSCALE'] = pixscale
#            header.append(('COMMENT', bkg), end=True)
#             header.append(('COMMENT', 'aligned to median geo center with python circlesym'), end=True)
            if pixscale != 0.011:
                header.append(('COMMENT', f'binned: {pixscale}/pixel'), end=True)
 
            sizepix = np.shape(data[i,:,:])[1]
            sizearc = sizepix/pixscale
            sizeAU = sizearc / 162.9
            header.append(('COMMENT', f'FOV = {sizepix} pix = {sizearc}" = {sizeAU}AU'), end=True)
    
            if i == 0:
                print(savedir + savefilename + '_' + vals[i] + extra + '.fits')


            fits.writeto(savedir + savefilename + '_' + vals[i] + extra + '.fits', data[i,:,:], header, overwrite=True,output_verify = 'fix')
            
            
def make_correlation_matrix(fils, filt, savedir, save_suffix=None):
    '''
    fils: list of files PSF and target in one band
    filt: name of filter
    save_suffix: None or string to go to end of correlation matrix name
    
    EX:
    fils = np.sort(glob.glob('/Volumes/ARCTURUS/LMIRCam_Jan15/circsym_tests/circsym_ind_2x2/*_single/circsym*.fits'))
    filt = 'Kshort'
    save_suffix = '2x2binning_ind'

    '''
    fils = np.sort(fils)

    t1 = time.time()
    print('started at:', datetime.datetime.now().time())

    s = np.shape(fits.getdata(fils[0]))
    print(s[0], s[1])

    print(f'starting {filt}')
    psflib_imgs = np.zeros((len(fils), s[0], s[1]))
    psflib_filenames = np.empty(len(fils), dtype=object)
    print('starting to make array [',len(fils), s[0], s[1], ']' )
    for i in np.arange(len(fils)): 
        data = fits.getdata(fils[i])
        name = fils[i].split('/')[-1]
        
        if data.shape != (250,250):
            shape = data.shape
            pad_im = np.zeros((250, 250))
            pad_im[:shape[0], :shape[1]] = data
            data = pad_im
            print('replacing', fils[i])
            fits.writeto(fils[i], data, header=fits.getheader(fils[i]), overwrite=True)
        
        psflib_imgs[i, :, :] = data
        psflib_filenames[i] = name

    # make the PSF library
    print('finished for loop. starting PSFLibrary')
#     aligned_center = np.array([int(s[0]/2),int(s[1]/2)])
    aligned_center = np.array([124.5,124.5])
    print(aligned_center)
    psflib = rdi.PSFLibrary(psflib_imgs, aligned_center, psflib_filenames, compute_correlation=True)
    print('finished PSFLibrary.saving...')

    if save_suffix is not None:
        psflib.save_correlation(f"{savedir}corr_matrix_{filt}_{save_suffix}.fits")#, overwrite=True)
    else:
        psflib.save_correlation(f"{savedir}corr_matrix_{filt}.fits")#, overwrite=True)
    t2 = time.time()
    print((t2-t1)/(60), ' min')
    
    

def make_dataset(fils, filt, savedir, save_suffix=None):
    '''
    fils: list of files target in one band
    filt: name of filter
    save_suffix: None or string to go to end of dataset name
    
    EX:
    fils = np.sort(glob.glob('/Volumes/ARCTURUS/LMIRCam_Jan15/circsym_tests/circsym_ind_2x2/ABAur-Kshort_single/circsym*.fits'))
    filt = 'Kshort'
    save_suffix = '2x2binning_ind'

    '''
    print('making dataset')
    dataset = LMIRCam.LMIRCamData(fils, verbose=False)
    dataset.IWA = 7
    
    if save_suffix is not None:
        with open(f"{savedir}dataset_{filt}_{save_suffix}.pickle", "wb") as output_file:
            pickle.dump(dataset, output_file)
    else:
        with open(f"{savedir}dataset_{filt}.pickle", "wb") as output_file:
            pickle.dump(dataset, output_file)
            
            
            
            
def run_KLIP(fils, filt, corrmatrixfil, outputdir, datasetfil=None, annuli=1, subsection=1, mode='RDI',
             **kwargs ):     
    '''
    fils: ndarray - list of all PSF and target files for 1 filter
    filt: string  - name of filter
    corrmatrixfil: string - path and name of correlation matrix 
    outputdir: string - location to save final KLIP image
    datasetfil: None/string (optional) - path and name of dataset
    annuli: float - number of annuli
    subsection:float - numbe of subsections
    **kwargs:
        filesuffix: string - extra information on end of final KLIP image name 
        targetlist: ndarray - list of target files used to make dataset.  datasetfil must be None to work.
        maxtargetlist: integer - maximum number of files to use to create dataset.  datasetfil must be None and targetlist must be supplied to work. 
        IWA: integer - inner working angle for KLIP.  datasetfil must be None and targetlist must be supplied to work
        
        
        
    EX:
    fils = np.sort(glob.glob('/Volumes/ARCTURUS/LMIRCam_Jan15/circsym_tests/circsym_ind_2x2/*_single/circsym*.fits'))
    filt = 'Kshort'
    corrmatrixfil = 'corr_matrix_Kshort_2x2binning_ind.fits'
    outputdir = /Volumes/ARCTURUS/LMIRCam_Jan15/KLIP/AB-Aur/Kshort/'
    datasetfil = 'dataset_Kshort_2x2binning_ind.pickle'
    annuli = 1
    subsection = 1
    
    filesuffix = '2x2binning_ind'

    
    OR
    
    datasetfil = None
    
    filesuffix = '2x2binning_ind'
    targetlist = np.sort(glob.glob('/Volumes/ARCTURUS/LMIRCam_Jan15/circsym_tests/circsym_ind_2x2/ABAur-Kshort_single/circsym*.fits'))
    maxtargetlist = 1000
    IWA = 7

    '''
            
    print(f'starting pyKLIP on AB-Aur {filt}')
    s = np.shape(fits.getdata(fils[0]))

    psflib_imgs = np.zeros((len(fils), s[0], s[1]))
    psflib_filenames = np.empty(len(fils), dtype=object)
    for i in np.arange(len(fils)): 
        data = fits.getdata(fils[i])
        name = fils[i].split('/')[-1]
        psflib_imgs[i, :, :] = data
        psflib_filenames[i] = name
    #aligned_center = np.array([int(s[0]/2),int(s[1]/2)])
    aligned_center = np.array([124.5,124.5])
    # read in the correlation matrix we already saved
    corr_matrix_hdulist = fits.open(corrmatrixfil)
    corr_matrix = corr_matrix_hdulist[0].data

    print('making PSF library')
    # make the PSF library again, this time we have the correlation matrix
    psflib = rdi.PSFLibrary(psflib_imgs, aligned_center, psflib_filenames, correlation_matrix=corr_matrix)

    if datasetfil is None:
        if 'targetlist' in kwargs:
            targetlist = kwargs['targetlist']
            if 'maxtargetlist' in kwargs:
                targetlist = targetlist[0:kwargs['maxtargetlist']]

            print('making dataset')
            dataset = LMIRCam.LMIRCamData(targetlist, verbose=False)
            if 'IWA' in kwargs:
                dataset.IWA = kwargs['IWA']
            else:
                dataset.IWA = 7
        else:
            raise ValueError('targetlist must be supplied to make dataset')
    else:
        print('open dataset')
        with open(datasetfil, "rb") as input_file:
            dataset = pickle.load(input_file)
        print(dataset)
        
    print("The input datacubes are stored in a 3-D array with a shape of {0}. ".format(dataset.input.shape)
          + "The temporal and spatial dimentions have been collapsed")
    print('preparing PSF library with dataset')
    psflib.prepare_library(dataset)

    print('starting KLIP')
    t1 = time.time()
    # now we can run RDI klip
    # as all RDI images are aligned to aligned_center, we need to pass in that aligned_center into KLIP
    numbasis=[1,2,3,4,5,10,20]#,50] # number of KL basis vectors to use to model the PSF. We will try several different ones
    maxnumbasis=150 # maximum number of most correlated PSFs to do PCA reconstruction with
    if 'movement' in kwargs:
        movement = kwargs['movement']
    else:
        movement=0

    if 'filesuffix' in kwargs:
        fileprefix = f"pyklip_{filt}_k{maxnumbasis}a{annuli}s{subsection}m{movement}_{kwargs['filesuffix']}"
    else:
        fileprefix = f"pyklip_{filt}_k{maxnumbasis}a{annuli}s{subsection}m{movement}_{kwargs['filesuffix']}"
        
    if 'highpass' in kwargs:
        highpass = kwargs['highpass']
    else:
        highpass = False

    print(f'output will save to {outputdir}{fileprefix}')
    print('mode: ', mode)
    if mode == 'RDI':
        parallelized.klip_dataset(dataset, outputdir=outputdir, fileprefix=fileprefix, annuli=annuli, 
                                  subsections=subsection, numbasis=numbasis, maxnumbasis=maxnumbasis, mode="RDI", 
                                  aligned_center=aligned_center, psf_library=psflib, movement=movement, time_collapse='median', 
                                  highpass=highpass)
        
    if mode == 'ADI':
        parallelized.klip_dataset(dataset, outputdir=outputdir, fileprefix=fileprefix, annuli=annuli, 
                          subsections=subsection, numbasis=numbasis, maxnumbasis=maxnumbasis, mode="ADI", 
                          aligned_center=aligned_center, movement=movement, time_collapse='median', highpass=highpass)
    print(f'KLIP finished')
    t2 = time.time()
    print('finished:  ', (t2-t1)/(60), 'min')



            
            
            
            








