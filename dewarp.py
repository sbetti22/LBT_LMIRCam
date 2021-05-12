'''
from Dewarp python package: https://github.com/mwanakijiji/dewarp.git

Dewarp Spalding and Stone (2019) 

'''

import numpy as np
from scipy.ndimage import map_coordinates

# added sparse grid option, 2016 Dec 09, E.S.

def dewarp(image,P,Q,order=3):
    '''2d polynomial dewarping. Output x,y are mapped to input xt, yt
    according to:
    xt=F(x,y|P)
    yt=F(x,y|Q)
    see the documentation for numpy.polynomial.polynomial.polygrid2d 
    for information on how to organize polynomial coefficients in P, and Q
    INPUTS:
    image: 2D array
    P:   2D array of polynomial coefficients for generating xt from x
    Q:   2D array of polynomial coefficients for generating yt from y'''
    sh=image.shape
    xt=np.polynomial.polynomial.polygrid2d([range(sh[0])],[range(sh[1])],P)
    yt=np.polynomial.polynomial.polygrid2d([range(sh[0])],[range(sh[1])],Q)
    return map_coordinates(image,[yt,xt],order=order)

def make_dewarp_coordinates(imshape,P,Q):
    '''the xt and yt don't need be computed everytime if the 
    coefficients aren't changing. Use this function to compute 
    them, and use the function dewarp_with_precomputed_coords
    to affect the distortion correction'''
    xt=np.polynomial.polynomial.polygrid2d([range(imshape[0])],[range(imshape[1])],P)
    yt=np.polynomial.polynomial.polygrid2d([range(imshape[0])],[range(imshape[1])],Q)
    return [yt, xt]

# function to make a sparse grid of sample points (i.e., non-consecutive integers)
def make_dewarp_coordinates_sparseGrid(xGridArray,yGridArray,P,Q):
    '''the xt and yt don't need be computed everytime if the 
    coefficients aren't changing. Use this function to compute 
    them, and use the function dewarp_with_precomputed_coords
    to affect the distortion correction'''
    xt=np.polynomial.polynomial.polygrid2d([yGridArray],[xGridArray],P) ## CHANGE MADE; flipped x-y in arguments
    yt=np.polynomial.polynomial.polygrid2d([yGridArray],[xGridArray],Q)
    return yt, xt

def dewarp_with_precomputed_coords(image,coords,order=3):
    '''use make_dewarp_coordinates to get coords, a list
    of y-coordinates and x-coordinate arrays.'''
    return map_coordinates(image,coords,order=order)
