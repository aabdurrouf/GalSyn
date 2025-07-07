import os, sys 
import numpy as np
from .utils import *

def kpc_per_pixel(z, arcsec_per_pix, cosmo=None, H0=70.0, Om0=0.3):
    if cosmo is None:
        from astropy.cosmology import Planck15 as cosmo
		
    kpc_per_arcmin = cosmo.kpc_proper_per_arcmin(z)
    kpc_per_pix = kpc_per_arcmin.value*arcsec_per_pix/60.0 
    
    return kpc_per_pix






