import os, sys 
import numpy as np

def interp_age_univ_from_z(z, cosmo=None):

    if cosmo is None:
        from astropy.cosmology import Planck15 as cosmo
    
    from scipy.interpolate import interp1d
    
    arr_z = np.linspace(0.0, 12.0, 100)
    age_univ = cosmo.age(arr_z).value

    f = interp1d(arr_z, age_univ, fill_value="extrapolate")
    return f(z)



