import numpy as np
import astropy.units as u
from .utils import *

def angular_to_physical(z, arcsec_per_pix, cosmo):
    kpc_per_arcmin = cosmo.kpc_proper_per_arcmin(z)
    kpc_per_pix = kpc_per_arcmin.value*arcsec_per_pix/60.0 
    
    return kpc_per_pix

def physical_to_angular(z, physical_size_kpc, cosmo):
    # Compute angular diameter distance in kpc
    D_A = cosmo.angular_diameter_distance(z).to(u.kpc)
    
    # Convert physical size to angular size in arcseconds
    angular_size_arcsec = (physical_size_kpc * u.kpc / D_A).to(u.rad).value * 206265  # arcsec
    
    return angular_size_arcsec

def convert_flux_map(flux_map, wave_eff, to_unit='nJy', pixel_scale_arcsec=None):
    """
    Convert a 2D flux map from erg/s/cm^2/Angstrom into specified units.

    Parameters:
    -----------
    flux_map : 2D numpy array
        Flux map in unit of erg/s/cm^2/Angstrom
    wave_eff : float
        Effective wavelength in Angstrom
    to_unit : str
        Target unit: 'MJy/sr', 'nJy', 'AB magnitude', or 'erg/s/cm2/A'
    pixel_scale_arcsec : float, optional
        Pixel size in arcseconds (needed for 'MJy/sr')

    Returns:
    --------
    flux_converted : 2D numpy array
        Flux map in the target unit
    """

    c = 2.99792458e18  # speed of light in Angstrom/s

    if to_unit == 'erg/s/cm2/A':
        return flux_map

    elif to_unit == 'nJy':
        # Convert to f_nu in erg/s/cm^2/Hz
        f_nu = flux_map * wave_eff**2 / c
        # Convert erg/s/cm^2/Hz to nJy (1 Jy = 1e-23 erg/s/cm^2/Hz)
        return f_nu / 1e-23 * 1e9  # nJy

    elif to_unit == 'AB magnitude':
        # Convert to f_nu
        f_nu = flux_map * wave_eff**2 / c
        # AB magnitude: -2.5 * log10(f_nu [erg/s/cm^2/Hz]) - 48.6
        mag = -2.5 * np.log10(np.clip(f_nu, 1e-50, None)) - 48.6
        return mag

    elif to_unit == 'MJy/sr':
        if pixel_scale_arcsec is None:
            raise ValueError("pixel_scale_arcsec is required for conversion to MJy/sr")

        # Convert to f_nu in erg/s/cm^2/Hz
        f_nu = flux_map * wave_eff**2 / c
        # Convert erg/s/cm^2/Hz to Jy
        f_nu_jy = f_nu / 1e-23

        # Pixel area in arcsec² to steradians
        pixel_area_sr = (pixel_scale_arcsec * np.pi / (180.0 * 3600.0))**2

        # Convert to MJy/sr
        return f_nu_jy / pixel_area_sr / 1e6

    else:
        raise ValueError(f"Unsupported target unit: {to_unit}")
