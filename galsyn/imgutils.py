import numpy as np
import astropy.units as u
from .utils import *

def angular_to_physical(z, arcsec_per_pix, cosmo):
    """
    Calculates the physical size (in kpc) corresponding to a single pixel
    at a given redshift.

    This function uses the provided cosmology object to determine the proper
    distance scale at a specific redshift. It then converts the angular size
    of a pixel from arcseconds to a physical size in kiloparsecs (kpc).

    Parameters
    ----------
    z : float
        The redshift of the object.

    arcsec_per_pix : float
        The angular size of a single pixel in arcseconds.

    cosmo : astropy.cosmology object
        The cosmology object to use for the distance calculation. This should be
        an instance from the `astropy.cosmology` module.

    Returns
    -------
    float
        The physical size of a single pixel in kiloparsecs (kpc).
    """
    kpc_per_arcmin = cosmo.kpc_proper_per_arcmin(z)
    # Use astropy units for conversion to avoid hardcoded factors
    kpc_per_pix = (kpc_per_arcmin * arcsec_per_pix * u.arcsec).to(u.kpc)
    return kpc_per_pix.value

def physical_to_angular(z, physical_size_kpc, cosmo):
    """
    Calculates the angular size (in arcseconds) of an object with a given
    physical size at a specific redshift.

    Parameters
    ----------
    z : float
        The redshift of the object.

    physical_size_kpc : float
        The physical size of the object in kiloparsecs (kpc).

    cosmo : astropy.cosmology object
        The cosmology object to use for the angular diameter distance
        calculation. This should be an instance from the `astropy.cosmology` module.

    Returns
    -------
    float
        The angular size of the object in arcseconds.
    """

    # Compute angular diameter distance in kpc
    D_A = cosmo.angular_diameter_distance(z).to(u.kpc)
    
    # Convert physical size to angular size in arcseconds
    #angular_size_arcsec = (physical_size_kpc * u.kpc / D_A).to(u.rad).value * 206265  # arcsec
    angular_size_arcsec = (physical_size_kpc * u.kpc / D_A).to(u.arcsec, u.dimensionless_angles()).value  # arcsec
    
    return angular_size_arcsec

def convert_flux_map(flux_map, wave_eff, to_unit='nJy', pixel_scale_arcsec=None):
    c_as_per_s = 2.99792458e18  # speed of light in Angstrom/s
    
    if to_unit == 'erg/s/cm2/A':
        return flux_map

    # Calculate f_nu (erg/s/cm2/Hz) once
    f_nu = flux_map * (wave_eff**2) / c_as_per_s

    if to_unit == 'nJy':
        return f_nu * 1e23 * 1e9  # 1e23 to Jy, 1e9 to nJy

    elif to_unit == 'AB magnitude':
        # Mask non-positive values to avoid log10 warnings
        mag = np.full_like(f_nu, np.nan)
        mask = f_nu > 0
        mag[mask] = -2.5 * np.log10(f_nu[mask]) - 48.6
        return mag

    elif to_unit == 'MJy/sr':
        if pixel_scale_arcsec is None:
            raise ValueError("pixel_scale_arcsec is required for MJy/sr")
        
        # Steradians per pixel
        sr_per_arcsec2 = (np.pi / (180.0 * 3600.0))**2
        pixel_area_sr = (pixel_scale_arcsec**2) * sr_per_arcsec2
        
        # Convert f_nu to Jy, then to MJy, then divide by area
        return (f_nu * 1e23) / 1e6 / pixel_area_sr

    else:
        raise ValueError(f"Unsupported target unit: {to_unit}")
