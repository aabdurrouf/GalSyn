import os, sys 
import numpy as np
#from astropy import units as u
from astropy.constants import L_sun
from scipy.interpolate import interp1d

def interp_age_univ_from_z(z, cosmo=None):

    if cosmo is None:
        from astropy.cosmology import Planck15 as cosmo
    
    from scipy.interpolate import interp1d
    
    arr_z = np.linspace(0.0, 12.0, 100)
    age_univ = cosmo.age(arr_z).value

    f = interp1d(arr_z, age_univ, fill_value="extrapolate")
    return f(z)

def cosmo_redshifting(wave_rest, L_lambda_rest, z, cosmo=None):
    """
    Performs cosmological redshifting of a spectrum.

    This function converts an emitted spectrum, provided in units of
    Solar luminosity per Angstrom, into an observed flux density in units
    of erg/s/cm^2/Angstrom, given a cosmological redshift.

    Parameters
    ----------
    wave_rest : `numpy.ndarray`
        Wavelengths of the emitted spectrum in Angstrom.
    L_lambda_rest : `numpy.ndarray` 
        Luminosity per unit wavelength of the emitted spectrum in Solar Luminosity per Angstrom ($L_{\odot}/\AA$).
    z : float
        The cosmological redshift (z) at which the spectrum is observed.

    Returns
    -------
    tuple of `astropy.units.Quantity`
        A tuple containing two astropy Quantity objects:
        - observed_wavelengths : Wavelengths in the observed frame, in Angstrom.
        - observed_flux_density : Flux density in the observed frame, in erg/s/cm^2/Angstrom.
    """

    if cosmo is None:
        from astropy.cosmology import Planck15 as cosmo

    # Observed wavelength (redshifted)
    wave_obs = wave_rest * (1 + z)

    # Convert luminosity to erg/s/Angstrom
    L_lambda_erg = L_lambda_rest * L_sun.to('erg/s').value  # Now in erg/s/Angstrom

    # Get luminosity distance in cm
    D_L_cm = cosmo.luminosity_distance(z).to('cm').value

    # Compute observed flux density using:
    # F_lambda = (1 / (4 * pi * D_L^2)) * (1 / (1 + z)) * L_lambda_rest
    F_lambda_obs = (L_lambda_erg / (4 * np.pi * D_L_cm**2)) / (1 + z)  # in erg/s/cm^2/Angstrom

    return wave_obs, F_lambda_obs
    

def filtering(wave_spec, flux_spec, wave_filter, trans_filter):
    """
    Compute photometric flux through a filter using spectrum in F_lambda units.

    Parameters
    ----------
    wave_spec : ndarray
        Wavelength array of the spectrum [Å].
    flux_spec : ndarray
        Flux density array of the spectrum [erg/s/cm²/Å].
    wave_filter : ndarray
        Wavelength array of the filter transmission curve [Å].
    trans_filter : ndarray
        Transmission values of the filter (unitless).

    Returns
    -------
    flux_phot : float
        Photometric flux in units of erg/s/cm²/Å.
    """
    # Interpolate spectrum and filter to common wavelength grid
    wave_min = max(np.min(wave_spec), np.min(wave_filter))
    wave_max = min(np.max(wave_spec), np.max(wave_filter))

    wave_common = np.linspace(wave_min, wave_max, 10000)

    flux_interp = interp1d(wave_spec, flux_spec, kind='linear', bounds_error=False, fill_value=0.0)
    trans_interp = interp1d(wave_filter, trans_filter, kind='linear', bounds_error=False, fill_value=0.0)

    F_lambda = flux_interp(wave_common)
    T_lambda = trans_interp(wave_common)

    numerator = np.trapz(F_lambda * T_lambda * wave_common, wave_common)
    denominator = np.trapz(T_lambda * wave_common, wave_common)

    flux_phot = numerator / denominator

    return flux_phot


# wave: wavelength in Angstroms
def igm_att_madau(wave,z):
    """
    Compute IGM attenuation (transmission) due to Lyman series and Lyman continuum
    following Madau et al. 1995.

    Parameters:
    -----------
    wave : array_like
        Observed-frame wavelength array in Angstroms.
    z : float
        Source redshift.

    Returns:
    --------
    transmission : ndarray
        Transmission factor at each wavelength (same shape as input).
    """

    wave = np.asarray(wave)
    nwaves = len(wave)

    lylambda = np.zeros(32)
    madauA = np.zeros(32)

    nseries = 4
    for ii in range(2,int(nseries)+2):
        lylambda[ii-2] = 912.0*np.power(ii,2.0)/(np.power(ii,2.0)-1.0)

    madauA[0] = 0.0036
    madauA[1] = 0.0017
    madauA[2] = 0.0012
    madauA[3] = 0.00093

    teffline = np.zeros(nwaves)
    for ii in range(0,nseries):
        lmin = lylambda[ii]
        lmax = lylambda[ii]*(1.0+z)

        idx0 = np.where((wave>=lmin) & (wave<=lmax))
        teffline[idx0[0]] += madauA[ii]*np.exp(3.46*np.log(wave[idx0[0]]/lylambda[ii]))

    xc = wave/912.0
    xem = 1.0+z

    idx = np.where(xc<1.0)
    xc[idx[0]] = 1.0

    idx = np.where(xc>xem)
    xc[idx[0]] = xem

    teffcont = 0.25*xc*xc*xc*(np.exp(0.46*np.log(xem)) - np.exp(0.46*np.log(xc)))
    teffcont = teffcont + 9.4*np.exp(1.5*np.log(xc))*(np.exp(0.18*np.log(xem)) - np.exp(0.18*np.log(xc)))
    teffcont = teffcont - 0.7*xc*xc*xc*(np.exp(-1.32*np.log(xc)) - np.exp(-1.32*np.log(xem)))
    teffcont = teffcont - 0.023*(np.exp(1.68*np.log(xem)) - np.exp(1.68*np.log(xc)))
    tefftot = teffline + teffcont

    return np.exp(-1.0*tefftot)


