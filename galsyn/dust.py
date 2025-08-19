import sys
import numpy as np
from galsyn import config
import importlib.resources 

wave_V = 0.5500

def tau_dust_given_z(z, norm_dust_z, norm_dust_tau):
    """
    Calculates the dust optical depth normalization at a given redshift.

    This function interpolates a given relation between redshift and dust tau,
    often based on simulation results like Vogelsberger+2020.

    Args:
        z (float): The target redshift.
        norm_dust_z (array-like): The array of redshifts for the normalization relation.
        norm_dust_tau (array-like): The array of corresponding tau values.

    Returns:
        float: The interpolated dust optical depth normalization at the given redshift.
    """
    norm_dust_z = np.asarray(norm_dust_z)
    norm_dust_tau = np.asarray(norm_dust_tau)
    
    from scipy.interpolate import interp1d
    
    # based on Vogelsberger+2020 (Table 3)
    f = interp1d(norm_dust_z, norm_dust_tau, fill_value="extrapolate")
    return f(z)

def drude_profile(Bump_strength, wave_um):
    """
    Calculates the Drude profile for the 2175 Angstrom UV bump.

    Args:
        Bump_strength (float): The amplitude or strength of the UV bump feature.
        wave_um (array-like): Wavelength in units of microns.

    Returns:
        np.ndarray: The Drude profile D(lambda) corresponding to the input wavelengths.
    """
    wave0 = 0.2175
    dwave = 0.035
    part1 = wave_um * wave_um * dwave * dwave
    part2 = np.square((wave_um*wave_um) - (wave0*wave0))  

    D_lambda = Bump_strength * part1 / (part2 + part1)
    return D_lambda

def bump_amp_from_dust_index(dust_index):
    """
    Derives the UV bump amplitude from the dust index.

    This relationship is based on the findings of Kriek & Conroy (2013).

    Args:
        dust_index (float): The power-law slope modifier for the dust curve.

    Returns:
        float: The calculated UV bump amplitude.
    """
    # Based on Kriek & Conroy (2013)
    return 0.85 - 1.9*dust_index

def unresolved_dust_birth_cloud_Alambda_per_AV(wave_ang, dust_index_bc=-0.7):
    """
    Calculates a simple power-law attenuation for stellar birth clouds.

    Args:
        wave_ang (array-like): Wavelength in Angstroms.
        dust_index_bc (float, optional): The power-law slope for the birth cloud
                                         component. Defaults to -0.7.

    Returns:
        np.ndarray: The normalized attenuation curve A(lambda)/A(V).
    """
    Alambda_per_AV = np.power(wave_ang/wave_V, dust_index_bc)
    return Alambda_per_AV

def calzetti_dust_klambda(wave_ang):
    """
    Calculates the k(lambda) reddening curve for the Calzetti+00 law.

    This is a helper function that implements the piecewise function for the
    Calzetti starburst attenuation law.

    Args:
        wave_ang (array-like): Wavelength in Angstroms.

    Returns:
        np.ndarray: The reddening curve k(lambda).
    """
    wave_um = wave_ang/1e+4     # in micron
    idx = np.where(wave_um <= 0.63)[0]
    k_lambda1 = 4.05 + (2.659*(-2.156 + (1.509/wave_um[idx]) - (0.198/wave_um[idx]/wave_um[idx]) + (0.011/wave_um[idx]/wave_um[idx]/wave_um[idx])))

    idx = np.where(wave_um > 0.63)[0]
    k_lambda2 = 4.05 + (2.659*(-1.857 + (1.040/wave_um[idx]))) 

    k_lambda = k_lambda1.tolist() + k_lambda2.tolist()

    return np.asarray(k_lambda)

def calzetti_dust_Alambda_per_AV(wave_ang):
    """
    Calculates the normalized Calzetti+00 starburst attenuation law.

    Args:
        wave_ang (array-like): Wavelength in Angstroms.

    Returns:
        np.ndarray: The normalized attenuation curve A(lambda)/A(V).
    """
    k_lambda = calzetti_dust_klambda(wave_ang)
    Alambda_per_AV = k_lambda/4.05
    return Alambda_per_AV

def modified_calzetti_dust_Alambda_per_AV(wave_ang, dust_index=0.0, bump_amp=None):
    """
    Calculates a modified Calzetti law with a UV bump and power-law tilt.

    Args:
        wave_ang (array-like): Wavelength in Angstroms.
        dust_index (float, optional): The power-law slope modifier. Defaults to 0.0.
        bump_amp (float, optional): The UV bump amplitude. If None, it is derived
                                    from `dust_index`. Defaults to None.

    Returns:
        np.ndarray: The normalized modified Calzetti attenuation curve.
    """
    k_lambda = calzetti_dust_klambda(wave_ang)

    if bump_amp is None:
        bump_amp = bump_amp_from_dust_index(dust_index)

    wave_um = wave_ang/1e+4
    D_lambda = drude_profile(bump_amp, wave_um)
    Alambda_per_AV = (k_lambda + D_lambda)*np.power(wave_um/wave_V, dust_index)/4.05
    return Alambda_per_AV

def salim18_dust_Alambda_per_AV(wave_ang, salim_a0, salim_a1, salim_a2, salim_a3, salim_B, salim_RV):
    """
    Calculates the Salim+18 dust attenuation law.

    Args:
        wave_ang (array-like): Wavelength in Angstroms.
        salim_a0, a1, a2, a3 (float): Polynomial coefficients for the curve.
        salim_B (float): The strength of the UV bump.
        salim_RV (float): The ratio of total to selective extinction, R(V).

    Returns:
        np.ndarray: The normalized Salim+18 attenuation curve A(lambda)/A(V).
    """
    wave_um = wave_ang/1e+4      # in micron

    D_lambda = drude_profile(salim_B, wave_um)
    k_lambda = salim_a0 + (salim_a1/wave_um) + (salim_a2/wave_um/wave_um) + (salim_a3/wave_um/wave_um/wave_um) + D_lambda + salim_RV

    Alambda_per_AV = k_lambda / salim_RV
    return Alambda_per_AV

def _load_and_interpolate_dust_law(wave_ang, data_file_name):
    """Internal helper to load and interpolate a tabulated dust law."""
    from scipy.interpolate import interp1d
    
    try:
        data_path = str(importlib.resources.files('galsyn.data').joinpath(data_file_name))
        data = np.loadtxt(data_path)
        f = interp1d(data[:, 0], data[:, 1], fill_value="extrapolate")
        return f(wave_ang)
    except Exception as e:
        print(f"Error loading dust data from {data_file_name}: {e}")
        sys.exit(1)

def fitzpatrick99_dust_Alambda_per_AV(wave_ang):
    """
    Loads and interpolates the Fitzpatrick (1999) Milky Way extinction law.

    Args:
        wave_ang (array-like): Wavelength in Angstroms.

    Returns:
        np.ndarray: The interpolated, normalized extinction curve A(lambda)/A(V).
    """
    return _load_and_interpolate_dust_law(wave_ang, "fitzpatrick99.txt")

def ccm89_dust_Alambda_per_AV(wave_ang):
    """
    Loads and interpolates the Cardelli, Clayton, & Mathis (1989) MW extinction law.

    Args:
        wave_ang (array-like): Wavelength in Angstroms.

    Returns:
        np.ndarray: The interpolated, normalized extinction curve A(lambda)/A(V).
    """
    return _load_and_interpolate_dust_law(wave_ang, "ccm89.txt")

def lmc_gordon2003_dust_Alambda_per_AV(wave_ang):
    """
    Loads and interpolates the Gordon+03 extinction law for the LMC.

    Args:
        wave_ang (array-like): Wavelength in Angstroms.

    Returns:
        np.ndarray: The interpolated, normalized extinction curve A(lambda)/A(V).
    """
    return _load_and_interpolate_dust_law(wave_ang, "lmc_gordon2003.txt")

def smc_gordon2003_dust_Alambda_per_AV(wave_ang):
    """
    Loads and interpolates the Gordon+03 extinction law for the SMC.

    Args:
        wave_ang (array-like): Wavelength in Angstroms.

    Returns:
        np.ndarray: The interpolated, normalized extinction curve A(lambda)/A(V).
    """
    return _load_and_interpolate_dust_law(wave_ang, "smc_gordon2003.txt")


