import h5py
import numpy as np
import fsps
import os
from tqdm.auto import tqdm
from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib
import multiprocessing

# FSPS logzsol is log10(Z/Z_sun)
FSPS_Z_SUN = 0.019

# Global variable for the FSPS StellarPopulation instance in each worker process
# This will be initialized once per worker by the initializer function.
_ssp_worker_sp_instance = None

def init_ssp_worker(imf_type_val, imf_upper_limit_val, imf_lower_limit_val,
                    imf1_val, imf2_val, imf3_val, vdmc_val, mdave_val):
    """
    Initializer function for each worker process in the SSP generation.
    Initializes the FSPS StellarPopulation object once per worker.
    """
    global _ssp_worker_sp_instance
    _ssp_worker_sp_instance = fsps.StellarPopulation(zcontinuous=1)
    _ssp_worker_sp_instance.params['imf_type'] = imf_type_val
    _ssp_worker_sp_instance.params["add_dust_emission"] = False
    _ssp_worker_sp_instance.params["fagn"] = 0
    _ssp_worker_sp_instance.params["sfh"] = 0       # SSP
    _ssp_worker_sp_instance.params["dust1"] = 0.0
    _ssp_worker_sp_instance.params["dust2"] = 0.0   # optical depth

    _ssp_worker_sp_instance.params['imf_upper_limit'] = imf_upper_limit_val
    _ssp_worker_sp_instance.params['imf_lower_limit'] = imf_lower_limit_val
    _ssp_worker_sp_instance.params['imf1'] = imf1_val
    _ssp_worker_sp_instance.params['imf2'] = imf2_val
    _ssp_worker_sp_instance.params['imf3'] = imf3_val
    _ssp_worker_sp_instance.params['vdmc'] = vdmc_val
    _ssp_worker_sp_instance.params['mdave'] = mdave_val


def _generate_single_ssp(age, logzsol, logu, log_zratio):
    """
    Helper function to generate a single SSP spectrum (stellar continuum only and nebular emission only)
    and its surviving stellar mass.
    This function will be called in parallel and uses the pre-initialized FSPS instance.

    Parameters:
    -----------
    age : float
        Age of the SSP in Gyr.
    logzsol : float
        Logarithm of metallicity relative to solar (log10(Z/Z_sun)).
    logu : float
        The logarithm of the dimensionless ionization parameter (log(U)), which characterizes the intensity of the ionizing radiation field.
    log_zratio : float
        The fixed logarithmic ratio between gas-phase metallicity and stellar metallicity in units of dex, used to calculate the gas metallicity for nebular emission models.

    Returns:
    --------
    tuple: (stellar_continuum_spectrum, nebular_emission_spectrum, stellar_mass)
        stellar_continuum_spectrum : np.ndarray
            SSP spectrum with stellar continuum only in L_sun/AA.
        nebular_emission_spectrum : np.ndarray
            SSP spectrum with nebular emission only in L_sun/AA.
        stellar_mass : float
            Surviving stellar mass of the SSP.
    """
    global _ssp_worker_sp_instance

    # Calculate gas metallicity: log10(Z_gas/Z_sun) = log10(Z_star/Z_sun) + log10(Z_gas/Z_star)
    logzsol_gas = logzsol + log_zratio

    # Set parameters for the current SSP
    _ssp_worker_sp_instance.params["logzsol"] = logzsol
    _ssp_worker_sp_instance.params['gas_logz'] = logzsol_gas # Ensure gas_logz is also set for nebular emission consistency
    _ssp_worker_sp_instance.params['gas_logu'] = logu

    # 1. Generate SSP spectrum including nebular emission
    _ssp_worker_sp_instance.params["add_neb_emission"] = 1
    _, spec_total = _ssp_worker_sp_instance.get_spectrum(peraa=True, tage=age)

    # 2. Generate SSP spectrum with stellar continuum only (no nebular emission)
    _ssp_worker_sp_instance.params["add_neb_emission"] = 0
    _, spec_stellar_continuum = _ssp_worker_sp_instance.get_spectrum(peraa=True, tage=age)

    # 3. Subtract to get nebular emission only
    spec_nebular_emission = spec_total - spec_stellar_continuum

    # Get the surviving stellar mass for this SSP
    stellar_mass = _ssp_worker_sp_instance.stellar_mass

    return spec_stellar_continuum, spec_nebular_emission, stellar_mass


def generate_ssp_grid(output_filename="ssp_spectra.hdf5",
                      ages_gyr=None,
                      logzsol_grid=None,
                      logu_grid=None,
                      log_zratio=0.4,
                      imf_type=1,
                      imf_upper_limit=120.0,
                      imf_lower_limit=0.08,
                      imf1=1.3,
                      imf2=2.3,
                      imf3=2.3,
                      vdmc=0.08,
                      mdave=0.5,
                      overwrite=False,
                      n_jobs=-1,
                      rest_wave_min=500,  
                      rest_wave_max=30000): 
    """
    Generates a grid of Simple Stellar Population (SSP) spectra (stellar continuum only and nebular emission only)
    and their corresponding surviving stellar masses using FSPS, and saves them to an HDF5 file.
    Supports parallel computation by initializing FSPS once per worker process.

    Parameters:
    -----------
    output_filename : str, optional
        The name of the HDF5 file to save the SSP grid.
        Defaults to "ssp_spectra.hdf5".
    ages_gyr : np.ndarray, optional
        A 1D numpy array of ages in Gyr for which to generate SSP spectra.
        If None, a default logarithmic grid from 0.001 Gyr to 13.8 Gyr is used.
    logzsol_grid : np.ndarray, optional
        A 1D numpy array of log10(Z/Z_sun) values for which to generate SSP spectra.
        If None, a default linear grid from -2.0 to 0.2 is used.
    logu_grid : np.ndarray, optional
        A 1D numpy array of log(U) values for which to generate SSP spectra.
    log_zratio : float, optional
        The fixed logarithmic ratio between gas-phase metallicity and stellar metallicity in units of dex.
        This is used to calculate the gas metallicity for nebular emission models.
    imf_type : int, optional
        Initial Mass Function (IMF) type for FSPS. Defaults to 1 (Chabrier).
    gas_logu : float, optional
        Logarithm of the ionization parameter for nebular emission. Defaults to -2.0.
    imf_upper_limit : float, optional
        The upper limit of the IMF, in solar masses. Defaults to 120.0.
    imf_lower_limit : float, optional
        The lower limit of the IMF, in solar masses. Defaults to 0.08.
    imf1 : float, optional
        Logarithmic slope of the IMF over the range. Only used if imf_type=2. Defaults to 1.3.
    imf2 : float, optional
        Logarithmic slope of the IMF over the range. Only used if imf_type=2. Defaults to 2.3.
    imf3 : float, optional
        Logarithmic slope of the IMF over the range. Only used if imf_type=2. Defaults to 2.3.
    vdmc : float, optional
        IMF parameter defined in van Dokkum (2008). Only used if imf_type=3. Defaults to 0.08.
    mdave : float, optional
        IMF parameter defined in Dave (2008). Only used if imf_type=4. Defaults to 0.5.
    overwrite : bool, optional
        If True, overwrite the output file if it already exists. Defaults to False.
    n_jobs : int, optional
        Number of CPU cores to use for parallel processing. Defaults to -1 (all available).
    rest_wave_min : float, optional
        Minimum rest-frame wavelength in Angstroms to include in the output spectra. Defaults to 500.
    rest_wave_max : float, optional
        Maximum rest-frame wavelength in Angstroms to include in the output spectra. Defaults to 30000.

    Returns:
    --------
    str
        The path to the generated HDF5 file.
    """

    if os.path.exists(output_filename) and not overwrite:
        print(f"SSP grid file '{output_filename}' already exists. "
              "Set overwrite=True to regenerate.")
        return output_filename

    print(f"Generating SSP grid and saving to {output_filename}...")

    if ages_gyr is None:
        ages_gyr = np.logspace(np.log10(0.001), np.log10(13.8), 100)

    if logzsol_grid is None:
        logzsol_grid = np.linspace(-2.0, 0.2, 20)

    if logu_grid is None: 
        logu_grid = np.linspace(-4.0, -1.0, 10)

    # Get the wavelength array once (it's constant for all SSPs)
    # Use a dummy FSPS instance just to get the wavelength grid
    dummy_sp = fsps.StellarPopulation(zcontinuous=1)
    full_wave, _ = dummy_sp.get_spectrum(peraa=True, tage=0.1)
    del dummy_sp # Delete dummy instance to free resources

    # Apply wavelength cut
    wave_mask = (full_wave >= rest_wave_min) & (full_wave <= rest_wave_max)
    wave = full_wave[wave_mask]

    # Initialize 4D spectra: (age, zstar, logu, wave)
    s_cont_cube = np.zeros((len(ages_gyr), len(logzsol_grid), len(logu_grid), len(wave)), dtype=np.float32)
    n_em_cube = np.zeros_like(s_cont_cube)
    s_mass_cube = np.zeros((len(ages_gyr), len(logzsol_grid), len(logu_grid)), dtype=np.float32)

    tasks = [(a, z, u, log_zratio) for a in ages_gyr for z in logzsol_grid for u in logu_grid]
    
    num_cores = n_jobs
    if num_cores == -1:
        num_cores = multiprocessing.cpu_count()
        
    print(f"Generating 3D SSP grid on {num_cores} cores...")

    with tqdm_joblib(total=len(tasks), desc="Generating 3D SSPs") as progress_bar:
        results = Parallel(n_jobs=num_cores, initializer=init_ssp_worker,
                           initargs=(imf_type, imf_upper_limit, imf_lower_limit, 
                                     imf1, imf2, imf3, vdmc, mdave))(
            delayed(_generate_single_ssp)(*t) for t in tasks
        )

    # Map results back to the 3D grid
    k = 0
    for i_a in range(len(ages_gyr)):
        for i_z in range(len(logzsol_grid)):
            for i_u in range(len(logu_grid)):
                res_s, res_n, res_m = results[k]
                s_cont_cube[i_a, i_z, i_u, :] = res_s[wave_mask]
                n_em_cube[i_a, i_z, i_u, :] = res_n[wave_mask]
                s_mass_cube[i_a, i_z, i_u] = res_m
                k += 1

    # Save with relevant IMF attributes
    with h5py.File(output_filename, 'w') as f:
        f.create_dataset('wavelength', data=wave, compression="gzip")
        f.create_dataset('ages_gyr', data=ages_gyr)
        f.create_dataset('logzsol', data=logzsol_grid)
        f.create_dataset('logu_grid', data=logu_grid)
        f.create_dataset('stellar_continuum_spectra', data=s_cont_cube, compression="gzip")
        f.create_dataset('nebular_emission_spectra', data=n_em_cube, compression="gzip")
        f.create_dataset('stellar_mass', data=s_mass_cube, compression="gzip")
        
        # Meta-data
        f.attrs['log_zratio'] = log_zratio
        f.attrs['z_sun'] = FSPS_Z_SUN
        f.attrs['imf_type'] = imf_type
        f.attrs['imf_upper_limit'] = imf_upper_limit
        f.attrs['imf_lower_limit'] = imf_lower_limit
        # Conditional IMF slopes
        if imf_type == 2:
            f.attrs['imf1'], f.attrs['imf2'], f.attrs['imf3'] = imf1, imf2, imf3
        
    print(f"SSP grid generation complete. Saved to '{output_filename}'.")
    return output_filename