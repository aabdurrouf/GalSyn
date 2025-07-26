import h5py
import numpy as np
import fsps
import os
from tqdm.auto import tqdm
from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib
import multiprocessing

# Constants for solar metallicity (from FSPS documentation or common usage)
# Z_sun = 0.019 (solar metallicity in FSPS for default BaSeL models)
# FSPS logzsol is log10(Z/Z_sun)
FSPS_Z_SUN = 0.019

# Global variable for the FSPS StellarPopulation instance in each worker process
# This will be initialized once per worker by the initializer function.
_ssp_worker_sp_instance = None

def init_ssp_worker(imf_type_val, add_neb_emission_val, gas_logu_val,
                    imf_upper_limit_val, imf_lower_limit_val,
                    imf1_val, imf2_val, imf3_val, vdmc_val, mdave_val):
    """
    Initializer function for each worker process in the SSP generation.
    Initializes the FSPS StellarPopulation object once per worker.
    """
    global _ssp_worker_sp_instance
    _ssp_worker_sp_instance = fsps.StellarPopulation(zcontinuous=1)
    _ssp_worker_sp_instance.params['imf_type'] = imf_type_val
    _ssp_worker_sp_instance.params["add_dust_emission"] = False
    _ssp_worker_sp_instance.params["add_neb_emission"] = add_neb_emission_val
    _ssp_worker_sp_instance.params['gas_logu'] = gas_logu_val
    _ssp_worker_sp_instance.params["fagn"] = 0
    _ssp_worker_sp_instance.params["sfh"] = 0   # SSP
    _ssp_worker_sp_instance.params["dust1"] = 0.0
    _ssp_worker_sp_instance.params["dust2"] = 0.0   # optical depth

    # Set new IMF parameters
    _ssp_worker_sp_instance.params['imf_upper_limit'] = imf_upper_limit_val
    _ssp_worker_sp_instance.params['imf_lower_limit'] = imf_lower_limit_val
    _ssp_worker_sp_instance.params['imf1'] = imf1_val
    _ssp_worker_sp_instance.params['imf2'] = imf2_val
    _ssp_worker_sp_instance.params['imf3'] = imf3_val
    _ssp_worker_sp_instance.params['vdmc'] = vdmc_val
    _ssp_worker_sp_instance.params['mdave'] = mdave_val


def _generate_single_ssp(age, logzsol):
    """
    Helper function to generate a single SSP spectrum and its surviving stellar mass.
    This function will be called in parallel and uses the pre-initialized FSPS instance.

    Parameters:
    -----------
    age : float
        Age of the SSP in Gyr.
    logzsol : float
        Logarithm of metallicity relative to solar (log10(Z/Z_sun)).

    Returns:
    --------
    tuple: (spectrum, stellar_mass)
        spectrum : np.ndarray
            SSP spectrum in L_sun/AA.
        stellar_mass : float
            Surviving stellar mass of the SSP.
    """
    global _ssp_worker_sp_instance

    # Set parameters for the current SSP
    _ssp_worker_sp_instance.params["logzsol"] = logzsol
    _ssp_worker_sp_instance.params['gas_logz'] = logzsol # Ensure gas_logz is also set for nebular emission consistency

    # Get spectrum in L_sun/AA
    _, spec = _ssp_worker_sp_instance.get_spectrum(peraa=True, tage=age)

    # Get the surviving stellar mass for this SSP
    stellar_mass = _ssp_worker_sp_instance.stellar_mass # This is the surviving stellar mass

    return spec, stellar_mass


def generate_ssp_grid(output_filename="ssp_spectra.hdf5",
                      ages_gyr=None,
                      logzsol_grid=None,
                      imf_type=1,
                      add_neb_emission=1,
                      gas_logu=-2.0,
                      imf_upper_limit=120.0, # New IMF parameter
                      imf_lower_limit=0.08,  # New IMF parameter
                      imf1=1.3,              # New IMF parameter
                      imf2=2.3,              # New IMF parameter
                      imf3=2.3,              # New IMF parameter
                      vdmc=0.08,             # New IMF parameter
                      mdave=0.5,             # New IMF parameter
                      overwrite=False,
                      n_jobs=-1): # Added n_jobs for parallelization
    """
    Generates a grid of Simple Stellar Population (SSP) spectra and their
    corresponding surviving stellar masses using FSPS, and saves them to an HDF5 file.
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
    imf_type : int, optional
        Initial Mass Function (IMF) type for FSPS. Defaults to 1 (Chabrier).
    add_neb_emission : int, optional
        Whether to add nebular emission (0=no, 1=yes). Defaults to 1.
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

    # Define default age grid if not provided
    if ages_gyr is None:
        # Logarithmic grid from 1 Myr to 13.8 Gyr (approx age of universe)
        ages_gyr = np.logspace(np.log10(0.001), np.log10(13.8), 100) # 100 ages

    # Define default metallicity grid if not provided
    if logzsol_grid is None:
        # Linear grid for log10(Z/Z_sun)
        logzsol_grid = np.linspace(-2.0, 0.2, 20) # 20 metallicities

    # Get the wavelength array once (it's constant for all SSPs)
    # Use a dummy FSPS instance just to get the wavelength grid
    dummy_sp = fsps.StellarPopulation(zcontinuous=1)
    wave, _ = dummy_sp.get_spectrum(peraa=True, tage=0.1)
    del dummy_sp # Delete dummy instance to free resources

    # Initialize arrays to store spectra and surviving stellar masses
    # Dimensions: (num_ages, num_metallicities, num_wavelengths) for spectra
    # Dimensions: (num_ages, num_metallicities) for stellar_mass
    ssp_spectra = np.zeros((len(ages_gyr), len(logzsol_grid), len(wave)), dtype=np.float32)
    ssp_stellar_masses = np.zeros((len(ages_gyr), len(logzsol_grid)), dtype=np.float32)

    # Determine the number of CPU cores to use
    num_cores = n_jobs
    if num_cores == -1:
        num_cores = multiprocessing.cpu_count() # Use all available cores

    print(f"Generating SSP spectra and surviving stellar masses on {num_cores} cores...")

    # Create a list of tasks for parallel processing
    tasks = []
    for age in ages_gyr:
        for logzsol in logzsol_grid:
            tasks.append((age, logzsol))

    # Execute tasks in parallel, with FSPS initialized once per worker
    with tqdm_joblib(total=len(tasks), desc="Generating SSPs") as progress_bar:
        results = Parallel(n_jobs=num_cores, verbose=0, initializer=init_ssp_worker,
                           initargs=(imf_type, add_neb_emission, gas_logu,
                                     imf_upper_limit, imf_lower_limit,
                                     imf1, imf2, imf3, vdmc, mdave))(
            delayed(_generate_single_ssp)(age, logzsol)
            for age, logzsol in tasks
        )

    # Populate the ssp_spectra and ssp_stellar_masses arrays from results
    # We need to map results back to their original grid positions
    k = 0
    for i_age, age in enumerate(ages_gyr):
        for i_z, logzsol in enumerate(logzsol_grid):
            spec, stellar_mass = results[k]
            ssp_spectra[i_age, i_z, :] = spec
            ssp_stellar_masses[i_age, i_z] = stellar_mass
            k += 1

    # Save to HDF5 file
    with h5py.File(output_filename, 'w') as f:
        f.create_dataset('wavelength', data=wave, compression="gzip")
        f.create_dataset('ages_gyr', data=ages_gyr, compression="gzip")
        f.create_dataset('logzsol', data=logzsol_grid, compression="gzip")
        f.create_dataset('spectra', data=ssp_spectra, compression="gzip")
        f.create_dataset('stellar_mass', data=ssp_stellar_masses, compression="gzip") # Save surviving stellar mass
        f.attrs['imf_type'] = imf_type
        f.attrs['add_neb_emission'] = add_neb_emission
        f.attrs['gas_logu'] = gas_logu
        f.attrs['imf_upper_limit'] = imf_upper_limit # Store new IMF parameter
        f.attrs['imf_lower_limit'] = imf_lower_limit # Store new IMF parameter
        if imf_type == 2:
            f.attrs['imf1'] = imf1                       # Store new IMF parameter
            f.attrs['imf2'] = imf2                       # Store new IMF parameter
            f.attrs['imf3'] = imf3                       # Store new IMF parameter
        if imf_type == 3:
            f.attrs['vdmc'] = vdmc                       # Store new IMF parameter
        if imf_type == 4:
            f.attrs['mdave'] = mdave                     # Store new IMF parameter
        f.attrs['z_sun'] = FSPS_Z_SUN # Store Z_sun used by FSPS

    print(f"SSP grid generation complete. Saved to '{output_filename}'.")
    return output_filename