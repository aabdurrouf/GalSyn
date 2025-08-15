import h5py
import numpy as np
import bagpipes as pipes
import os
from tqdm.auto import tqdm
from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib
import multiprocessing
from scipy.interpolate import interp1d # Import interp1d

# Constants
L_SUN_ERG_S = 3.828e33
BAGPIPES_Z_SUN = 0.02

_ssp_worker_bagpipes_instance = None
gas_logu = None

def init_ssp_worker(gas_logu_val, rest_frame_wave_val):
    """
    Initializer function for each worker process in the SSP generation.
    Initializes the common components for the Bagpipes model.
    """
    global _ssp_worker_bagpipes_instance
    global gas_logu
    gas_logu = gas_logu_val

    dust = {}
    dust["type"] = "Calzetti"
    dust["Av"] = 0.0
    dust["eta"] = 1.0

    # Nebular emission will be handled dynamically in _generate_single_ssp
    #nebular = {"logU": gas_logu_val}

    model_components = {}
    model_components["redshift"] = 0.0
    model_components["veldisp"] = 0
    model_components["dust"] = dust
    #model_components["nebular"] = nebular # Always include nebular dict, logU will be set
    
    # Store the target rest_frame_wave for use by _generate_single_ssp
    model_components["_rest_frame_wave_target"] = rest_frame_wave_val

    _ssp_worker_bagpipes_instance = model_components


def _generate_single_ssp(age, logzsol):
    """
    Helper function to generate a single SSP spectrum (stellar continuum only and nebular emission only)
    using Bagpipes, then interpolates it to the target wavelength grid.
    """
    global _ssp_worker_bagpipes_instance

    # Retrieve the target rest_frame_wave from the global worker instance
    rest_frame_wave_target = _ssp_worker_bagpipes_instance["_rest_frame_wave_target"]

    metallicity_z_zsun = 10**logzsol

    burst = {}
    burst["age"] = age
    burst["massformed"] = 1.0
    burst["metallicity"] = metallicity_z_zsun

    # --- Generate spectrum with nebular emission ---
    current_model_components_total = _ssp_worker_bagpipes_instance.copy()
    current_model_components_total["burst"] = burst

    # Ensure nebular is enabled for total spectrum
    nebular = {"logU": gas_logu}
    current_model_components_total["nebular"] = nebular
    #current_model_components_total["nebular"] = {"logU": _ssp_worker_bagpipes_instance["nebular"]["logU"]} 
    
    model_total = pipes.model_galaxy(current_model_components_total, spec_wavs=np.arange(1000., 10000., 5.))
    full_wave = model_total.wavelengths
    full_fluxes_total_erg_s_aa = model_total.spectrum_full
    full_fluxes_total_l_sun_aa = full_fluxes_total_erg_s_aa / L_SUN_ERG_S
    interp_func_total = interp1d(full_wave, full_fluxes_total_l_sun_aa, kind='linear', 
                                 bounds_error=False, fill_value=0.0)
    spec_total_interpolated = interp_func_total(rest_frame_wave_target)

    # --- Generate spectrum with stellar continuum only (no nebular emission) ---
    current_model_components_stellar = _ssp_worker_bagpipes_instance.copy()
    current_model_components_stellar["burst"] = burst
    # Disable nebular emission for stellar continuum only
    # Change "current_model_components_stellar["nebular"] = None" to:
    #current_model_components_stellar["nebular"] = {} # Set to empty dictionary instead of None 

    model_stellar = pipes.model_galaxy(current_model_components_stellar, spec_wavs=np.arange(1000., 10000., 5.))
    full_wave_stellar = model_stellar.wavelengths # Should be the same as full_wave, but for clarity
    full_fluxes_stellar_erg_s_aa = model_stellar.spectrum_full
    full_fluxes_stellar_l_sun_aa = full_fluxes_stellar_erg_s_aa / L_SUN_ERG_S
    interp_func_stellar = interp1d(full_wave_stellar, full_fluxes_stellar_l_sun_aa, kind='linear', 
                                   bounds_error=False, fill_value=0.0)
    spec_stellar_continuum_interpolated = interp_func_stellar(rest_frame_wave_target)

    # --- Calculate nebular emission only ---
    spec_nebular_emission_interpolated = spec_total_interpolated - spec_stellar_continuum_interpolated

    surv_stellar_mass = model_total.sfh.stellar_mass

    return spec_stellar_continuum_interpolated, spec_nebular_emission_interpolated, surv_stellar_mass


def generate_ssp_grid_bagpipes(output_filename="ssp_spectra_bagpipes.hdf5",
                               ages_gyr=None,
                               logzsol_grid=None,
                               gas_logu=-2.0,
                               overwrite=False,
                               n_jobs=-1):
    """
    Generates a grid of Simple Stellar Population (SSP) models using Bagpipes.

    This function calculates SSPs for a grid of ages and metallicities, saving
    the results to a single HDF5 file. It separates the output spectra into two
    components: the stellar continuum and the nebular emission lines. It also
    calculates the surviving stellar mass fraction for each SSP. The process is
    parallelized across multiple CPU cores for efficiency.

    Args:
        output_filename (str, optional): The path and name for the output HDF5 file.
                                         Defaults to "ssp_spectra_bagpipes.hdf5".
        ages_gyr (array-like, optional): A 1D array of stellar population ages in Gyr.
                                         If None, a default log-spaced grid is used.
                                         Defaults to None.
        logzsol_grid (array-like, optional): A 1D array of metallicities in
                                             log(Z/Z_sun) units. If None, a default
                                             linear grid is used. Defaults to None.
        gas_logu (float, optional): The ionization parameter for nebular emission.
                                    Defaults to -2.0.
        overwrite (bool, optional): If True, an existing file at `output_filename`
                                    will be overwritten. If False, the function
                                    will exit if the file already exists.
                                    Defaults to False.
        n_jobs (int, optional): The number of CPU cores to use for parallel
                                processing. Defaults to -1 (use all available cores).

    Returns:
        str: The path to the generated HDF5 file.
    """
    if os.path.exists(output_filename) and not overwrite:
        print(f"SSP grid file '{output_filename}' already exists. "
              "Set overwrite=True to regenerate.")
        return output_filename

    print(f"Generating SSP grid and saving to {output_filename} using Bagpipes...")

    if ages_gyr is None:
        ages_gyr = np.logspace(np.log10(0.001), np.log10(13.8), 100)

    if logzsol_grid is None:
        logzsol_grid = np.linspace(-2.0, 0.2, 20)

    # Define the target rest-frame wavelength array upfront - This is the master grid
    rest_frame_wave = np.arange(100., 30000., 5.)

    # Initialize arrays with the dimensions based on the explicitly defined rest_frame_wave
    ssp_stellar_continuum_spectra = np.zeros((len(ages_gyr), len(logzsol_grid), len(rest_frame_wave)), dtype=np.float32)
    ssp_nebular_emission_spectra = np.zeros((len(ages_gyr), len(logzsol_grid), len(rest_frame_wave)), dtype=np.float32)
    ssp_stellar_masses = np.zeros((len(ages_gyr), len(logzsol_grid)), dtype=np.float32)

    num_cores = n_jobs
    if num_cores == -1:
        num_cores = multiprocessing.cpu_count()

    print(f"Generating SSP spectra and surviving stellar masses on {num_cores} cores...")

    tasks = []
    for age in ages_gyr:
        for logzsol in logzsol_grid:
            tasks.append((age, logzsol))

    with tqdm_joblib(total=len(tasks), desc="Generating SSPs with Bagpipes") as progress_bar:
        results = Parallel(n_jobs=num_cores, verbose=0, initializer=init_ssp_worker,
                           initargs=(gas_logu, rest_frame_wave))( # Pass the target wave to initializer
            delayed(_generate_single_ssp)(age, logzsol)
            for age, logzsol in tasks
        )

    k = 0
    for i_age, age in enumerate(ages_gyr):
        for i_z, logzsol in enumerate(logzsol_grid):
            spec_stellar_continuum, spec_nebular_emission, stellar_mass = results[k]
            ssp_stellar_continuum_spectra[i_age, i_z, :] = spec_stellar_continuum
            ssp_nebular_emission_spectra[i_age, i_z, :] = spec_nebular_emission
            ssp_stellar_masses[i_age, i_z] = stellar_mass
            k += 1

    with h5py.File(output_filename, 'w') as f:
        f.create_dataset('wavelength', data=rest_frame_wave, compression="gzip")
        f.create_dataset('ages_gyr', data=ages_gyr, compression="gzip")
        f.create_dataset('logzsol', data=logzsol_grid, compression="gzip")
        f.create_dataset('stellar_continuum_spectra', data=ssp_stellar_continuum_spectra, compression="gzip")
        f.create_dataset('nebular_emission_spectra', data=ssp_nebular_emission_spectra, compression="gzip")
        f.create_dataset('stellar_mass', data=ssp_stellar_masses, compression="gzip")

        f.attrs['imf_type'] = 'Kroupa (2001)'
        f.attrs['gas_logu'] = gas_logu
        f.attrs['z_sun'] = BAGPIPES_Z_SUN
        f.attrs['flux_unit'] = 'L_sun/Angstrom'
        f.attrs['code'] = 'Bagpipes'

    print(f"SSP grid generation complete. Saved to '{output_filename}'.")
    return output_filename