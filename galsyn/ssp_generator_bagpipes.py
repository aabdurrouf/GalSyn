import h5py
import numpy as np
import bagpipes as pipes
import os
from tqdm.auto import tqdm
from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib
import multiprocessing
from scipy.interpolate import interp1d

# Constants
L_SUN_ERG_S = 3.828e33
BAGPIPES_Z_SUN = 0.02

_ssp_worker_bagpipes_instance = None

def init_ssp_worker(rest_frame_wave_val):
    """
    Initializer function for each worker process in the SSP generation.
    Initializes the common components for the Bagpipes model.
    """
    global _ssp_worker_bagpipes_instance
    
    dust = {"type": "Calzetti", "Av": 0.0, "eta": 0.0}

    model_components = {
        "redshift": 0.0,
        "veldisp": 0,
        "dust": dust,
        "_rest_frame_wave_target": rest_frame_wave_val
    }

    _ssp_worker_bagpipes_instance = model_components


def _generate_single_ssp(age, logzsol, logu):
    """
    Generates a single SSP spectrum with a specific log(U).
    """
    global _ssp_worker_bagpipes_instance

    rest_frame_wave_target = _ssp_worker_bagpipes_instance["_rest_frame_wave_target"]
    metallicity_z_zsun = 10**logzsol

    burst = {"age": age, "massformed": 0.0, "metallicity": metallicity_z_zsun}

    # Total Spectrum (Stellar + Nebular)
    current_model_components_total = _ssp_worker_bagpipes_instance.copy()
    current_model_components_total["burst"] = burst
    current_model_components_total["nebular"] = {"logU": logu}
    
    # We use a standard internal wave grid for Bagpipes generation
    model_total = pipes.model_galaxy(current_model_components_total, spec_wavs=np.arange(100., 35000., 5.))
    
    interp_func_total = interp1d(model_total.wavelengths, model_total.spectrum_full / L_SUN_ERG_S, 
                                 kind='linear', bounds_error=False, fill_value=0.0)
    spec_total_interpolated = interp_func_total(rest_frame_wave_target)

    # Stellar Continuum Only (Nebular disabled)
    current_model_components_stellar = _ssp_worker_bagpipes_instance.copy()
    current_model_components_stellar["burst"] = burst
    
    model_stellar = pipes.model_galaxy(current_model_components_stellar, spec_wavs=np.arange(100., 35000., 5.))
    
    interp_func_stellar = interp1d(model_stellar.wavelengths, model_stellar.spectrum_full / L_SUN_ERG_S, 
                                   kind='linear', bounds_error=False, fill_value=0.0)
    spec_stellar_continuum_interpolated = interp_func_stellar(rest_frame_wave_target)

    # Nebular Only
    spec_nebular_emission_interpolated = spec_total_interpolated - spec_stellar_continuum_interpolated

    # surviving stellar mass fraction
    surv_stellar_mass = 1.0 

    return spec_stellar_continuum_interpolated.astype(np.float32), \
           spec_nebular_emission_interpolated.astype(np.float32), \
           np.float32(surv_stellar_mass)


def generate_ssp_grid_bagpipes(output_filename="ssp_spectra_bagpipes.hdf5",
                               ages_gyr=None,
                               logzsol_grid=None,
                               logu_grid=None,
                               overwrite=False,
                               n_jobs=-1,
                               rest_wave_min=500,  
                               rest_wave_max=30000,
                               delta_wave=5.0):
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
        logu_grid (array-like, optional): A 1D numpy array of log(U) values for which to generate SSP spectra.
        overwrite (bool, optional): If True, an existing file at `output_filename`
                                    will be overwritten. If False, the function
                                    will exit if the file already exists.
                                    Defaults to False.
        n_jobs (int, optional): The number of CPU cores to use for parallel
                                processing. Defaults to -1 (use all available cores).
        rest_wave_min (float, optional): Minimum wavelength in Angstroms for the output spectra. Defaults to 500.
        rest_wave_max (float, optional): Maximum wavelength in Angstroms for the output spectra. Defaults to 30000.
        delta_wave (float, optional): Wavelength step in Angstroms for the output spectra. Defaults to 5.0.

    Returns:
        str: The path to the generated HDF5 file.
    """
    if os.path.exists(output_filename) and not overwrite:
        print(f"File '{output_filename}' exists. Set overwrite=True to regenerate.")
        return output_filename

    # Default grids if not provided
    if ages_gyr is None: 
        ages_gyr = np.logspace(np.log10(0.001), np.log10(13.8), 100)

    if logzsol_grid is None:
        logzsol_grid = np.linspace(-2.0, 0.2, 20)

    if logu_grid is None: 
        logu_grid = np.linspace(-4.0, -1.0, 10)

    # Define master wavelength grid
    rest_frame_wave = np.arange(rest_wave_min, rest_wave_max, delta_wave)

    # Initialize 4D spectra cube: (age, zstar, logu, wave)
    shape_spec = (len(ages_gyr), len(logzsol_grid), len(logu_grid), len(rest_frame_wave))
    shape_mass = (len(ages_gyr), len(logzsol_grid), len(logu_grid))
    
    s_cont_cube = np.zeros(shape_spec, dtype=np.float32)
    n_em_cube = np.zeros(shape_spec, dtype=np.float32)
    s_mass_cube = np.zeros(shape_mass, dtype=np.float32)

    num_cores = multiprocessing.cpu_count() if n_jobs == -1 else n_jobs
    
    # Create task list for 3D grid
    tasks = [(a, z, u) for a in ages_gyr for z in logzsol_grid for u in logu_grid]

    print(f"Generating 3D Bagpipes SSP grid on {num_cores} cores...")

    with tqdm_joblib(total=len(tasks), desc="Generating Bagpipes SSPs") as progress_bar:
        results = Parallel(n_jobs=num_cores, initializer=init_ssp_worker,
                           initargs=(rest_frame_wave,))(
            delayed(_generate_single_ssp)(*t) for t in tasks
        )

    # Reconstruct the grid from results
    k = 0
    for i_a in range(len(ages_gyr)):
        for i_z in range(len(logzsol_grid)):
            for i_u in range(len(logu_grid)):
                res_s, res_n, res_m = results[k]
                s_cont_cube[i_a, i_z, i_u, :] = res_s
                n_em_cube[i_a, i_z, i_u, :] = res_n
                s_mass_cube[i_a, i_z, i_u] = res_m
                k += 1

    with h5py.File(output_filename, 'w') as f:
        f.create_dataset('wavelength', data=rest_frame_wave, compression="gzip")
        f.create_dataset('ages_gyr', data=ages_gyr, compression="gzip")
        f.create_dataset('logzsol', data=logzsol_grid, compression="gzip")
        f.create_dataset('logu_grid', data=logu_grid, compression="gzip")
        f.create_dataset('stellar_continuum_spectra', data=s_cont_cube, compression="gzip")
        f.create_dataset('nebular_emission_spectra', data=n_em_cube, compression="gzip")
        f.create_dataset('stellar_mass', data=s_mass_cube, compression="gzip")

        f.attrs['imf_type'] = 'Kroupa (2001)'
        f.attrs['z_sun'] = BAGPIPES_Z_SUN
        f.attrs['code'] = 'Bagpipes'

    print(f"SSP grid generation complete. Saved to '{output_filename}'.")
    return output_filename