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

def init_ssp_worker(add_neb_emission_val, gas_logu_val, rest_frame_wave_val):
    """
    Initializer function for each worker process in the SSP generation.
    Initializes the common components for the Bagpipes model.
    """
    global _ssp_worker_bagpipes_instance

    dust = {}
    dust["type"] = "Calzetti"
    dust["Av"] = 0.0
    dust["eta"] = 1.0

    nebular = {}
    if add_neb_emission_val:
        nebular["logU"] = gas_logu_val
    else:
        nebular = None

    model_components = {}
    model_components["redshift"] = 0.0
    model_components["veldisp"] = 0
    model_components["dust"] = dust
    if nebular:
        model_components["nebular"] = nebular
    
    # Store the target rest_frame_wave for use by _generate_single_ssp
    model_components["_rest_frame_wave_target"] = rest_frame_wave_val

    _ssp_worker_bagpipes_instance = model_components


def _generate_single_ssp(age, logzsol):
    """
    Helper function to generate a single SSP spectrum and its surviving stellar mass
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

    current_model_components = _ssp_worker_bagpipes_instance.copy()
    current_model_components["burst"] = burst

    # Generate the full internal spectrum from Bagpipes.
    # Do NOT pass spec_wavs here if model.spectrum_full is always the full internal one.
    model = pipes.model_galaxy(current_model_components, spec_wavs=np.arange(1000., 10000., 5.))

    # Get the full wavelength and flux arrays from Bagpipes' internal calculation
    full_wave = model.wavelengths
    full_fluxes_erg_s_aa = model.spectrum_full

    # Convert to L_sun/Angstrom
    full_fluxes_l_sun_aa = full_fluxes_erg_s_aa / L_SUN_ERG_S

    # Perform linear interpolation onto the target wavelength grid
    interp_func = interp1d(full_wave, full_fluxes_l_sun_aa, kind='linear', 
                           bounds_error=False, fill_value=0.0)
    
    spec_interpolated = interp_func(rest_frame_wave_target)

    surv_stellar_mass = model.sfh.stellar_mass

    return spec_interpolated, surv_stellar_mass


def generate_ssp_grid_bagpipes(output_filename="ssp_spectra_bagpipes.hdf5",
                               ages_gyr=None,
                               logzsol_grid=None,
                               add_neb_emission=True,
                               gas_logu=-2.0,
                               overwrite=False,
                               n_jobs=-1):

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
    ssp_spectra = np.zeros((len(ages_gyr), len(logzsol_grid), len(rest_frame_wave)), dtype=np.float32)
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
                           initargs=(add_neb_emission, gas_logu, rest_frame_wave))( # Pass the target wave to initializer
            delayed(_generate_single_ssp)(age, logzsol)
            for age, logzsol in tasks
        )

    k = 0
    for i_age, age in enumerate(ages_gyr):
        for i_z, logzsol in enumerate(logzsol_grid):
            spec, stellar_mass = results[k]
            ssp_spectra[i_age, i_z, :] = spec
            ssp_stellar_masses[i_age, i_z] = stellar_mass
            k += 1

    with h5py.File(output_filename, 'w') as f:
        f.create_dataset('wavelength', data=rest_frame_wave, compression="gzip")
        f.create_dataset('ages_gyr', data=ages_gyr, compression="gzip")
        f.create_dataset('logzsol', data=logzsol_grid, compression="gzip")
        f.create_dataset('spectra', data=ssp_spectra, compression="gzip")
        f.create_dataset('stellar_mass', data=ssp_stellar_masses, compression="gzip")

        f.attrs['imf_type'] = 'Kroupa (2001)'
        f.attrs['add_neb_emission'] = add_neb_emission
        f.attrs['gas_logu'] = gas_logu
        f.attrs['z_sun'] = BAGPIPES_Z_SUN
        f.attrs['flux_unit'] = 'L_sun/Angstrom'
        f.attrs['code'] = 'Bagpipes'

    print(f"SSP grid generation complete. Saved to '{output_filename}'.")
    return output_filename