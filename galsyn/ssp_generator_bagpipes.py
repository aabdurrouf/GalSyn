import h5py
import numpy as np
import bagpipes as pipes
import os
from tqdm.auto import tqdm
from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib
import multiprocessing

# Constants
L_SUN_ERG_S = 3.828e33
BAGPIPES_Z_SUN = 0.02

_ssp_worker_bagpipes_instance = None

def init_ssp_worker(add_neb_emission_val, gas_logu_val):
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

    _ssp_worker_bagpipes_instance = model_components


def _generate_single_ssp(age, logzsol, rest_frame_wave):
    global _ssp_worker_bagpipes_instance

    metallicity_z_zsun = 10**logzsol

    burst = {}
    burst["age"] = age
    burst["massformed"] = 1.0
    burst["metallicity"] = metallicity_z_zsun

    current_model_components = _ssp_worker_bagpipes_instance.copy()
    current_model_components["burst"] = burst

    # Use the provided rest_frame_wave for spectrum generation
    model = pipes.model_galaxy(current_model_components, spec_wavs=rest_frame_wave)

    rest_frame_fluxes_erg_s_aa = model.spectrum_full

    rest_frame_fluxes_l_sun_aa = rest_frame_fluxes_erg_s_aa / L_SUN_ERG_S

    surv_stellar_mass = model.sfh.stellar_mass

    return rest_frame_fluxes_l_sun_aa, surv_stellar_mass


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

    # --- Start of modification in ssp_generator_bagpipes.py ---
    # Perform a dummy Bagpipes call to get the default wavelength array
    # This ensures consistency with Bagpipes' internal wavelength grid.
    # Initialize a temporary model with some arbitrary parameters to get the default wavelength grid
    temp_dust = {"type": "Calzetti", "Av": 0.0, "eta": 1.0}
    temp_nebular = {"logU": gas_logu} if add_neb_emission else None
    temp_model_components = {
        "redshift": 0.0,
        "veldisp": 0,
        "dust": temp_dust
    }
    if temp_nebular:
        temp_model_components["nebular"] = temp_nebular
    
    # Add a burst component for the dummy call
    temp_model_components["burst"] = {"age": 1.0, "massformed": 1.0, "metallicity": 1.0}
    
    # Create a Bagpipes model to extract the default wavelengths
    # Do not specify spec_wavs here, so Bagpipes uses its internal default.
    temp_model = pipes.model_galaxy(temp_model_components)
    rest_frame_wave = temp_model.wavelengths # Get Bagpipes' default wavelength array
    # --- End of modification in ssp_generator_bagpipes.py ---

    ssp_spectra = np.zeros((len(ages_gyr), len(logzsol_grid), len(rest_frame_wave)), dtype=np.float32)
    ssp_stellar_masses = np.zeros((len(ages_gyr), len(logzsol_grid)), dtype=np.float32)

    num_cores = n_jobs
    if num_cores == -1:
        num_cores = multiprocessing.cpu_count()

    print(f"Generating SSP spectra and surviving stellar masses on {num_cores} cores...")

    tasks = []
    for age in ages_gyr:
        for logzsol in logzsol_grid:
            tasks.append((age, logzsol, rest_frame_wave)) # Pass the determined rest_frame_wave

    with tqdm_joblib(total=len(tasks), desc="Generating SSPs with Bagpipes") as progress_bar:
        results = Parallel(n_jobs=num_cores, verbose=0, initializer=init_ssp_worker,
                           initargs=(add_neb_emission, gas_logu))(
            delayed(_generate_single_ssp)(age, logzsol, rest_frame_wave)
            for age, logzsol, rest_frame_wave in tasks
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