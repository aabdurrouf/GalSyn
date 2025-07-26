import h5py
import numpy as np
import bagpipes as pipes
import os
from tqdm.auto import tqdm
from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib
import multiprocessing

# Constants
# Solar luminosity in erg/s (from Bagpipes documentation or common usage)
# L_sun = 3.828e33 erg/s
L_SUN_ERG_S = 3.828e33

BAGPIPES_Z_SUN = 0.02

# Global variable for the Bagpipes model_galaxy instance in each worker process
# This will be initialized once per worker by the initializer function.
_ssp_worker_bagpipes_instance = None

def init_ssp_worker(add_neb_emission_val, gas_logu_val):
    """
    Initializer function for each worker process in the SSP generation.
    Initializes the common components for the Bagpipes model.
    """
    global _ssp_worker_bagpipes_instance

    # These components are fixed for all SSPs in the grid
    dust = {}
    dust["type"] = "Calzetti"
    dust["Av"] = 0.0  # No dust for SSP generation
    dust["eta"] = 1.0

    nebular = {}
    if add_neb_emission_val:
        nebular["logU"] = gas_logu_val
    else:
        nebular = None # No nebular emission if turned off

    model_components = {}
    model_components["redshift"] = 0.0  # Always rest-frame
    model_components["veldisp"] = 0     # No velocity dispersion for SSPs
    model_components["dust"] = dust
    if nebular:
        model_components["nebular"] = nebular

    _ssp_worker_bagpipes_instance = model_components


def _generate_single_ssp(age, logzsol, rest_frame_wave):
    """
    Helper function to generate a single SSP spectrum and its surviving stellar mass
    using Bagpipes. This function will be called in parallel and uses the pre-initialized
    Bagpipes model components.

    Parameters:
    -----------
    age : float
        Age of the SSP in Gyr.
    logzsol : float
        Logarithm of metallicity relative to solar (log10(Z/Z_sun)).
    rest_frame_wave : np.ndarray
        Wavelength array for the spectrum.

    Returns:
    --------
    tuple: (spectrum, stellar_mass)
        spectrum : np.ndarray
            SSP spectrum in L_sun/AA.
        stellar_mass : float
            Surviving stellar mass of the SSP.
    """
    global _ssp_worker_bagpipes_instance

    # Convert logzsol back to Z/Z_sun for Bagpipes
    # Bagpipes expects metallicity in units of Z/Z_sun
    metallicity_z_zsun = 10**logzsol

    burst = {}
    burst["age"] = age
    burst["massformed"] = 1.0  # Mass formed is 1 solar mass for SSP
    burst["metallicity"] = metallicity_z_zsun # Bagpipes uses Z/Z_sun

    # Create a copy of the base model components and add the burst component
    current_model_components = _ssp_worker_bagpipes_instance.copy()
    current_model_components["burst"] = burst

    # Generate the model galaxy
    model = pipes.model_galaxy(current_model_components, spec_wavs=rest_frame_wave)

    # Get spectrum in erg/s/Angstrom
    rest_frame_fluxes_erg_s_aa = model.spectrum_full

    # Convert spectrum from erg/s/Angstrom to L_sun/Angstrom
    # L_sun/Angstrom = (erg/s/Angstrom) / (L_SUN_ERG_S)
    rest_frame_fluxes_l_sun_aa = rest_frame_fluxes_erg_s_aa / L_SUN_ERG_S

    # Get the surviving stellar mass for this SSP
    surv_stellar_mass = model.sfh.stellar_mass

    return rest_frame_fluxes_l_sun_aa, surv_stellar_mass


def generate_ssp_grid_bagpipes(output_filename="ssp_spectra_bagpipes.hdf5",
                               ages_gyr=None,
                               logzsol_grid=None,
                               add_neb_emission=True,
                               gas_logu=-2.0,
                               overwrite=False,
                               n_jobs=-1):
    """
    Generates a grid of Simple Stellar Population (SSP) spectra and their
    corresponding surviving stellar masses using Bagpipes, and saves them to an HDF5 file.
    Supports parallel computation by initializing Bagpipes components once per worker process.

    Parameters:
    -----------
    output_filename : str, optional
        The name of the HDF5 file to save the SSP grid.
        Defaults to "ssp_spectra_bagpipes.hdf5".
    ages_gyr : np.ndarray, optional
        A 1D numpy array of ages in Gyr for which to generate SSP spectra.
        If None, a default logarithmic grid from 0.001 Gyr to 13.8 Gyr is used.
    logzsol_grid : np.ndarray, optional
        A 1D numpy array of log10(Z/Z_sun) values for which to generate SSP spectra.
        If None, a default linear grid from -2.0 to 0.2 is used.
    add_neb_emission : bool, optional
        Whether to add nebular emission. Defaults to True.
    gas_logu : float, optional
        Logarithm of the ionization parameter for nebular emission. Defaults to -2.0.
        Only used if add_neb_emission is True.
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

    print(f"Generating SSP grid and saving to {output_filename} using Bagpipes...")

    # Define default age grid if not provided
    if ages_gyr is None:
        # Logarithmic grid from 1 Myr (0.001 Gyr) to 13.8 Gyr
        ages_gyr = np.logspace(np.log10(0.001), np.log10(13.8), 100) # 100 ages

    # Define default metallicity grid if not provided
    if logzsol_grid is None:
        # Linear grid for log10(Z/Z_sun)
        logzsol_grid = np.linspace(-2.0, 0.2, 20) # 20 metallicities

    # Define the rest-frame wavelength array for Bagpipes
    rest_frame_wave = np.arange(100., 30000., 5.)

    # Initialize arrays to store spectra and surviving stellar masses
    # Dimensions: (num_ages, num_metallicities, num_wavelengths) for spectra
    # Dimensions: (num_ages, num_metallicities) for stellar_mass
    ssp_spectra = np.zeros((len(ages_gyr), len(logzsol_grid), len(rest_frame_wave)), dtype=np.float32)
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
            tasks.append((age, logzsol, rest_frame_wave))

    # Execute tasks in parallel, with Bagpipes components initialized once per worker
    with tqdm_joblib(total=len(tasks), desc="Generating SSPs with Bagpipes") as progress_bar:
        results = Parallel(n_jobs=num_cores, verbose=0, initializer=init_ssp_worker,
                           initargs=(add_neb_emission, gas_logu))(
            delayed(_generate_single_ssp)(age, logzsol, rest_frame_wave)
            for age, logzsol, rest_frame_wave in tasks
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
        f.create_dataset('wavelength', data=rest_frame_wave, compression="gzip")
        f.create_dataset('ages_gyr', data=ages_gyr, compression="gzip")
        f.create_dataset('logzsol', data=logzsol_grid, compression="gzip")
        f.create_dataset('spectra', data=ssp_spectra, compression="gzip")
        f.create_dataset('stellar_mass', data=ssp_stellar_masses, compression="gzip") # Save surviving stellar mass

        # Store Bagpipes specific attributes
        f.attrs['imf_type'] = 'Kroupa (2001)'
        f.attrs['add_neb_emission'] = add_neb_emission
        f.attrs['gas_logu'] = gas_logu
        f.attrs['z_sun'] = BAGPIPES_Z_SUN # Store Z_sun used by Bagpipes
        f.attrs['flux_unit'] = 'L_sun/Angstrom'
        f.attrs['code'] = 'Bagpipes'

    print(f"SSP grid generation complete. Saved to '{output_filename}'.")
    return output_filename