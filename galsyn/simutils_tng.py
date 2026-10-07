import os, sys 
import numpy as np
from astropy.io import fits 
from .imgutils import *
from .utils import *

baseUrl_tng = 'https://www.tng-project.org/api/'
headers = {}

# Particle types and fields needed by make_sim_file_from_tng_data. Skipping dark
# matter and black holes (and unused fields) greatly reduces the size of the
# download and avoids server-side 504 time-outs for massive subhalos.
CUTOUT_PARAMS_STARS_GAS = {
    "gas": "Coordinates,Masses,Velocities,GFM_Metallicity,StarFormationRate,InternalEnergy,ElectronAbundance",
    "stars": "Coordinates,Masses,Velocities,GFM_InitialMass,GFM_StellarFormationTime,GFM_Metallicity",
}

# Maps the TNG API particle-type query keys to their HDF5 group names.
PARTTYPE_GROUP = {
    "gas": "PartType0",
    "dm": "PartType1",
    "tracers": "PartType3",
    "stars": "PartType4",
    "bhs": "PartType5",
}

def get(path, params=None, api_key=None, max_retries=8, out_dir=".", max_wait=180, filename=None):
    """
    Handles TNG API requests for JSON metadata and HDF5 cutouts.

    Downloads are streamed to a temporary file in `out_dir` and atomically renamed
    on success. Failed requests (including 504 time-outs) are retried with
    exponential back-off. `filename`, if given, overrides the name suggested by the
    server; this is required when several requests for the same subhalo are made
    into one directory, since the server suggests the same name for all of them.
    If `api_key` is None, the module-level `headers` is used.
    """
    import time
    import requests

    hdrs = {"api-key": api_key} if api_key is not None else headers

    for attempt in range(max_retries):
        try:
            r = requests.get(path, params=params, headers=hdrs, stream=True, timeout=(15, 300))
            r.raise_for_status()

            if "json" in r.headers.get("content-type", "").lower():
                return r.json()

            dest = filename
            disp = r.headers.get("content-disposition", "")
            if not dest and "filename=" in disp:
                dest = disp.split("filename=")[1].strip('" \t')

            if not dest:
                return r

            final_path = os.path.join(out_dir, dest)
            part_path = f"{final_path}.tmp_{os.getpid()}"
            with open(part_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=4 * 1024 * 1024):
                    if chunk:
                        f.write(chunk)
            os.replace(part_path, final_path)
            return final_path

        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                break
            wait = min(5 * (2 ** attempt), max_wait)
            print(f"[Retry {attempt+1}/{max_retries}] Failed for {path}: {e}. Retrying in {wait}s...")
            time.sleep(wait)

    raise RuntimeError(f"Failed to fetch data from {path} after {max_retries} attempts.")

def _merge_particle_cutouts(file_paths_by_ptype, target_path):
    """
    Merges single-particle-type cutout files into one HDF5 file at `target_path`,
    each under its normal PartTypeN group. Particle types absent from the subhalo
    (e.g. gas in a gas-free galaxy) are skipped with a note.
    """
    import h5py

    with h5py.File(target_path, "w") as fout:
        for ptype, path in file_paths_by_ptype.items():
            group_name = PARTTYPE_GROUP[ptype]
            with h5py.File(path, "r") as fin:
                if group_name in fin:
                    fin.copy(fin[group_name], fout, name=group_name)
                else:
                    print(f"Note: no {group_name} ({ptype}) data found for this subhalo.")

def get_tng_snaps_info(sim='TNG50-1', api_key="api-key"):
    """
    Retrieves metadata for all snapshots of a given TNG simulation.

    Args:
        sim (str): The name of the TNG simulation (e.g., 'TNG50-1').
        api_key (str): Your personal TNG API key.

    Returns:
        list: A list of dictionaries, each containing information for a snapshot.
    """
    global headers
    headers = {"api-key":api_key}
    r = get(baseUrl_tng)
    names = [sim['name'] for sim in r['simulations']]
    sim_info = get( r['simulations'][names.index(sim)]['url'])
    snaps = get(sim_info['snapshots'])
    return snaps

def get_snap_z(snap_number, sim='TNG50-1', snaps_info=None, api_key="api-key"):
    """
    Gets the redshift for a single snapshot number.

    Args:
        snap_number (int): The snapshot number.
        sim (str): The name of the TNG simulation.
        snaps_info (list, optional): Pre-fetched snapshot info to avoid a new API call.
        api_key (str): Your TNG API key, used if snaps_info is not provided.

    Returns:
        float: The redshift of the specified snapshot.
    """
    if snaps_info is None:
        snaps_info = get_tng_snaps_info(sim, api_key=api_key)
    return snaps_info[int(snap_number)]['redshift']

def get_snap_z_batch(snap_numbers, sim='TNG50-1', snaps_info=None, api_key="api-key"):
    """
    Gets the redshifts for a list of snapshot numbers.

    Args:
        snap_numbers (list of int): A list of snapshot numbers.
        sim (str): The name of the TNG simulation.
        snaps_info (list, optional): Pre-fetched snapshot info to avoid new API calls.
        api_key (str): Your TNG API key, used if snaps_info is not provided.

    Returns:
        np.ndarray: An array of redshifts corresponding to the input snapshots.
    """
    if snaps_info is None:
        snaps_info = get_tng_snaps_info(sim, api_key=api_key)

    snap_z = []
    for ii in snap_numbers:
        snap_z.append(snaps_info[ii]['redshift'])
 
    return np.asarray(snap_z)

def get_num_subhalos(snap_number, sim='TNG50-1', snaps_info=None, api_key="api-key"):
    """
    Gets the total number of subhalos in a specific snapshot.

    Args:
        snap_number (int): The snapshot number.
        sim (str): The name of the TNG simulation.
        snaps_info (list, optional): Pre-fetched snapshot info to avoid a new API call.
        api_key (str): Your TNG API key, used if snaps_info is not provided.

    Returns:
        int: The number of subhalos.
    """
    if snaps_info is None:
        snaps_info = get_tng_snaps_info(sim, api_key=api_key)
    return snaps_info[int(snap_number)]['num_groups_subfind']

def cosmic_times_snapshots(sim='TNG50-1',snaps_info=None, cosmo='Planck18', api_key="api-key"):
    """
    Calculates the age of the universe (cosmic time) for all 100 snapshots.

    Args:
        sim (str): The name of the TNG simulation.
        snaps_info (list, optional): Pre-fetched snapshot info.
        cosmo (str): The cosmological model for age calculation.
        api_key (str): Your TNG API key.

    Returns:
        np.ndarray: An array of cosmic times in Gyr for each snapshot from 0 to 99.
    """
    if snaps_info is None:
        snaps_info = get_tng_snaps_info(sim, api_key=api_key)

    nsnaps = 100
    cosmic_times = np.zeros(nsnaps)
    for ii in range(nsnaps):
        snap_z = get_snap_z(ii, sim=sim, snaps_info=snaps_info, api_key=api_key)
        cosmic_times[ii] = cosmo.age(snap_z).value

    return cosmic_times

def cosmic_times_of_snapshots(snaps, sim='TNG50-1', snaps_info=None, cosmo='Planck18', api_key="api-key"):
    """
    Calculates the age of the universe (cosmic time) for a specific list of snapshots.

    Args:
        snaps (list of int): A list of snapshot numbers.
        sim (str): The name of the TNG simulation.
        snaps_info (list, optional): Pre-fetched snapshot info.
        cosmo (str): The cosmological model for age calculation.
        api_key (str): Your TNG API key.

    Returns:
        np.ndarray: An array of cosmic times in Gyr for the specified snapshots.
    """
    if snaps_info is None:
        snaps_info = get_tng_snaps_info(sim, api_key=api_key)

    cosmic_times = []
    for ii in snaps:
        snap_z = get_snap_z(ii, sim=sim, snaps_info=snaps_info, api_key=api_key)
        cosmic_times.append(cosmo.age(snap_z).value)

    return np.asarray(cosmic_times)

def _download_cutout(cutout_url, name, tag, api_key, params, split_requests):
    """
    Shared download logic: fetches `cutout_url` to `name`, optionally one particle
    type per request (merged locally). `tag` labels temporary files.
    """
    out_dir = os.path.dirname(name) or "."
    os.makedirs(out_dir, exist_ok=True)

    if split_requests and params and len(params) > 1:
        temp_paths = {}
        try:
            for ptype, fields in params.items():
                print(f"Fetching {ptype} particles for {tag}...")
                temp_paths[ptype] = get(cutout_url, params={ptype: fields}, api_key=api_key,
                                        out_dir=out_dir,
                                        filename=f"_split_{ptype}_{tag}_{os.getpid()}.hdf5")
            _merge_particle_cutouts(temp_paths, name)
        finally:
            for path in temp_paths.values():
                if path and os.path.exists(path):
                    os.remove(path)
        return name

    tmp = get(cutout_url, params=params, api_key=api_key, out_dir=out_dir,
              filename=f"_tmp_{tag}_{os.getpid()}.hdf5")
    os.replace(tmp, name)
    return name

def download_cutout_subhalo_hdf5(snap_number, subhalo_id, api_key="api-key", sim='TNG50-1',
                                 params=CUTOUT_PARAMS_STARS_GAS, name=None, split_requests=True):
    """
    Downloads the HDF5 data cutout for a specific subhalo.

    By default only the star and gas particles, and only the fields required by
    `make_sim_file_from_tng_data`, are requested (see CUTOUT_PARAMS_STARS_GAS). This
    keeps the download small and avoids time-outs for massive galaxies.

    Args:
        snap_number (int): The snapshot number.
        subhalo_id (int): The ID of the target subhalo.
        api_key (str): Your TNG API key.
        sim (str): The name of the TNG simulation.
        params (dict, optional): Particle-type -> comma-separated fields (or 'all'),
                                 e.g., {'gas':'Coordinates,Masses', 'stars':'all'}.
                                 Pass None to download the full cutout.
        name (str, optional): Desired output name.
        split_requests (bool): If True and `params` has several particle types, each
                               type is requested separately and merged locally,
                               which lowers the load on the server per request.

    Returns:
        str: The filename of the downloaded HDF5 file.
    """
    snap_number, subhalo_id = int(snap_number), int(subhalo_id)
    url = f"{baseUrl_tng}{sim}/snapshots/{snap_number}/subhalos/{subhalo_id}/"
    sub = get(url, api_key=api_key)
    cutout_url = sub['cutouts']['subhalo']

    if name is None:
        name = f'cutout_shalo_{snap_number}_{subhalo_id}.hdf5'
    return _download_cutout(cutout_url, name, f"{snap_number}_{subhalo_id}", api_key, params, split_requests)

def download_cutout_parent_halo_hdf5(snap_number, subhalo_id, api_key="api-key", sim='TNG50-1',
                                     params=CUTOUT_PARAMS_STARS_GAS, name=None, split_requests=True):
    """
    Downloads the HDF5 data cutout for the parent halo of a specified subhalo.

    Parent halos are much larger than subhalos, so by default only star and gas
    particles with the fields in CUTOUT_PARAMS_STARS_GAS are requested, one particle
    type per request (merged locally), with retries on time-outs.

    Args:
        snap_number (int): The snapshot number of the subhalo.
        subhalo_id (int): The ID of the target subhalo.
        api_key (str): Your TNG API key.
        sim (str): The name of the TNG simulation.
        params (dict, optional): Particle-type -> comma-separated fields (or 'all').
                                 Pass None to download the full cutout.
        name (str, optional): Desired output name.
        split_requests (bool): Request each particle type separately and merge locally.

    Returns:
        str: The filename of the downloaded HDF5 file.
    """
    snap_number, subhalo_id = int(snap_number), int(subhalo_id)
    url = f"{baseUrl_tng}{sim}/snapshots/{snap_number}/subhalos/{subhalo_id}/"
    sub = get(url, api_key=api_key)
    if name is None:
        name = f'cutout_phalo_{snap_number}_{subhalo_id}.hdf5'
    return _download_cutout(sub['cutouts']['parent_halo'], name, f"phalo_{snap_number}_{subhalo_id}",
                            api_key, params, split_requests)

def get_basic_subhalo_properties(snap_number, subhalo_id, api_key="api-key", sim='TNG50-1', params=None):
    """
    Fetches basic properties (metadata) for a specific subhalo.

    This returns a JSON object with information like mass, position, etc.,
    without downloading the full particle data cutout.

    Args:
        snap_number (int): The snapshot number.
        subhalo_id (int): The ID of the target subhalo.
        api_key (str): Your TNG API key.
        sim (str): The name of the TNG simulation.

    Returns:
        dict: A dictionary containing the subhalo's properties.
    """
    global headers
    headers = {"api-key":api_key}
    url = "http://www.tng-project.org/api/" + sim + "/snapshots/" + str(int(snap_number)) + "/subhalos/" + str(int(subhalo_id))
    sub = get(url, params=params)
    return sub

def make_sim_file_from_tng_data(input_hdf5, z, cosmo_h=0.6774, XH=0.76, output_hdf5='sim_file_tng.hdf5'):
    """
    Converts a raw TNG cutout into a standardized HDF5 file for analysis.

    This function extracts star and gas particle data, converts units from
    TNG-specific conventions to physical units (e.g., kpc, Msun), calculates
    additional properties like gas temperature, and saves the result to a
    new HDF5 file.

    Args:
        input_hdf5 (str): Path to the raw TNG cutout HDF5 file to process.
        z (float): The redshift of the snapshot.
        cosmo_h (float): The dimensionless Hubble parameter 'h'.
        XH (float): The primordial mass fraction of hydrogen.
        output_hdf5 (str): The path for the new, processed HDF5 file.

    Returns:
        None
    """
    
    import h5py

    f = h5py.File(input_hdf5,'r')

    # get star particles data
    stars_init_mass = f['PartType4']['GFM_InitialMass'][:] * 1e+10 / cosmo_h
    stars_form_a = f['PartType4']['GFM_StellarFormationTime'][:]
    stars_form_z = (1.0/stars_form_a) - 1.0
    stars_mass = f['PartType4']['Masses'][:] * 1e+10 / cosmo_h
    stars_zmet = f['PartType4']['GFM_Metallicity'][:]
    snap_a = 1.0/(1.0 + z)
    stars_coords = f['PartType4']['Coordinates'][:] * snap_a / cosmo_h  # in kpc
    stars_vel = f['PartType4']['Velocities'][:] * np.sqrt(snap_a)  # peculiar velocities in km/s

    idx = np.where(stars_form_a>0)[0]
    stars_init_mass = stars_init_mass[idx]
    stars_form_z = stars_form_z[idx]
    stars_mass = stars_mass[idx]
    stars_zmet = stars_zmet[idx]
    stars_coords = stars_coords[idx,:]
    stars_vel = stars_vel[idx,:]

    # get gas particles data
    if 'PartType0' in f:
        gas_mass = f['PartType0']['Masses'][:] * 1e+10 / cosmo_h
        gas_zmet = f['PartType0']['GFM_Metallicity'][:]
        gas_sfr_inst = f['PartType0']['StarFormationRate'][:]   # in Msun/yr
        u = f['PartType0']['InternalEnergy'][:]
        Xe = f['PartType0']['ElectronAbundance'][:]
        gamma = 5.0/3.0
        KB = 1.3807e-16
        mp = 1.6726e-24
        mu = (4*mp)/(1 + (3*XH) + (4*XH*Xe))
        gas_temp = (gamma-1.0)*(u/KB)*mu*1e+10
        gas_coords = f['PartType0']['Coordinates'][:] * snap_a / cosmo_h   # in kpc
        gas_vel = f['PartType0']['Velocities'][:] * np.sqrt(snap_a)   # peculiar velocity in km/s
        gas_mass_H = gas_mass * XH

    else:
        gas_mass = [0]
        gas_zmet = [0]
        gas_sfr_inst = [0]
        gas_temp = [0]
        gas_coords = np.zeros((1,3))
        gas_vel = np.zeros((1,3))
        gas_mass_H = [0]
    
    f.close()

    create_hdf5_file(output_hdf5, stars_init_mass, stars_form_z, stars_mass, stars_zmet, stars_coords,
                    stars_vel, gas_mass, gas_zmet, gas_sfr_inst, gas_temp, gas_coords, gas_vel, gas_mass_H)






