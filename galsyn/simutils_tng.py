import os, sys 
import numpy as np
from astropy.io import fits 
from .imgutils import *
from .utils import *

baseUrl_tng = 'http://www.tng-project.org/api/'
headers = {}

def get(path, params=None):
    import requests

    # make HTTP GET request to path
    r = requests.get(path, params=params, headers=headers)

    # raise exception if response code is not HTTP SUCCESS (200)
    r.raise_for_status()

    if r.headers['content-type'] == 'application/json':
        return r.json() # parse json responses automatically

    if 'content-disposition' in r.headers:
        filename = r.headers['content-disposition'].split("filename=")[1]
        with open(filename, 'wb') as f:
            f.write(r.content)
        return filename # return the filename string

    return r

def get_tng_snaps_info(sim='TNG50-1', api_key="api-key"):
    global headers
    headers = {"api-key":api_key}
    r = get(baseUrl_tng)
    names = [sim['name'] for sim in r['simulations']]
    sim_info = get( r['simulations'][names.index(sim)]['url'])
    snaps = get(sim_info['snapshots'])
    return snaps

def get_snap_z(snap_number, sim='TNG50-1', snaps_info=None, api_key="api-key"):
    if snaps_info is None:
        snaps_info = get_tng_snaps_info(sim, api_key=api_key)
    return snaps_info[int(snap_number)]['redshift']

def get_snap_z_batch(snap_numbers, sim='TNG50-1', snaps_info=None, api_key="api-key"):
    if snaps_info is None:
        snaps_info = get_tng_snaps_info(sim, api_key=api_key)

    snap_z = []
    for ii in snap_numbers:
        snap_z.append(snaps_info[ii]['redshift'])
 
    return np.asarray(snap_z)

def get_num_subhalos(snap_number, sim='TNG50-1', snaps_info=None, api_key="api-key"):
    if snaps_info is None:
        snaps_info = get_tng_snaps_info(sim, api_key=api_key)
    return snaps_info[int(snap_number)]['num_groups_subfind']

def cosmic_times_snapshots(sim='TNG50-1',snaps_info=None, cosmo='Planck18', api_key="api-key"):

    if snaps_info is None:
        snaps_info = get_tng_snaps_info(sim, api_key=api_key)

    nsnaps = 100
    cosmic_times = np.zeros(nsnaps)
    for ii in range(nsnaps):
        snap_z = get_snap_z(ii, sim=sim, snaps_info=snaps_info, api_key=api_key)
        cosmic_times[ii] = cosmo.age(snap_z).value

    return cosmic_times

def cosmic_times_of_snapshots(snaps, sim='TNG50-1', snaps_info=None, cosmo='Planck18', api_key="api-key"):

    if snaps_info is None:
        snaps_info = get_tng_snaps_info(sim, api_key=api_key)

    cosmic_times = []
    for ii in snaps:
        snap_z = get_snap_z(ii, sim=sim, snaps_info=snaps_info, api_key=api_key)
        cosmic_times.append(cosmo.age(snap_z).value)

    return np.asarray(cosmic_times)

def download_cutout_subhalo_hdf5(snap_number, subhalo_id, api_key="api-key", sim='TNG50-1', params=None):
    global headers
    headers = {"api-key":api_key}
    url = "http://www.tng-project.org/api/" + sim + "/snapshots/" + str(int(snap_number)) + "/subhalos/" + str(int(subhalo_id))
    sub = get(url, params=params)
    name0 = get(sub['cutouts']['subhalo'],params)
    name = 'cutout_shalo_'+str(int(snap_number))+'_'+str(int(subhalo_id))+'.hdf5'
    os.rename(name0, name)
    return name 

def download_cutout_parent_halo_hdf5(snap_number, subhalo_id, api_key="api-key", sim='TNG50-1', params=None):
    global headers
    headers = {"api-key":api_key}
    url = "http://www.tng-project.org/api/" + sim + "/snapshots/" + str(int(snap_number)) + "/subhalos/" + str(int(subhalo_id))
    sub = get(url, params=params)
    name0 = get(sub['cutouts']['parent_halo'],params)
    name = 'cutout_phalo_'+str(int(snap_number))+'_'+str(int(subhalo_id))+'.hdf5'
    os.rename(name0, name)
    return name

def get_basic_subhalo_properties(snap_number, subhalo_id, api_key="api-key", sim='TNG50-1', params=None):
    global headers
    headers = {"api-key":api_key}
    url = "http://www.tng-project.org/api/" + sim + "/snapshots/" + str(int(snap_number)) + "/subhalos/" + str(int(subhalo_id))
    sub = get(url, params=params)
    return sub

def make_sim_file_from_tng_data(input_hdf5, z, cosmo_h=0.6774, XH=0.76, output_hdf5='sim_file_tng.hdf5'):
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
    
    f.close()

    create_hdf5_file(output_hdf5, stars_init_mass, stars_form_z, stars_mass, stars_zmet, stars_coords,
                    stars_vel, gas_mass, gas_zmet, gas_sfr_inst, gas_temp, gas_coords, gas_vel, gas_mass_H)






