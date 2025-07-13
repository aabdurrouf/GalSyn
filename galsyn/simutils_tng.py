import os, sys 
import numpy as np
from astropy.io import fits 
from .imgutils import *
from .utils import *

from .config import API_KEY
headers = {"api-key":API_KEY}

cosmo, cosmo_h = define_cosmo()
imf_type, add_neb_emission, add_igm_absorption, igm_type, dust_index_bc, gas_logu, dust_index, t_esc = fsps_setup()

baseUrl_tng = 'http://www.tng-project.org/api/'

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

def get_tng_snaps_info(sim='TNG50-1'):
    r = get(baseUrl_tng)
    names = [sim['name'] for sim in r['simulations']]
    sim_info = get( r['simulations'][names.index(sim)]['url'] )
    snaps = get(sim_info['snapshots'])
    return snaps

def get_snap_z(snap_number, sim='TNG50-1', snaps_info=None):
    if snaps_info is None:
        snaps_info = get_tng_snaps_info(sim)
    return snaps_info[int(snap_number)]['redshift']

def get_snap_z_batch(snap_numbers, sim='TNG50-1', snaps_info=None):
    if snaps_info is None:
        snaps_info = get_tng_snaps_info(sim)

    snap_z = []
    for ii in snap_numbers:
        snap_z.append(snaps_info[ii]['redshift'])
 
    return np.asarray(snap_z)

def get_num_subhalos(snap_number, sim='TNG50-1', snaps_info=None):
    if snaps_info is None:
        snaps_info = get_tng_snaps_info(sim)
    return snaps_info[int(snap_number)]['num_groups_subfind']

def cosmic_times_snapshots(sim='TNG50-1',snaps_info=None):

    if snaps_info is None:
        snaps_info = get_tng_snaps_info(sim)

    nsnaps = 100
    cosmic_times = np.zeros(nsnaps)
    for ii in range(nsnaps):
        snap_z = get_snap_z(ii, sim=sim, snaps_info=snaps_info)
        cosmic_times[ii] = cosmo.age(snap_z).value

    return cosmic_times

def cosmic_times_of_snapshots(snaps, sim='TNG50-1', snaps_info=None):

    if snaps_info is None:
        snaps_info = get_tng_snaps_info(sim)

    cosmic_times = []
    for ii in snaps:
        snap_z = get_snap_z(ii, sim=sim, snaps_info=snaps_info)
        cosmic_times.append(cosmo.age(snap_z).value)

    return np.asarray(cosmic_times)

def download_cutout_subhalo_hdf5(snap_number, subhalo_id, sim='TNG50-1', params=None):
    url = "http://www.tng-project.org/api/" + sim + "/snapshots/" + str(int(snap_number)) + "/subhalos/" + str(int(subhalo_id))
    sub = get(url)
    name0 = get(sub['cutouts']['subhalo'],params)
    name = 'cutout_shalo_'+str(int(snap_number))+'_'+str(int(subhalo_id))+'.hdf5'
    os.rename(name0, name)
    return name 

def download_cutout_parent_halo_hdf5(snap_number, subhalo_id, sim='TNG50-1', params=None):
    url = "http://www.tng-project.org/api/" + sim + "/snapshots/" + str(int(snap_number)) + "/subhalos/" + str(int(subhalo_id))
    sub = get(url)
    name0 = get(sub['cutouts']['parent_halo'],params)
    name = 'cutout_phalo_'+str(int(snap_number))+'_'+str(int(subhalo_id))+'.hdf5'
    os.rename(name0, name)
    return name

def get_basic_subhalo_properties(snap_number, subhalo_id, sim='TNG50-1'):
	url = "http://www.tng-project.org/api/" + sim + "/snapshots/" + str(int(snap_number)) + "/subhalos/" + str(int(subhalo_id))
	sub = get(url)
	return sub

