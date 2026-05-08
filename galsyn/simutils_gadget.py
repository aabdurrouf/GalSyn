import os, sys 
import numpy as np
from astropy.io import fits 
from .imgutils import *
from .utils import *
import h5py

def make_sim_file_from_gadget_data(input_path, snapshot_name, target_halo_number=0, XH=0.76, output_hdf5='./sim_file_gadget.hdf5', dark_matter_particle_type=1, star_particle_type=4):
    """
    Converts a Gadget-4 cutout into a standardized HDF5 file for analysis.
    This function extracts star and gas particle data, converts units from
    Gadget conventions to physical units (e.g., kpc, Msun), calculates
    additional properties like gas temperature, and saves the result to a
    new HDF5 file.

    This function assumes that the user ran the FOF and Subfind options 
    when running the Gadget-4 Simulation, which arranges the particle data
    in the snapshot files according to their halo membership.

    Args:
        input_path (str): Path to the raw Gadget cutout HDF5 file to process.
        snapshot_name (str): The name of the snapshot file (e.g., 'snapshot_099.hdf5').
        target_halo_number (int): The index of the halo to extract data from.
        XH (float): The primordial mass fraction of hydrogen.
        output_hdf5 (str): The path for the new, processed HDF5 file.
        dark_matter_particle_type (int): The particle type index for dark matter in Gadget (default is 1).
        star_particle_type (int): The particle type index for stars in Gadget (default is 4).
    Returns:
        output_hdf5 (str): The path to the newly created HDF5 file.
    """
        
    snapshot_number = snapshot_name.split('_')[-1].split('.')[0]
    haloinfo_fname  = f'/fof_subhalo_tab_{snapshot_number}.hdf5'

    try:
        snap_data     = h5py.File(input_path + snapshot_name, 'r')
    except:
        print("Error occurred while reading gadget snapshot data.")
    
    try:
        haloinfo_data = h5py.File(input_path + haloinfo_fname, 'r')
    except:
        print("Error occurred while reading gadget halo info data.")

    group_offset_type  = np.array(haloinfo_data['Group']['GroupOffsetType'], dtype=np.int64)
    group_len_type     = np.array(haloinfo_data['Group']['GroupLenType'], dtype=np.int64)
 
    target_offset      = group_offset_type[target_halo_number]
    target_len         = group_len_type[target_halo_number]

    gas_start_index    = target_offset[0]
    gas_end_index      = target_offset[0] + target_len[0]
 
    star_start_index   = target_offset[star_particle_type]
    star_end_index     = target_offset[star_particle_type] + target_len[star_particle_type]

    halo_pos    = haloinfo_data['Group']['GroupPos'][target_halo_number]
    halo_radius = haloinfo_data['Group']['Group_R_Crit200'][target_halo_number]
    

    cosmo_h    = snap_data['Parameters'].attrs['HubbleParam']
    z          = snap_data['Header'].attrs['Redshift']
    snap_a     = snap_data['Header'].attrs['Time']
    kpc_factor = snap_data['Parameters'].attrs['UnitLength_in_cm'] / (3.085678e+21)  # Allows for run time conversion to kpc units

    print(f'Extracting data for halo {target_halo_number} at position {np.round(halo_pos,4)} and radius {np.round(halo_radius*kpc_factor,4)} kpc')
    print(f'Number of star particles in halo {target_halo_number}: {np.sum(target_len[star_particle_type])}')

    stars_mass      = snap_data[f'PartType{star_particle_type}']['Masses'][star_start_index:star_end_index] * 1e+10 / cosmo_h
    stars_init_mass = snap_data[f'PartType{star_particle_type}']['Masses'][star_start_index:star_end_index] * 1e+10 / cosmo_h # Currently Gadget-4 has no mass loss prescription, so 'initial mass' is the same 'mass'
    stars_zmet      = snap_data[f'PartType{star_particle_type}']['Metallicity'][star_start_index:star_end_index] 
    stars_coords    = snap_data[f'PartType{star_particle_type}']['Coordinates'][star_start_index:star_end_index] * snap_a / cosmo_h * kpc_factor # in kpc
    stars_vel       = snap_data[f'PartType{star_particle_type}']['Velocities'][star_start_index:star_end_index] * np.sqrt(snap_a)  # peculiar velocities in km/s
    stars_form_a    = snap_data[f'PartType{star_particle_type}']['StellarFormationTime'][star_start_index:star_end_index]
    stars_form_z    = (1.0/stars_form_a) - 1.0

    # get gas particles data
    if 'PartType0' in snap_data:
        gas_coords   = snap_data['PartType0']['Coordinates'][gas_start_index:gas_end_index] * snap_a / cosmo_h * kpc_factor # in kpc
        gas_vel      = snap_data['PartType0']['Velocities'][gas_start_index:gas_end_index] * np.sqrt(snap_a)   # peculiar velocity in km/s
        gas_mass     = snap_data['PartType0']['Masses'][gas_start_index:gas_end_index] * 1e+10 / cosmo_h
        gas_zmet     = snap_data['PartType0']['Metallicity'][gas_start_index:gas_end_index]
        gas_sfr_inst = snap_data['PartType0']['StarFormationRate'][gas_start_index:gas_end_index]   # in Msun/yr
        u            = snap_data['PartType0']['InternalEnergy'][gas_start_index:gas_end_index]
        Xe           = snap_data['PartType0']['ElectronAbundance'][gas_start_index:gas_end_index]
        gamma = 5.0/3.0
        KB = 1.3807e-16
        mp = 1.6726e-24
        mu = (4*mp)/(1 + (3*XH) + (4*XH*Xe))
        gas_temp = (gamma-1.0)*(u/KB)*mu*1e+10
        
        gas_mass_H = gas_mass * XH

    else:
        gas_mass     = [0]
        gas_zmet     = [0]
        gas_sfr_inst = [0]
        gas_temp     = [0]
        gas_coords   = np.zeros((1,3))
        gas_vel      = np.zeros((1,3))
        gas_mass_H   = [0]
    
    snap_data.close()

    create_hdf5_file(output_hdf5, stars_init_mass, stars_form_z, stars_mass, stars_zmet, stars_coords,
                    stars_vel, gas_mass, gas_zmet, gas_sfr_inst, gas_temp, gas_coords, gas_vel, gas_mass_H)
    
    return str(output_hdf5)