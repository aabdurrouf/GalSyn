import os, sys 
import numpy as np
from astropy.io import fits
from .imgutils import *
from .utils import *
import h5py

def get_particle_id_mask(snapshot, halo_pos, halo_rad, particle_type):
    """
    Creates a boolean mask for particles of a given type that are within a specified radius of a halo's position.
    There's a faster way to do this using the Cells dataset in the snapshot, but this should work for most haloes.

    Args:
        snapshot (h5py.File): The HDF5 file containing the snapshot data.
        halo_pos (array-like): The (x, y, z) position of the halo.
        halo_rad (float): The radius within which to select particles (in the same units as the coordinates).
        particle_type (int): The particle type index to select (e.g., 0 for gas, 1 for dark matter, 4 for stars).
    Returns:
        np.ndarray: A boolean mask array where True indicates particles of the specified type within the halo radius.   
    """

    part_coords  = np.array(snapshot[f'PartType{particle_type}']['Coordinates'], dtype=np.float64)
    part_coords_minus_halo_pos  = np.sqrt( (part_coords[:,0]  - halo_pos[0])**2 + (part_coords[:,1]  - halo_pos[1])**2 + (part_coords[:,2]  - halo_pos[2])**2 )
    part_mask  = (part_coords_minus_halo_pos  <= halo_rad)

    return part_mask


def make_sim_file_from_swift_data(input_path, snapshot_name, target_halo_number=0, XH=0.76, output_hdf5='./sim_file_swift.hdf5', star_particle_type=4):
    """
    Converts a Swift-Eagle cutout into a standardized HDF5 file for analysis.
    This function extracts star and gas particle data, converts units from
    Swift conventions to physical units (e.g., kpc, Msun), calculates
    additional properties like gas temperature, and saves the result to a
    new HDF5 file.

    Args:
        input_path (str): Path to the raw Swift cutout HDF5 file to process.
        snapshot_name (str): The name of the snapshot file (e.g., 'snapshot_099.hdf5').
        target_halo_number (int): The index of the halo to extract data from.
        XH (float): The primordial mass fraction of hydrogen.
        output_hdf5 (str): The path for the new, processed HDF5 file.
        star_particle_type (int): The particle type index for stars in Swift (default is 4).
    Returns:
        output_hdf5 (str): The path to the newly created HDF5 file.
    """
        
    snapshot_number       = snapshot_name.split('_')[-1].split('.')[0]
    snap_data             = h5py.File(input_path + snapshot_name, 'r')
    reduced_snapshot_name = snapshot_name.split('.')[0]
    try:
        halo_properties = h5py.File(f'{input_path}/halos/{reduced_snapshot_name}.VELOCIraptor.properties.0', 'r')
    except:
        try:
            halo_properties = h5py.File(f'{input_path}/haloes/{reduced_snapshot_name}.VELOCIraptor.properties.0', 'r')
        except:
            print("Error occurred while reading swift halo properties.")


    halo_pos = np.array((halo_properties['Xc'][target_halo_number], halo_properties['Yc'][target_halo_number], halo_properties['Zc'][target_halo_number]))
    halo_radius = halo_properties['R_200crit'][target_halo_number]

    
    
    cosmo_h    = snap_data['Cosmology'].attrs['h']
    z          = snap_data['Cosmology'].attrs['Redshift']
    snap_a     = snap_data['Cosmology'].attrs['Scale-factor']
    solar_mass = snap_data['PhysicalConstants/CGS'].attrs['solar_mass']  
    Z_sun      = 1/0.02

    stellar_mass_phys_conversion   = snap_data[f'PartType{star_particle_type}/Masses'].attrs['Conversion factor to physical CGS (including cosmological corrections)']/solar_mass
    stellar_coords_phys_conversion = snap_data[f'PartType{star_particle_type}/Coordinates'].attrs['Conversion factor to physical CGS (including cosmological corrections)'] / (3.085678e+21)

    star_mask = get_particle_id_mask(snap_data,  halo_pos, halo_radius, particle_type=4)
    
    print(f'Extracting data for halo {target_halo_number} at position {np.round(halo_pos,4)} and radius {np.round(halo_radius*stellar_coords_phys_conversion,4)} kpc')
    print(f'Number of star particles in halo {target_halo_number}: {np.sum(star_mask)}')

    stars_mass      = snap_data[f'PartType{star_particle_type}']['Masses'][star_mask] * stellar_mass_phys_conversion
    stars_init_mass = snap_data[f'PartType{star_particle_type}']['InitialMasses'][star_mask] * stellar_mass_phys_conversion
    stars_zmet      = snap_data[f'PartType{star_particle_type}']['MetalMassFractions'][star_mask] #* Z_sun #* snap_data[f'PartType{star_particle_type}']['Masses'][star_mask] * 1e10
    stars_coords    = snap_data[f'PartType{star_particle_type}']['Coordinates'][star_mask] * stellar_coords_phys_conversion # in kpc
    stars_vel       = snap_data[f'PartType{star_particle_type}']['Velocities'][star_mask] # * np.sqrt(snap_a)  # peculiar velocities in km/s
    stars_form_a    = snap_data[f'PartType{star_particle_type}']['BirthScaleFactors'][star_mask]
    stars_form_z    = (1.0/stars_form_a) - 1.0

    # get gas particles data
    if 'PartType0' in snap_data:
        gas_mask = get_particle_id_mask(snap_data,  halo_pos, halo_radius, particle_type=0)
        gas_mass_phys_conversion   = snap_data[f'PartType0/Masses'].attrs['Conversion factor to physical CGS (including cosmological corrections)']/solar_mass
        gas_coords_phys_conversion = snap_data[f'PartType0/Coordinates'].attrs['Conversion factor to physical CGS (including cosmological corrections)'] / (3.085678e+21)
        
        gas_coords   = snap_data['PartType0']['Coordinates'][gas_mask] * gas_coords_phys_conversion # in kpc
        gas_vel      = snap_data['PartType0']['Velocities'][gas_mask] # * np.sqrt(snap_a)   # peculiar velocity in km/s
        gas_mass     = snap_data['PartType0']['Masses'][gas_mask] * gas_mass_phys_conversion
        gas_zmet     = snap_data['PartType0']['MetalMassFractions'][gas_mask] #* Z_sun #* snap_data['PartType0']['Masses'][gas_mask] * 1e10
        gas_sfr_inst = snap_data['PartType0']['StarFormationRates'][gas_mask]   # in Msun/yr
        u            = snap_data['PartType0']['InternalEnergies'][gas_mask]
        # Xe           = snap_data['PartType0']['ElectronAbundance'][gas_mask]

        # gamma = 5.0/3.0
        # KB = 1.3807e-16
        # mp = 1.6726e-24
        # mu = (4*mp)/(1 + (3*XH) + (4*XH*Xe))
        # gas_temp = (gamma-1.0)*(u/KB)*mu*1e+10
        gas_temp = snap_data['PartType0']['Temperatures'][gas_mask]
        
        # gas_mass_H = gas_mass * XH
        gas_mass_H = gas_mass * snap_data['PartType0/ElementMassFractions'][gas_mask,0]

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