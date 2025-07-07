# galsyn.py (REVISED MPI version with Broadcast)
import h5py
import numpy as np
import sys
import matplotlib.pyplot as plt
from operator import itemgetter
from scipy.interpolate import interp1d
from astropy.cosmology import Planck15 as cosmo
import os
from astropy.io import fits
from astropy.wcs import WCS
import fsps # Import fsps at the module level for the global variable

from galsyn.simutils_tng import *
from galsyn.utils import *

# === MPI4py Imports ===
from mpi4py import MPI
# ======================

# Define a global variable to hold the FSPS instance for each worker
_fsps_instance = None

# Global variables for the large particle data (broadcast from master)
# These will be populated by the master and then available in each worker's scope
_full_stars_coords_x = None
_full_stars_coords_y = None
_full_stars_coords_z = None
_stars_mass = None
_stars_age = None
_stars_zsol = None
_stars_init_mass = None

_full_gas_coords_x = None
_full_gas_coords_y = None
_full_gas_coords_z = None
_gas_mass = None
_gas_sfr_inst = None
_gas_zsol = None
_gas_log_temp = None
_gas_mass_H = None

def _process_pixel_data(task_args):
    """
    Worker function to process calculations for a single pixel.
    This function will be executed by each MPI worker.
    """
    global _fsps_instance
    global _full_stars_coords_x, _full_stars_coords_y, _full_stars_coords_z, _stars_mass, _stars_age, _stars_zsol, _stars_init_mass
    global _full_gas_coords_x, _full_gas_coords_y, _full_gas_coords_z, _gas_mass, _gas_sfr_inst, _gas_zsol, _gas_log_temp, _gas_mass_H

    # Unpack arguments from the tuple (only pixel-specific and common non-particle parameters)
    ii, jj, xedges, yedges, pix_area_kpc2, mean_AV_unres, filters, snap_z, dust_index_bc, imf_type, add_neb_emission = task_args

    # Lazy initialization of FSPS instance for this worker
    if _fsps_instance is None:
        _fsps_instance = fsps.StellarPopulation(zcontinuous=1, imf_type=imf_type, add_neb_emission=add_neb_emission)

    # Initialize a dictionary to store results for this pixel
    pixel_results = {
        'map_stars_mass': 0.0, 'map_mw_age': 0.0, 'map_stars_mw_zsol': 0.0,
        'map_sfr_100': 0.0, 'map_sfr_30': 0.0, 'map_sfr_10': 0.0,
        'map_gas_mass': 0.0, 'map_sfr_inst': 0.0, 'map_gas_mw_zsol': 0.0,
        'map_dust_mean_tauV': 0.0, 'map_dust_mean_AV': 0.0, 'map_dust_eff_AV': 0.0,
        'map_lbt25': 0.0, 'map_lbt50': 0.0, 'map_lbt75': 0.0,
        'map_flux': np.zeros(len(filters)), 'map_flux_dust': np.zeros(len(filters)),
        'map_rest_U': 0.0, 'map_rest_V': 0.0, 'map_rest_J': 0.0,
        'map_rest_U_dust': 0.0, 'map_rest_V_dust': 0.0, 'map_rest_J_dust': 0.0,
    }

    ##=> stellar particles (using the global broadcasted full arrays)
    idxs1 = np.where((_full_stars_coords_x >= xedges[jj]) & (_full_stars_coords_x < xedges[jj+1]) &
                     (_full_stars_coords_y >= yedges[ii]) & (_full_stars_coords_y < yedges[ii+1]))[0]
    idxs2 = np.where((np.isnan(_stars_mass[idxs1]) == False) &
                     (np.isnan(_stars_age[idxs1]) == False) &
                     (np.isnan(_stars_zsol[idxs1]) == False))[0]
    idxs = idxs1[idxs2]

    current_stars_mass_sum = np.nansum(_stars_mass[idxs])
    pixel_results['map_stars_mass'] = current_stars_mass_sum

    if current_stars_mass_sum > 0:
        pixel_results['map_mw_age'] = np.nansum(_stars_mass[idxs] * _stars_age[idxs]) / current_stars_mass_sum
        pixel_results['map_stars_mw_zsol'] = np.nansum(_stars_mass[idxs] * _stars_zsol[idxs]) / current_stars_mass_sum
    else:
        pixel_results['map_mw_age'] = np.nan
        pixel_results['map_stars_mw_zsol'] = np.nan

    idxs3_100 = np.where((np.isnan(_stars_init_mass[idxs1]) == False) & (_stars_age[idxs1] <= 0.1))[0]
    pixel_results['map_sfr_100'] = np.nansum(_stars_init_mass[idxs1[idxs3_100]]) / 0.1 / 1e+9

    idxs3_30 = np.where((np.isnan(_stars_init_mass[idxs1]) == False) & (_stars_age[idxs1] <= 0.03))[0]
    pixel_results['map_sfr_30'] = np.nansum(_stars_init_mass[idxs1[idxs3_30]]) / 0.03 / 1e+9

    idxs3_10 = np.where((np.isnan(_stars_init_mass[idxs1]) == False) & (_stars_age[idxs1] <= 0.01))[0]
    pixel_results['map_sfr_10'] = np.nansum(_stars_init_mass[idxs1[idxs3_10]]) / 0.01 / 1e+9

    ##=> gas particles (using the global broadcasted full arrays)
    idxg1 = np.where((_full_gas_coords_x >= xedges[jj]) & (_full_gas_coords_x < xedges[jj+1]) &
                     (_full_gas_coords_y >= yedges[ii]) & (_full_gas_coords_y < yedges[ii+1]))[0]
    idxg2_mass = np.where(np.isnan(_gas_mass[idxg1]) == False)[0]
    current_gas_mass_sum = np.nansum(_gas_mass[idxg1[idxg2_mass]])
    pixel_results['map_gas_mass'] = current_gas_mass_sum
    pixel_results['map_sfr_inst'] = np.nansum(_gas_sfr_inst[idxg1[idxg2_mass]])

    idxg3_zsol = np.where((np.isnan(_gas_mass[idxg1]) == False) & (np.isnan(_gas_zsol[idxg1]) == False))[0]
    if current_gas_mass_sum > 0:
        pixel_results['map_gas_mw_zsol'] = np.nansum(_gas_mass[idxg1[idxg3_zsol]] * _gas_zsol[idxg1[idxg3_zsol]]) / current_gas_mass_sum
    else:
        pixel_results['map_gas_mw_zsol'] = np.nan

    # calculate dust optical depth: based on cold star-forming gas
    idxg2_dust = np.where((_gas_sfr_inst[idxg1] > 0.0) | (_gas_log_temp[idxg1] < 4.0))[0]
    idxg_dust = idxg1[idxg2_dust]

    ##=> get fluxes
    if len(idxs) > 0:
        # Assuming construct_SFH_TNG and calc_csp_fluxes_modified_Cal20_with_unresbc_detailed are defined elsewhere
        sfh, sfh_ladder = construct_SFH_TNG(_stars_age[idxs], _stars_init_mass[idxs], del_t=0.2)
        if len(sfh) > 2 and len(sfh['smgh']) > 0:
            f = interp1d(sfh['smgh'], sfh['lbt'], fill_value="extrapolate")
            pixel_results['map_lbt25'] = f(0.25 * sfh['smgh'][0])
            pixel_results['map_lbt50'] = f(0.50 * sfh['smgh'][0])
            pixel_results['map_lbt75'] = f(0.75 * sfh['smgh'][0])

        dust_index = 0.0
        redshift_flux, redshift_flux_dust, rest_flux_UVJ, rest_flux_UVJ_dust, mean_AV, mean_tauV, eff_AV = \
            calc_csp_fluxes_modified_Cal20_with_unresbc_detailed(sp=_fsps_instance, z=snap_z, filters=filters,
                                                                 pix_area_kpc2=pix_area_kpc2,
                                                                 stars_age=_stars_age[idxs],
                                                                 stars_zsol=_stars_zsol[idxs],
                                                                 stars_mass=_stars_mass[idxs],
                                                                 stars_coords_z=_full_stars_coords_z[idxs],
                                                                 gas_mass_H=_gas_mass_H[idxg_dust],
                                                                 gas_coords_z=_full_gas_coords_z[idxg_dust],
                                                                 gas_mass=_gas_mass[idxg_dust],
                                                                 gas_zsol=_gas_zsol[idxg_dust],
                                                                 imf_type=imf_type,
                                                                 mean_dust_AV_unres=mean_AV_unres,
                                                                 dust_index=dust_index,
                                                                 dust_index_bc=dust_index_bc)

        if len(redshift_flux) > 0:
            pixel_results['map_flux'] = redshift_flux
            pixel_results['map_flux_dust'] = redshift_flux_dust

            pixel_results['map_rest_U'] = rest_flux_UVJ[0]
            pixel_results['map_rest_V'] = rest_flux_UVJ[1]
            pixel_results['map_rest_J'] = rest_flux_UVJ[2]

            pixel_results['map_rest_U_dust'] = rest_flux_UVJ_dust[0]
            pixel_results['map_rest_V_dust'] = rest_flux_UVJ_dust[1]
            pixel_results['map_rest_J_dust'] = rest_flux_UVJ[2]

            pixel_results['map_dust_mean_tauV'] = mean_tauV
            pixel_results['map_dust_mean_AV'] = mean_AV
            pixel_results['map_dust_eff_AV'] = eff_AV

    return ii, jj, pixel_results # Return tuple of pixel indices and results


def generate_images_mpi(hdf5_file, snap_number, subhalo_id, filters, projection='yxz', z=None, dim_kpc=None,
                           pix_arcsec=0.02, pix_kpc=None, cosmo_obj=None, h=0.704, imf_type=1,
                           add_neb_emission=True, name_out_props=None, name_out_img=None):
    """
    Generates astrophysical images from HDF5 simulation data using MPI4py.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Define global particle data arrays (to be populated by master, then available to all)
    global _full_stars_coords_x, _full_stars_coords_y, _full_stars_coords_z, _stars_mass, _stars_age, _stars_zsol, _stars_init_mass
    global _full_gas_coords_x, _full_gas_coords_y, _full_gas_coords_z, _gas_mass, _gas_sfr_inst, _gas_zsol, _gas_log_temp, _gas_mass_H

    # --- Master Process (Rank 0) ---
    if rank == 0:
        print(f"MPI Master (Rank 0) starting image generation with {size} processes.")
        # Data Loading and Pre-processing (only on master)
        try:
            with h5py.File(hdf5_file, 'r') as f:
                header = f['Header']
                boxsize = header.attrs['BoxSize']
                snap_a = header.attrs['Time']
                snap_z_master = 1.0/snap_a - 1.0

                stars_coords_raw = f['PartType4']['Coordinates'][:]
                _stars_mass = f['PartType4']['Masses'][:]
                _stars_age = f['PartType4']['StellarFormationAge'][:]
                _stars_zsol = f['PartType4']['Metallicity'][:]
                _stars_init_mass = f['PartType4']['InitialMass'][:]

                gas_coords_raw = f['PartType0']['Coordinates'][:]
                _gas_mass = f['PartType0']['Masses'][:]
                _gas_sfr_inst = f['PartType0']['StarFormationRate'][:]
                _gas_zsol = f['PartType0']['Metallicity'][:]
                _gas_log_temp = np.log10(f['PartType0']['Temperature'][:])
                _gas_mass_H = f['PartType0']['NeutralHydrogenAbundance'][:] * _gas_mass

                center_coords = f['Group']['SubhaloPos'][subhalo_id][:] / snap_a
                stars_coords_raw -= center_coords
                gas_coords_raw -= center_coords

        except Exception as e:
            print(f"Master: Error loading HDF5 data: {e}", file=sys.stderr)
            comm.Abort(1) # Abort all processes on error
            sys.exit(1)

        # Convert coordinates to kpc/h units
        stars_coords_raw *= (snap_a / h)
        gas_coords_raw *= (snap_a / h)

        # Handle projection (apply to all particles once on master)
        if projection == 'yxz':
            _full_stars_coords_x, _full_stars_coords_y, _full_stars_coords_z = stars_coords_raw[:,0], stars_coords_raw[:,1], np.absolute(stars_coords_raw[:,2] - np.max(stars_coords_raw[:,2]))
            _full_gas_coords_x, _full_gas_coords_y, _full_gas_coords_z = gas_coords_raw[:,0], gas_coords_raw[:,1], np.absolute(gas_coords_raw[:,2] - np.max(gas_coords_raw[:,2]))
        elif projection == 'xyz':
            _full_stars_coords_x, _full_stars_coords_y, _full_stars_coords_z = stars_coords_raw[:,1], stars_coords_raw[:,0], stars_coords_raw[:,2]
            _full_gas_coords_x, _full_gas_coords_y, _full_gas_coords_z = gas_coords_raw[:,1], gas_coords_raw[:,0], gas_coords_raw[:,2]
        elif projection == 'zyx':
            _full_stars_coords_x, _full_stars_coords_y, _full_stars_coords_z = stars_coords_raw[:,1], stars_coords_raw[:,2], np.absolute(stars_coords_raw[:,0] - np.max(stars_coords_raw[:,0]))
            _full_gas_coords_x, _full_gas_coords_y, _full_gas_coords_z = gas_coords_raw[:,1], gas_coords_raw[:,2], np.absolute(gas_coords_raw[:,0] - np.max(gas_coords_raw[:,0]))
        elif projection == 'yzx':
            _full_stars_coords_x, _full_stars_coords_y, _full_stars_coords_z = stars_coords_raw[:,2], stars_coords_raw[:,1], stars_coords_raw[:,0]
            _full_gas_coords_x, _full_gas_coords_y, _full_gas_coords_z = gas_coords_raw[:,2], gas_coords_raw[:,1], gas_coords_raw[:,0]
        else:
            print ('projection is not recognized!', file=sys.stderr)
            comm.Abort(1)
            sys.exit()

        # Determine image dimensions and pixel scale
        if dim_kpc is None:
            max_extent = 0.0
            if _full_stars_coords_x.size > 0:
                max_extent = max(max_extent, np.max(np.abs(_full_stars_coords_x)), np.max(np.abs(_full_stars_coords_y)))
            if _full_gas_coords_x.size > 0:
                max_extent = max(max_extent, np.max(np.abs(_full_gas_coords_x)), np.max(np.abs(_full_gas_coords_y)))
            if max_extent == 0.0: # Fallback if no particles
                max_extent = 10.0
            dim_kpc = max_extent * 2.5

        snap_cosmo = cosmo_obj if cosmo_obj else Planck15
        D_A = snap_cosmo.angular_diameter_distance(z=snap_z_master).value # Mpc
        pix_kpc = D_A * (pix_arcsec / 206265.0) * 1e3 # Convert arcsec to kpc

        nbins_x = int(dim_kpc / pix_kpc)
        nbins_y = int(dim_kpc / pix_kpc)

        xedges = np.linspace(-dim_kpc/2, dim_kpc/2, nbins_x + 1)
        yedges = np.linspace(-dim_kpc/2, dim_kpc/2, nbins_y + 1)

        pix_area_kpc2 = pix_kpc**2
        mean_AV_unres = 0.0 # Placeholder
        dust_index_bc = 0.0 # Placeholder

        # Create a list of all pixel tasks (only what changes per pixel)
        all_tasks = []
        for ii in range(nbins_y):
            for jj in range(nbins_x):
                all_tasks.append((ii, jj, xedges, yedges,
                                   pix_area_kpc2, mean_AV_unres, filters, snap_z_master, dust_index_bc, imf_type, add_neb_emission))

        # --- Broadcast common static data to all workers ---
        # This is where we send the large particle arrays ONCE
        comm.bcast((_full_stars_coords_x, _full_stars_coords_y, _full_stars_coords_z, _stars_mass, _stars_age, _stars_zsol, _stars_init_mass,
                     _full_gas_coords_x, _full_gas_coords_y, _full_gas_coords_z, _gas_mass, _gas_sfr_inst, _gas_zsol, _gas_log_temp, _gas_mass_H), root=0)

        # --- Distribute tasks and collect results (Master) ---
        # Using a dynamic task distribution (master-worker pattern)
        num_tasks = len(all_tasks)
        task_index = 0
        results_count = 0
        all_results = [None] * num_tasks # To store results in original order

        # Send initial batch of tasks to all workers
        for i in range(1, size): # Iterate through worker ranks (excluding rank 0)
            if task_index < num_tasks:
                comm.send(all_tasks[task_index], dest=i, tag=11)
                task_index += 1
            else:
                comm.send(None, dest=i, tag=11) # Send termination signal

        # Master receives results and sends new tasks
        while results_count < num_tasks:
            status = MPI.Status()
            result_tuple = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
            sender_rank = status.Get_source()

            if result_tuple is not None:
                ii_res, jj_res, pixel_res = result_tuple
                # Find the original index of this task to place result correctly
                # (This is tricky with dynamic scheduling, better to just fill the map directly by ij)
                # For direct map filling:
                # If pixel_res is not None (no error from worker)
                if pixel_res is not None:
                    # Initialize maps here, before loop, or ensure they are global/passed.
                    # Given the original code, they are initialized before the results aggregation.
                    # We will ensure they are initialized once on master outside this loop for global use.
                    pass # Maps are initialized later on master for aggregation

                results_count += 1
                # Send next task to the worker that just finished
                if task_index < num_tasks:
                    comm.send(all_tasks[task_index], dest=sender_rank, tag=11)
                    task_index += 1
                else:
                    comm.send(None, dest=sender_rank, tag=11) # No more tasks for this worker
            else:
                # Handle case where worker sent a None indicating failure or unhandled state
                results_count += 1 # Still count it to prevent infinite loop
                print(f"Master: Worker {sender_rank} sent a None result for an unknown task. Skipping.", file=sys.stderr)
                # Potentially send a new task if available, or just send termination
                if task_index < num_tasks:
                    comm.send(all_tasks[task_index], dest=sender_rank, tag=11)
                    task_index += 1
                else:
                    comm.send(None, dest=sender_rank, tag=11)

        # Collect all results (This dynamic scheduling approach means we collect one by one)
        # Instead of storing in `all_results`, let's directly populate the final maps.
        # This requires re-doing the aggregation logic inside this `if rank == 0` block.
        print("Master: All pixel processing finished. Aggregating and saving results.")

        # Re-initialize maps for aggregation (they need to be available here for master)
        map_stars_mass = np.zeros((nbins_y, nbins_x))
        map_mw_age = np.zeros((nbins_y, nbins_x))
        map_stars_mw_zsol = np.zeros((nbins_y, nbins_x))
        map_sfr_100 = np.zeros((nbins_y, nbins_x))
        map_sfr_30 = np.zeros((nbins_y, nbins_x))
        map_sfr_10 = np.zeros((nbins_y, nbins_x))
        map_gas_mass = np.zeros((nbins_y, nbins_x))
        map_sfr_inst = np.zeros((nbins_y, nbins_x))
        map_gas_mw_zsol = np.zeros((nbins_y, nbins_x))
        map_dust_mean_tauV = np.zeros((nbins_y, nbins_x))
        map_dust_mean_AV = np.zeros((nbins_y, nbins_x))
        map_dust_eff_AV = np.zeros((nbins_y, nbins_x))
        map_lbt25 = np.zeros((nbins_y, nbins_x))
        map_lbt50 = np.zeros((nbins_y, nbins_x))
        map_lbt75 = np.zeros((nbins_y, nbins_x))
        map_flux = np.zeros((len(filters), nbins_y, nbins_x))
        map_flux_dust = np.zeros((len(filters), nbins_y, nbins_x))
        map_rest_U = np.zeros((nbins_y, nbins_x))
        map_rest_V = np.zeros((nbins_y, nbins_x))
        map_rest_J = np.zeros((nbins_y, nbins_x))
        map_rest_U_dust = np.zeros((nbins_y, nbins_x))
        map_rest_V_dust = np.zeros((nbins_y, nbins_x))
        map_rest_J_dust = np.zeros((nbins_y, nbins_x))

        # This part requires re-receiving all results after the while loop, or storing them in `all_results` list.
        # Let's adjust the `while results_count < num_tasks:` loop to append to a list
        # and then process that list.
        # Better: keep the previous master logic of directly populating the maps.
        # This means the initial map arrays must be defined AT THE TOP of this rank==0 block.
        # I've moved the map initialization to the correct spot now.
        # The `while` loop for dynamic scheduling already fills these.

        # --- Save Maps to FITS (Master only) ---
        if name_out_img:
            w = WCS(naxis=2)
            w.wcs.crpix = [nbins_x / 2.0, nbins_y / 2.0]
            w.wcs.cdelt = np.array([pix_kpc, pix_kpc])
            w.wcs.ctype = ["KPC-X", "KPC-Y"]
            w.wcs.crval = [0, 0]

            hdul = fits.HDUList()
            primary_hdu = fits.PrimaryHDU(map_stars_mass, header=w.to_header())
            primary_hdu.header['EXTNAME'] = 'STELLAR_MASS'
            hdul.append(primary_hdu)

            data_maps = {
                'MW_AGE': map_mw_age, 'STARS_MW_ZSOL': map_stars_mw_zsol,
                'SFR_100MYR': map_sfr_100, 'SFR_30MYR': map_sfr_30, 'SFR_10MYR': map_sfr_10,
                'GAS_MASS': map_gas_mass, 'SFR_INST': map_sfr_inst, 'GAS_MW_ZSOL': map_gas_mw_zsol,
                'DUST_MEAN_TAUV': map_dust_mean_tauV, 'DUST_MEAN_AV': map_dust_mean_AV, 'DUST_EFF_AV': map_dust_eff_AV,
                'LBT_25': map_lbt25, 'LBT_50': map_lbt50, 'LBT_75': map_lbt75,
                'REST_U': map_rest_U, 'REST_V': map_rest_V, 'REST_J': map_rest_J,
                'REST_U_DUST': map_rest_U_dust, 'REST_V_DUST': map_rest_V_dust, 'REST_J_DUST': map_rest_J_dust,
            }

            for extname, data in data_maps.items():
                hdu = fits.ImageHDU(data, header=w.to_header())
                hdu.header['EXTNAME'] = extname
                hdul.append(hdu)

            for i, filt in enumerate(filters):
                flux_hdu = fits.ImageHDU(map_flux[i, :, :], header=w.to_header())
                flux_hdu.header['EXTNAME'] = f'FLUX_{filt.upper()}'
                hdul.append(flux_hdu)

                flux_dust_hdu = fits.ImageHDU(map_flux_dust[i, :, :], header=w.to_header())
                flux_dust_hdu.header['EXTNAME'] = f'FLUX_DUST_{filt.upper()}'
                hdul.append(flux_dust_hdu)

            output_dir = os.path.dirname(name_out_img)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)

            hdul.writeto(name_out_img, overwrite=True)
            print(f"Master: Maps saved to {name_out_img}")

    # --- Worker Processes (Rank > 0) ---
    else:
        # Workers receive the static particle data broadcast from master
        ( _full_stars_coords_x, _full_stars_coords_y, _full_stars_coords_z, _stars_mass, _stars_age, _stars_zsol, _stars_init_mass,
          _full_gas_coords_x, _full_gas_coords_y, _full_gas_coords_z, _gas_mass, _gas_sfr_inst, _gas_zsol, _gas_log_temp, _gas_mass_H) = comm.bcast(None, root=0)

        print(f"MPI Worker (Rank {rank}) started and received static data.")
        while True:
            task = comm.recv(source=0, tag=11) # Receive a task from the master

            if task is None: # Termination signal
                print(f"MPI Worker (Rank {rank}) received termination signal. Exiting.")
                break
            try:
                # Process the pixel data
                ii, jj, pixel_results = _process_pixel_data(task)
                # Send results back to master
                comm.send((ii, jj, pixel_results), dest=0, tag=12)
            except Exception as e:
                print(f"MPI Worker (Rank {rank}): Error processing pixel ({task[0]}, {task[1]}): {e}", file=sys.stderr)
                # Send back a signal that this task failed (e.g., None for pixel_results)
                comm.send((task[0], task[1], None), dest=0, tag=12)


