import h5py
import os, sys
import numpy as np
from astropy.io import fits
from .imgutils import *
from .utils import *
from .dust import *
from joblib import Parallel, delayed
from tqdm.auto import tqdm
from tqdm_joblib import tqdm_joblib
import multiprocessing
from scipy.interpolate import interp1d, RegularGridInterpolator
from scipy.integrate import simpson
import importlib.resources

import fsps

FSPS_Z_SUN = 0.019

# Global variables for SSP data (loaded once per worker if use_precomputed_ssp is True)
ssp_wave = None
ssp_ages_gyr = None
ssp_logzsol_grid = None
ssp_stellar_mass_grid = None
ssp_code_z_sun = None
ssp_stellar_continuum_grid = None # New: stores stellar continuum component
ssp_nebular_emission_grid = None    # New: stores nebular emission component

_global_ssp_stellar_mass_interpolator = None

# New interpolators for SSP components
_global_ssp_stellar_continuum_interpolator = None
_global_ssp_nebular_emission_interpolator = None

# Global variables for FSPS instance (initialized once per worker if use_precomputed_ssp is False)
sp_instance = None

# Other global worker variables (particle data, etc. - these will be set by init_worker)
igm_trans = None
snap_z = None
pix_area_kpc2 = None
gas_logu = None
igm_type = None
dust_index_bc = None
dust_index = None
t_esc = None
dust_eta = None
dust_law = None
bump_amp = None
salim_a0 = None
salim_a1 = None
salim_a2 = None
salim_a3 = None
salim_RV = None
salim_B = None
dust_Alambda_per_AV = None
func_interp_dust_index = None
use_precomputed_ssp = False
ssp_interpolation_method = 'nearest'

# Global variables for dustindexAV_AV and dustindexAV_dust_index
dustindexAV_AV = None
dustindexAV_dust_index = None

_worker_scale_dust_tau = None

output_pixel_spectra_flag = False
# Renamed from global_output_obs_wave to _worker_output_obs_wave
# as it's defined and used within each worker for interpolation.
# The actual common output grid is 'fixed_global_output_obs_wave'
# determined in the main process.
_worker_output_obs_wave_grid = None 

_worker_filters = None
_worker_filter_transmission = None
_worker_filter_wave_eff = None
_worker_imf_type = None
_worker_cosmo = None
_worker_imf_upper_limit = None
_worker_imf_lower_limit = None
_worker_imf1 = None
_worker_imf2 = None
_worker_imf3 = None
_worker_vdmc = None
_worker_mdave = None

# New global worker variables for particle data - populated by init_worker
_worker_stars_mass = None
_worker_stars_age = None
_worker_stars_zmet = None
_worker_stars_init_mass = None
_worker_stars_vel_los_proj = None
_worker_stars_coords = None # New: for 3D distance calculation

_worker_gas_mass = None
_worker_gas_sfr_inst = None
_worker_gas_zmet = None
_worker_gas_log_temp = None
_worker_gas_mass_H = None
_worker_gas_vel_los_proj = None
_worker_gas_coords = None # New: for 3D distance calculation


# Global variables for light-weighted calculation wavelength range
_lw_wave_min_rest = 1000.0 # Angstrom
_lw_wave_max_rest = 30000.0 # Angstrom


def _load_filter_transmission_from_paths(filters_list, filter_transmission_path_dict):
    """
    Loads filter transmission data from specified text files and calculates
    pivot wavelengths.
    """
    filter_transmission_data = {}
    filter_wave_pivot_data = {}

    for f_name in filters_list:
        if f_name not in filter_transmission_path_dict:
            raise ValueError(f"Path for filter '{f_name}' not provided in filter_transmission_path.")
        
        file_path = filter_transmission_path_dict[f_name]
        try:
            # Load data from the text file (assuming 2 columns: wavelength, transmission)
            data = np.loadtxt(file_path)
            if data.shape[1] != 2:
                raise ValueError(f"Filter file {file_path} must have exactly two columns (wavelength, transmission).")
            
            wave = data[:, 0]
            trans = data[:, 1]

            filter_transmission_data[f_name] = {'wave': wave, 'trans': trans}

            # Calculate pivot wavelength
            # Lambda_p = sqrt(integral(lambda * T(lambda) d_lambda) / integral(T(lambda) / lambda d_lambda))
            # Numerator: integral(lambda * T(lambda) d_lambda)
            numerator = simpson(wave * trans, wave)
            # Denominator: integral(T(lambda) / lambda d_lambda)
            # Avoid division by zero if trans is zero or wave is zero at some points
            integrand_denominator = np.where(wave != 0, trans / wave, 0)
            denominator = simpson(integrand_denominator, wave)

            if denominator > 0:
                pivot_wavelength = np.sqrt(numerator / denominator)
            else:
                pivot_wavelength = np.nan # Or handle as an error/default value

            filter_wave_pivot_data[f_name] = pivot_wavelength # Storing pivot wavelength

        except Exception as e:
            print(f"Error loading or processing filter file {file_path} for filter {f_name}: {e}")
            raise

    return filter_transmission_data, filter_wave_pivot_data


def init_worker(ssp_code_val, snap_z_val, pix_area_kpc2_val,  
                filters_list_val, filter_transmission_path_val,
                imf_type_val, imf_upper_limit_val, imf_lower_limit_val, 
                imf1_val, imf2_val, imf3_val, vdmc_val, mdave_val,     
                gas_logu_val, igm_type_val, dust_index_bc_val, 
                dust_index_val, t_esc_val, dust_eta_val, precomputed_scale_dust_tau_val,
                cosmo_str_val, dust_law_val, bump_amp_val, relation_AVslope_val, salim_a0_val, 
                salim_a1_val, salim_a2_val, salim_a3_val, salim_RV_val, salim_B_val, use_precomputed_ssp_val, 
                stars_mass_arr, stars_age_arr, stars_zmet_arr, stars_init_mass_arr, stars_vel_los_proj_arr, stars_coords_arr,
                gas_mass_arr, gas_sfr_inst_arr, gas_zmet_arr, gas_log_temp_arr, gas_mass_H_arr, gas_vel_los_proj_arr, gas_coords_arr, 
                ssp_filepath_val=None, ssp_interpolation_method_val='nearest', 
                output_pixel_spectra_val=False, output_obs_wave_grid_val=None): 
    
    global ssp_wave, ssp_ages_gyr, ssp_logzsol_grid, ssp_stellar_mass_grid, ssp_code_z_sun
    global ssp_stellar_continuum_grid, ssp_nebular_emission_grid # New SSP component grids
    global _global_ssp_stellar_continuum_interpolator, _global_ssp_nebular_emission_interpolator, _global_ssp_stellar_mass_interpolator
    global sp_instance, igm_trans, snap_z, pix_area_kpc2
    global gas_logu, igm_type
    global dust_index_bc, dust_index, t_esc, dust_eta, dust_law, bump_amp, salim_a0, salim_a1, salim_a2, salim_a3 
    global salim_RV, salim_B, dust_Alambda_per_AV, func_interp_dust_index
    global use_precomputed_ssp, ssp_interpolation_method 
    global output_pixel_spectra_flag, _worker_output_obs_wave_grid 
    global _worker_filters, _worker_filter_transmission, _worker_filter_wave_eff, _worker_imf_type, _worker_cosmo
    global _worker_imf_upper_limit, _worker_imf_lower_limit, _worker_imf1, _worker_imf2, _worker_imf3, _worker_vdmc, _worker_mdave
    global dustindexAV_AV, dustindexAV_dust_index 
    global _worker_scale_dust_tau

    # Assign particle data to worker-global variables
    global _worker_stars_mass, _worker_stars_age, _worker_stars_zmet, _worker_stars_init_mass, _worker_stars_vel_los_proj, _worker_stars_coords
    global _worker_gas_mass, _worker_gas_sfr_inst, _worker_gas_zmet, _worker_gas_log_temp, _worker_gas_mass_H, _worker_gas_vel_los_proj, _worker_gas_coords

    _worker_stars_mass = stars_mass_arr
    _worker_stars_age = stars_age_arr
    _worker_stars_zmet = stars_zmet_arr
    _worker_stars_init_mass = stars_init_mass_arr
    _worker_stars_vel_los_proj = stars_vel_los_proj_arr
    _worker_stars_coords = stars_coords_arr

    _worker_gas_mass = gas_mass_arr
    _worker_gas_sfr_inst = gas_sfr_inst_arr
    _worker_gas_zmet = gas_zmet_arr
    _worker_gas_log_temp = gas_log_temp_arr
    _worker_gas_mass_H = gas_mass_H_arr
    _worker_gas_vel_los_proj = gas_vel_los_proj_arr
    _worker_gas_coords = gas_coords_arr

    snap_z = snap_z_val
    pix_area_kpc2 = pix_area_kpc2_val
    
    _worker_imf_type = imf_type_val
    _worker_imf_upper_limit = imf_upper_limit_val
    _worker_imf_lower_limit = imf_lower_limit_val
    _worker_imf1 = imf1_val 
    _worker_imf2 = imf2_val 
    _worker_imf3 = imf3_val 
    _worker_vdmc = vdmc_val 
    _worker_mdave = mdave_val

    gas_logu = gas_logu_val
    igm_type = igm_type_val
    dust_index_bc = dust_index_bc_val
    t_esc = t_esc_val
    dust_eta = dust_eta_val
    _worker_scale_dust_tau = precomputed_scale_dust_tau_val
    
    _worker_cosmo = define_cosmo(cosmo_str_val)
    
    dust_law = dust_law_val
    salim_a0 = salim_a0_val
    salim_a1 = salim_a1_val
    salim_a2 = salim_a2_val
    salim_a3 = salim_a3_val
    salim_RV = salim_RV_val
    salim_B = salim_B_val
    dust_index = dust_index_val
    bump_amp = bump_amp_val
    use_precomputed_ssp = use_precomputed_ssp_val 
    ssp_interpolation_method = ssp_interpolation_method_val 

    _worker_filters = filters_list_val
    _worker_filter_transmission, _worker_filter_wave_eff = _load_filter_transmission_from_paths(
        _worker_filters, filter_transmission_path_val
    )

    output_pixel_spectra_flag = output_pixel_spectra_val
    _worker_output_obs_wave_grid = output_obs_wave_grid_val # Set the global worker wave grid

    if use_precomputed_ssp:
        if ssp_filepath_val is None:
            print("Error: ssp_filepath must be provided when use_precomputed_ssp is True.")
            sys.exit(1)
        try:
            with h5py.File(ssp_filepath_val, 'r') as f_ssp:
                ssp_wave = f_ssp['wavelength'][:]
                ssp_ages_gyr = f_ssp['ages_gyr'][:]
                ssp_logzsol_grid = f_ssp['logzsol'][:]
                ssp_stellar_continuum_grid = f_ssp['stellar_continuum_spectra'][:] # New: Load stellar continuum
                ssp_nebular_emission_grid = f_ssp['nebular_emission_spectra'][:]   # New: Load nebular emission
                ssp_stellar_mass_grid = f_ssp['stellar_mass'][:]
                ssp_code_z_sun = f_ssp.attrs['z_sun'] 
                
                if ssp_interpolation_method == 'linear': 
                    _global_ssp_stellar_continuum_interpolator = RegularGridInterpolator( # New interpolator
                        (ssp_ages_gyr, ssp_logzsol_grid), ssp_stellar_continuum_grid, 
                        method='linear', bounds_error=False, fill_value=0.0
                    )
                    _global_ssp_nebular_emission_interpolator = RegularGridInterpolator( # New interpolator
                        (ssp_ages_gyr, ssp_logzsol_grid), ssp_nebular_emission_grid, 
                        method='linear', bounds_error=False, fill_value=0.0
                    )
                    _global_ssp_stellar_mass_interpolator = RegularGridInterpolator(
                        (ssp_ages_gyr, ssp_logzsol_grid), ssp_stellar_mass_grid, 
                        method='linear', bounds_error=False, fill_value=0.0
                    )
                elif ssp_interpolation_method == 'cubic':
                    _global_ssp_stellar_continuum_interpolator = RegularGridInterpolator( # New interpolator
                        (ssp_ages_gyr, ssp_logzsol_grid), ssp_stellar_continuum_grid, 
                        method='cubic', bounds_error=False, fill_value=0.0
                    )
                    _global_ssp_nebular_emission_interpolator = RegularGridInterpolator( # New interpolator
                        (ssp_ages_gyr, ssp_logzsol_grid), ssp_nebular_emission_grid, 
                        method='cubic', bounds_error=False, fill_value=0.0
                    )
                    _global_ssp_stellar_mass_interpolator = RegularGridInterpolator(
                        (ssp_ages_gyr, ssp_logzsol_grid), ssp_stellar_mass_grid, 
                        method='cubic', bounds_error=False, fill_value=0.0
                    )

        except Exception as e:
            print(f"Error loading SSP grid from {ssp_filepath_val}: {e}")
            sys.exit(1)
    else: # Generate on-the-fly using FSPS
        sp_instance = fsps.StellarPopulation(zcontinuous=1)
        sp_instance.params['imf_type'] = _worker_imf_type
        sp_instance.params['imf_upper_limit'] = _worker_imf_upper_limit
        sp_instance.params['imf_lower_limit'] = _worker_imf_lower_limit
        sp_instance.params['imf1'] = imf1_val 
        sp_instance.params['imf2'] = imf2_val 
        sp_instance.params['imf3'] = imf3_val 
        sp_instance.params['vdmc'] = vdmc_val 
        sp_instance.params['mdave'] = mdave_val 
        sp_instance.params["add_dust_emission"] = False
        sp_instance.params["fagn"] = 0
        sp_instance.params["sfh"] = 0
        sp_instance.params["dust1"] = 0.0
        sp_instance.params["dust2"] = 0.0
        # Get wavelength grid from a dummy call
        sp_instance.params["add_neb_emission"] = True # Can be either, just need wave for grid
        ssp_wave, _ = sp_instance.get_spectrum(peraa=True, tage=1.0)
        ssp_code_z_sun = FSPS_Z_SUN 

    # Handle relation_AVslope_val
    if isinstance(relation_AVslope_val, str):
        data_file_name = f"{relation_AVslope_val}_AV_dust_index.txt"
        try:
            data_path = str(importlib.resources.files('galsyn.data').joinpath(data_file_name))
            data = np.loadtxt(data_path)
            dustindexAV_AV = data[:, 0]
            dustindexAV_dust_index = data[:, 1]
        except Exception as e:
            print(f"Error loading dust relation data from {data_file_name}: {e}")
            sys.exit(1) 
    elif isinstance(relation_AVslope_val, dict):
        dustindexAV_AV = np.asarray(relation_AVslope_val["AV"])
        dustindexAV_dust_index = np.asarray(relation_AVslope_val["dust_index"])
    else:
        print("Error: Invalid relation_AVslope_val type passed to init_worker.")
        dustindexAV_AV = np.array([])
        dustindexAV_dust_index = np.array([])
        sys.exit(1)

    if dust_law <= 1:
        global func_interp_dust_index
        func_interp_dust_index = interp1d(dustindexAV_AV, dustindexAV_dust_index, bounds_error=False, fill_value='extrapolate')

    elif dust_law == 2:
        bump_amp = bump_amp_from_dust_index(dust_index)
        dust_Alambda_per_AV = modified_calzetti_dust_Alambda_per_AV(ssp_wave, dust_index=dust_index, bump_amp=bump_amp)

    elif dust_law == 3:
        bump_amp = bump_amp
        dust_Alambda_per_AV = modified_calzetti_dust_Alambda_per_AV(ssp_wave, dust_index=dust_index, bump_amp=bump_amp)

    elif dust_law == 4:
        dust_Alambda_per_AV = salim18_dust_Alambda_per_AV(ssp_wave, salim_a0, salim_a1, salim_a2, salim_a3, salim_B, salim_RV)

    elif dust_law == 5:
        dust_Alambda_per_AV = calzetti_dust_Alambda_per_AV(ssp_wave)

    elif dust_law == 6:
        dust_Alambda_per_AV = smc_gordon2003_dust_Alambda_per_AV(ssp_wave)
    
    elif dust_law == 7:
        dust_Alambda_per_AV = lmc_gordon2003_dust_Alambda_per_AV(ssp_wave)

    elif dust_law == 8:
        dust_Alambda_per_AV = ccm89_dust_Alambda_per_AV(ssp_wave)

    elif dust_law == 9:
        dust_Alambda_per_AV = fitzpatrick99_dust_Alambda_per_AV(ssp_wave)

    # Always add IGM absorption
    if igm_type == 0:
        igm_trans = igm_att_madau(ssp_wave * (1.0+snap_z), snap_z)
    elif igm_type == 1:
        igm_trans = igm_att_inoue(ssp_wave * (1.0+snap_z), snap_z)
    else:
        print ('igm_type is not recognized! options are: 1 for Madau+1995 and Inoue+2014')
        sys.exit()

def dust_reddening_diffuse_ism(dust_AV, wave, dust_law):
    if dust_law == 0:
        dust_index1 = func_interp_dust_index(dust_AV)
        bump_amp1 = bump_amp_from_dust_index(dust_index1)
        Alambda = modified_calzetti_dust_Alambda_per_AV(wave, dust_index=dust_index1, bump_amp=bump_amp1) * dust_AV

    elif dust_law == 1:
        dust_index1 = func_interp_dust_index(dust_AV)
        Alambda = modified_calzetti_dust_Alambda_per_AV(wave, dust_index=dust_index1, bump_amp=bump_amp) * dust_AV

    return Alambda


# Modify _process_pixel_data to use global particle data and handle separate Doppler shifts
def _process_pixel_data(ii, jj, star_particle_membership_list, gas_particle_membership_list):
    """
    ii=y jj=x
    Worker function to process calculations for a single pixel (ii, jj).
    This function will be executed in parallel.
    
    star_particle_membership_list: List of (original_particle_index, line_of_sight_distance) for THIS pixel.
    gas_particle_membership_list: List of (original_particle_index, line_of_sight_distance) for THIS pixel.
    """

    # Use the pre-defined _worker_output_obs_wave_grid for the output spectra dimensions
    current_num_obs_wave_points = len(_worker_output_obs_wave_grid) if output_pixel_spectra_flag else 0

    pixel_results = {
        'map_stars_mass': 0.0,
        'map_mw_age': 0.0,
        'map_stars_mw_zsol': 0.0,
        'map_sfr_100': 0.0,
        'map_sfr_30': 0.0,
        'map_sfr_10': 0.0,
        'map_gas_mass': 0.0,
        'map_sfr_inst': 0.0,
        'map_gas_mw_zsol': 0.0,
        'map_dust_mean_tauV': 0.0,
        'map_dust_mean_AV': 0.0,
        'map_flux': np.zeros(len(_worker_filters)),
        'map_flux_dust': np.zeros(len(_worker_filters)),
        'obs_spectra_nodust_igm': np.zeros(current_num_obs_wave_points),
        'obs_spectra_dust_igm': np.zeros(current_num_obs_wave_points),
        'map_lw_age_nodust': np.nan,
        'map_lw_age_dust': np.nan,
        'map_lw_zsol_nodust': np.nan,
        'map_lw_zsol_dust': np.nan,
        'map_stars_mw_vel_los': np.nan,
        'map_gas_mw_vel_los': np.nan,
        'map_stars_vel_disp_los': np.nan,
        'map_gas_vel_disp_los': np.nan,
        'map_lw_vel_los_nodust': np.nan,
        'map_lw_vel_los_dust': np.nan,
        'map_lw_vel_los_nebular': np.nan # New: for nebular light-weighted velocity
    }

    star_ids0 = np.asarray([x[0] for x in star_particle_membership_list], dtype=int)
    star_los_dist0 = np.asarray([x[1] for x in star_particle_membership_list])

    # Access particle data from worker-global variables
    stars_mass = _worker_stars_mass
    stars_age = _worker_stars_age
    stars_zmet = _worker_stars_zmet
    stars_init_mass = _worker_stars_init_mass
    stars_vel_los_proj = _worker_stars_vel_los_proj
    stars_coords = _worker_stars_coords 

    gas_mass = _worker_gas_mass
    gas_sfr_inst = _worker_gas_sfr_inst
    gas_zmet = _worker_gas_zmet
    gas_log_temp = _worker_gas_log_temp
    gas_mass_H = _worker_gas_mass_H
    gas_vel_los_proj = _worker_gas_vel_los_proj
    gas_coords = _worker_gas_coords 

    idx_valid_stars_in_pixel = np.where((np.isnan(stars_mass[star_ids0]) == False) &
                                        (np.isnan(stars_age[star_ids0]) == False) &
                                        (np.isnan(stars_zmet[star_ids0]) == False) &
                                        (np.isnan(stars_vel_los_proj[star_ids0]) == False))[0]
    star_ids = star_ids0[idx_valid_stars_in_pixel]
    star_los_dist = star_los_dist0[idx_valid_stars_in_pixel]

    gas_ids0 = np.asarray([x[0] for x in gas_particle_membership_list], dtype=int)
    gas_los_dist0 = np.asarray([x[1] for x in gas_particle_membership_list])
    
    idxg = np.where((np.isnan(gas_mass[gas_ids0]) == False) &
                    (np.isnan(gas_vel_los_proj[gas_ids0]) == False))[0]
    gas_ids = gas_ids0[idxg]
    gas_los_dist = gas_los_dist0[idxg]

    current_stars_mass_sum = np.nansum(stars_mass[star_ids])
    pixel_results['map_stars_mass'] = current_stars_mass_sum

    if current_stars_mass_sum > 0:
        pixel_results['map_mw_age'] = np.nansum(stars_mass[star_ids] * stars_age[star_ids]) / current_stars_mass_sum
        pixel_results['map_stars_mw_zsol'] = np.nansum(stars_mass[star_ids] * stars_zmet[star_ids] / FSPS_Z_SUN) / current_stars_mass_sum
        pixel_results['map_stars_mw_vel_los'] = np.nansum(stars_mass[star_ids] * stars_vel_los_proj[star_ids]) / current_stars_mass_sum
        pixel_results['map_stars_vel_disp_los'] = np.sqrt(np.nansum(stars_mass[star_ids] * (stars_vel_los_proj[star_ids] - pixel_results['map_stars_mw_vel_los'])**2) / current_stars_mass_sum)
    else:
        pixel_results['map_mw_age'] = np.nan
        pixel_results['map_stars_mw_zsol'] = np.nan
        pixel_results['map_stars_mw_vel_los'] = np.nan
        pixel_results['map_stars_vel_disp_los'] = np.nan

    idxs3_100 = np.where((np.isnan(stars_init_mass[star_ids]) == False) & (stars_age[star_ids] <= 0.1))[0]
    pixel_results['map_sfr_100'] = np.nansum(stars_init_mass[star_ids[idxs3_100]]) / 0.1 / 1e+9

    idxs3_30 = np.where((np.isnan(stars_init_mass[star_ids]) == False) & (stars_age[star_ids] <= 0.03))[0]
    pixel_results['map_sfr_30'] = np.nansum(stars_init_mass[star_ids[idxs3_30]]) / 0.03 / 1e+9

    idxs3_10 = np.where((np.isnan(stars_init_mass[star_ids]) == False) & (stars_age[star_ids] <= 0.01))[0]
    pixel_results['map_sfr_10'] = np.nansum(stars_init_mass[star_ids[idxs3_10]]) / 0.01 / 1e+9

    current_gas_mass_sum = np.nansum(gas_mass[gas_ids])
    pixel_results['map_gas_mass'] = current_gas_mass_sum
    pixel_results['map_sfr_inst'] = np.nansum(gas_sfr_inst[gas_ids])

    if current_gas_mass_sum > 0:
        pixel_results['map_gas_mw_zsol'] = np.nansum(gas_mass[gas_ids] * gas_zmet[gas_ids] / FSPS_Z_SUN) / current_gas_mass_sum
        pixel_results['map_gas_mw_vel_los'] = np.nansum(gas_mass[gas_ids] * gas_vel_los_proj[gas_ids]) / current_gas_mass_sum
        pixel_results['map_gas_vel_disp_los'] = np.sqrt(np.nansum(gas_mass[gas_ids] * (gas_vel_los_proj[gas_ids] - pixel_results['map_gas_mw_vel_los'])**2) / current_gas_mass_sum)
    else:
        pixel_results['map_gas_mw_zsol'] = np.nan
        pixel_results['map_gas_mw_vel_los'] = np.nan
        pixel_results['map_gas_vel_disp_los'] = np.nan

    if len(star_ids) > 0:

        array_spec = []
        array_spec_dust = []
        array_AV = []
        array_tauV = []
        
        # New arrays for light-weighted calculations
        array_L_nodust = []
        array_L_dust = []
        array_vel_los = [] # For stellar light-weighted velocity
        array_L_nebular = [] # New: For nebular light-weighted velocity
        array_vel_los_nebular_weighted = [] # New: For nebular light-weighted velocity (based on gas)

        wave = ssp_wave

        # Define the rest-frame wavelength range for light-weighting
        lw_wave_idx = np.where((wave >= _lw_wave_min_rest) & (wave <= _lw_wave_max_rest))[0]
        if lw_wave_idx.size == 0:
            # If no wavelengths fall in range, light-weighted quantities will be NaN
            pass

        for i_sid in range(len(star_ids)):
            star_id = star_ids[i_sid]
            star_vel_los_current = stars_vel_los_proj[star_id]
            star_coords_current = stars_coords[star_id] # FIX: Define stars_coords_current here

            # --- Get Stellar Continuum and Nebular Emission Spectra ---
            spec_stellar_continuum = None
            spec_nebular_emission = None

            if use_precomputed_ssp:
                # Convert simulation metallicity (Z/Z_primordial_solar) to log(Z/Z_ssp_code_solar)
                particle_logzsol_ssp_code = np.log10(stars_zmet[star_id]/ssp_code_z_sun)
                
                points = np.array([[stars_age[star_id], particle_logzsol_ssp_code]])

                if ssp_interpolation_method == 'linear':
                    spec_stellar_continuum = _global_ssp_stellar_continuum_interpolator(points)[0]
                    spec_nebular_emission = _global_ssp_nebular_emission_interpolator(points)[0]
                    ssp_mass_formed = _global_ssp_stellar_mass_interpolator(points)[0]
                elif ssp_interpolation_method == 'cubic':
                    spec_stellar_continuum = _global_ssp_stellar_continuum_interpolator(points)[0]
                    spec_nebular_emission = _global_ssp_nebular_emission_interpolator(points)[0]
                    ssp_mass_formed = _global_ssp_stellar_mass_interpolator(points)[0]
                else: # 'nearest' method
                    age_idx = np.argmin(np.abs(ssp_ages_gyr - stars_age[star_id]))
                    z_idx = np.argmin(np.abs(ssp_logzsol_grid - particle_logzsol_ssp_code))
                    
                    spec_stellar_continuum = ssp_stellar_continuum_grid[age_idx, z_idx, :]
                    spec_nebular_emission = ssp_nebular_emission_grid[age_idx, z_idx, :]
                    ssp_mass_formed = ssp_stellar_mass_grid[age_idx, z_idx]
            else: # On-the-fly FSPS
                logzsol = np.log10(stars_zmet[star_id]/FSPS_Z_SUN)
                sp_instance.params["logzsol"] = logzsol   
                sp_instance.params['gas_logz'] = logzsol

                # Get total spectrum (stellar + nebular)
                sp_instance.params["add_neb_emission"] = 1
                _, spec_total_from_fsps = sp_instance.get_spectrum(peraa=True, tage=stars_age[star_id])
                
                # Get stellar continuum only spectrum
                sp_instance.params["add_neb_emission"] = 0
                _, spec_stellar_continuum = sp_instance.get_spectrum(peraa=True, tage=stars_age[star_id])

                # Get nebular emission by subtraction
                spec_nebular_emission = spec_total_from_fsps - spec_stellar_continuum
                ssp_mass_formed = sp_instance.stellar_mass

            # --- Apply Doppler shift to Stellar Continuum ---
            wave_doppler_stellar, spec_stellar_continuum_doppler = doppler_shift_spectrum(wave, spec_stellar_continuum, star_vel_los_current)
            # Interpolate back to original wavelength grid
            spec_stellar_continuum_interp = interp1d(wave_doppler_stellar, spec_stellar_continuum_doppler, kind='linear', bounds_error=False, fill_value=0.0)(wave)

            # --- Determine gas velocity for Nebular Emission ---
            gas_vel_los_avg_for_nebular = 0.0 # Default if no suitable gas is found

            # Find gas particles in front of the star, star-forming, and within 300 pc 3D distance
            # Using current pixel's gas_ids and original global gas_coords
            
            # 1. Filter gas particles in the current pixel that are "in front" (smaller LOS distance)
            idxg_front = np.where(gas_los_dist < star_los_dist[i_sid])[0]
            front_gas_ids = gas_ids[idxg_front] # These are original particle indices
            
            # 2. Further filter for star-forming gas (SFR > 0)
            idx_sf_gas_in_front = np.where(gas_sfr_inst[front_gas_ids] > 0.0)[0]
            sf_gas_ids_pre_distance_filter = front_gas_ids[idx_sf_gas_in_front]

            # 3. Further filter for gas within 300 pc (0.3 kpc) 3D distance from the star
            if len(sf_gas_ids_pre_distance_filter) > 0:
                dist_3d = np.linalg.norm(star_coords_current - gas_coords[sf_gas_ids_pre_distance_filter], axis=1)
                idx_within_dist = np.where(dist_3d < 0.3)[0] # 0.3 kpc for 300 pc
                sf_gas_ids_for_nebular = sf_gas_ids_pre_distance_filter[idx_within_dist]
            else:
                sf_gas_ids_for_nebular = np.array([], dtype=int)


            if len(sf_gas_ids_for_nebular) > 0:
                gas_vel_los_avg_for_nebular = np.nansum(gas_mass[sf_gas_ids_for_nebular] * gas_vel_los_proj[sf_gas_ids_for_nebular]) / np.nansum(gas_mass[sf_gas_ids_for_nebular])
            else:
                # If no star-forming gas found within criteria, use 0.0 velocity for nebular emission
                gas_vel_los_avg_for_nebular = 0.0


            # --- Apply Doppler shift to Nebular Emission ---
            wave_doppler_nebular, spec_nebular_emission_doppler = doppler_shift_spectrum(wave, spec_nebular_emission, gas_vel_los_avg_for_nebular)
            # Interpolate back to original wavelength grid
            spec_nebular_emission_interp = interp1d(wave_doppler_nebular, spec_nebular_emission_doppler, kind='linear', bounds_error=False, fill_value=0.0)(wave)

            # --- Combine Doppler-shifted components ---
            spec = spec_stellar_continuum_interp + spec_nebular_emission_interp
            spec_dust = spec.copy() # Initialize spec_dust with original combined spec

            # --- Apply dust attenuation from diffuse ISM ---
            # The filtering for cold_front_gas_ids might need adjustment based on its original intent.
            # Assuming this part remains for *diffuse ISM dust attenuation* (not directly related to nebular velocity source)
            # Original logic for cold_front_gas_ids based on gas_los_dist and temp:
            idxg_front_dust = np.where(gas_los_dist < star_los_dist[i_sid])[0]
            front_gas_ids_dust = gas_ids[idxg_front_dust]
            idx_cold_gas_for_dust = np.where((gas_sfr_inst[front_gas_ids_dust] > 0.0) | (gas_log_temp[front_gas_ids_dust] < 3.9))[0]
            cold_front_gas_ids = front_gas_ids_dust[idx_cold_gas_for_dust]

            if len(cold_front_gas_ids) > 0:
                temp_mw_gas_zsol = np.nansum(gas_mass[cold_front_gas_ids]*gas_zmet[cold_front_gas_ids]/FSPS_Z_SUN)/np.nansum(gas_mass[cold_front_gas_ids])
                nH = np.nansum(gas_mass_H[cold_front_gas_ids])*1.247914e+14/pix_area_kpc2
                tauV = _worker_scale_dust_tau * temp_mw_gas_zsol * nH / 2.1e+21
                dust_AV = -2.5*np.log10((1.0 - np.exp(-1.0*tauV))/tauV)

                if not (np.isnan(dust_AV) or dust_AV == 0.0):
                    if dust_law <= 1:
                        Alambda = dust_reddening_diffuse_ism(dust_AV, wave, dust_law)
                    else:
                        Alambda = dust_Alambda_per_AV * dust_AV
                    
                    spec_dust = spec_dust*np.power(10.0, -0.4*Alambda)
                    array_tauV.append(tauV)
                    array_AV.append(dust_AV)
            else:
                dust_AV = 0.0
                
            # --- Apply dust attenuation from birth clouds ---
            if stars_age[star_id] <= t_esc:
                Alambda = unresolved_dust_birth_cloud_Alambda_per_AV(wave, dust_index_bc=dust_index_bc) * dust_AV * dust_eta
                spec_dust = spec_dust*np.power(10.0, -0.4*Alambda)
    
            norm = stars_mass[star_id] / ssp_mass_formed

            if len(np.asarray(spec_dust).shape) == 1:
                array_spec.append(spec*norm)
                array_spec_dust.append(spec_dust*norm)

                # Calculate luminosity for light-weighting
                # Integrate over specified rest-frame wavelength range
                if lw_wave_idx.size > 1: # Need at least 2 points to integrate
                    L_nodust_particle = simpson(spec[lw_wave_idx]*norm, wave[lw_wave_idx])
                    L_dust_particle = simpson(spec_dust[lw_wave_idx]*norm, wave[lw_wave_idx])
                    L_nebular_particle = simpson(spec_nebular_emission_interp[lw_wave_idx]*norm, wave[lw_wave_idx]) # New: Nebular luminosity
                else: # No valid wavelength range for integration
                    L_nodust_particle = 0.0
                    L_dust_particle = 0.0
                    L_nebular_particle = 0.0
                
                array_L_nodust.append(L_nodust_particle)
                array_L_dust.append(L_dust_particle)
                array_vel_los.append(star_vel_los_current)
                array_L_nebular.append(L_nebular_particle) # New
                array_vel_los_nebular_weighted.append(gas_vel_los_avg_for_nebular) # New

        array_AV, array_tauV = np.asarray(array_AV), np.asarray(array_tauV)
        if array_AV.size == 0:
            mean_AV = np.nan
            mean_tauV = np.nan
        else:
            mean_AV = np.nanmean(np.asarray(array_AV))
            mean_tauV = np.nanmean(np.asarray(array_tauV))

        redshift_flux, redshift_flux_dust = [], []
        
        if len(array_spec) > 0:
            spec_lum = np.nansum(array_spec, axis=0)
            spec_lum_dust = np.nansum(array_spec_dust, axis=0)

            spec_wave_obs, spec_flux_obs = cosmo_redshifting(wave, spec_lum, snap_z, _worker_cosmo)
            spec_wave_obs, spec_flux_dust_obs = cosmo_redshifting(wave, spec_lum_dust, snap_z, _worker_cosmo)

            spec_flux_obs_igm = spec_flux_obs * igm_trans
            spec_flux_dust_obs_igm = spec_flux_dust_obs * igm_trans

            if output_pixel_spectra_flag:
                # Interpolate onto the fixed output observed-frame wavelength grid
                if _worker_output_obs_wave_grid.size > 0:
                    interp_func_nodust = interp1d(spec_wave_obs, spec_flux_obs_igm, kind='linear', 
                                                  bounds_error=False, fill_value=0.0)
                    interp_func_dust = interp1d(spec_wave_obs, spec_flux_dust_obs_igm, kind='linear', 
                                                bounds_error=False, fill_value=0.0)
                    
                    pixel_results['obs_spectra_nodust_igm'] = interp_func_nodust(_worker_output_obs_wave_grid)
                    pixel_results['obs_spectra_dust_igm'] = interp_func_dust(_worker_output_obs_wave_grid)
                else:
                    # If the target output grid is empty, ensure the output spectra arrays are also empty
                    pixel_results['obs_spectra_nodust_igm'] = np.zeros(0)
                    pixel_results['obs_spectra_dust_igm'] = np.zeros(0)


            nbands = len(_worker_filters)
            redshift_flux = np.zeros(nbands)
            redshift_flux_dust = np.zeros(nbands)

            for i_band in range(nbands):
                redshift_flux[i_band] = filtering(spec_wave_obs, spec_flux_obs_igm, _worker_filter_transmission[_worker_filters[i_band]]['wave'], _worker_filter_transmission[_worker_filters[i_band]]['trans'])
                redshift_flux_dust[i_band] = filtering(spec_wave_obs, spec_flux_dust_obs_igm, _worker_filter_transmission[_worker_filters[i_band]]['wave'], _worker_filter_transmission[_worker_filters[i_band]]['trans'])

            if len(redshift_flux) > 0:
                pixel_results['map_flux'] = redshift_flux
                pixel_results['map_flux_dust'] = redshift_flux_dust

                pixel_results['map_dust_mean_tauV'] = mean_tauV
                pixel_results['map_dust_mean_AV'] = mean_AV

            # Calculate Light-weighted age and metallicity (stellar light)
            total_L_nodust = np.nansum(array_L_nodust)
            total_L_dust = np.nansum(array_L_dust)
            array_vel_los = np.asarray(array_vel_los)

            if total_L_nodust > 0:
                pixel_results['map_lw_age_nodust'] = np.nansum(np.asarray(array_L_nodust) * stars_age[star_ids]) / total_L_nodust
                pixel_results['map_lw_zsol_nodust'] = np.nansum(np.asarray(array_L_nodust) * stars_zmet[star_ids] / FSPS_Z_SUN) / total_L_nodust
                pixel_results['map_lw_vel_los_nodust'] = np.nansum(np.asarray(array_L_nodust) * array_vel_los) / total_L_nodust
            else:
                pixel_results['map_lw_age_nodust'] = np.nan
                pixel_results['map_lw_zsol_nodust'] = np.nan
                pixel_results['map_lw_vel_los_nodust'] = np.nan

            if total_L_dust > 0:
                pixel_results['map_lw_age_dust'] = np.nansum(np.asarray(array_L_dust) * stars_age[star_ids]) / total_L_dust
                pixel_results['map_lw_zsol_dust'] = np.nansum(np.asarray(array_L_dust) * stars_zmet[star_ids] / FSPS_Z_SUN) / total_L_dust
                pixel_results['map_lw_vel_los_dust'] = np.nansum(np.asarray(array_L_dust) * array_vel_los) / total_L_dust
            else:
                pixel_results['map_lw_age_dust'] = np.nan
                pixel_results['map_lw_zsol_dust'] = np.nan
                pixel_results['map_lw_vel_los_dust'] = np.nan

            # New: Calculate Light-weighted velocity for nebular emission
            total_L_nebular = np.nansum(array_L_nebular)
            if total_L_nebular > 0:
                pixel_results['map_lw_vel_los_nebular'] = np.nansum(np.asarray(array_L_nebular) * array_vel_los_nebular_weighted) / total_L_nebular
            else:
                pixel_results['map_lw_vel_los_nebular'] = np.nan
                
    return ii, jj, pixel_results


def generate_images(sim_file, z, filters, filter_transmission_path, dim_kpc=None,
                    pix_arcsec=0.02, flux_unit='MJy/sr', polar_angle_deg=0, azimuth_angle_deg=0,
                    name_out_img=None, n_jobs=-1, ssp_code='FSPS', imf_type=1, imf_upper_limit=120.0, imf_lower_limit=0.08,
                    imf1=1.3, imf2=2.3, imf3=2.3, vdmc=0.08, mdave=0.5, gas_logu=-2.0,
                    igm_type=0, dust_index_bc=-0.7, dust_index=0.0, t_esc=0.01, dust_eta=1.0,
                    scale_dust_redshift="Vogelsberger20", cosmo_str='Planck18',  
                    dust_law=0, bump_amp=0.85, relation_AVslope="Salim18", salim_a0=-4.30, 
                    salim_a1=2.71, salim_a2= -0.191, salim_a3=0.0121, salim_RV=3.15, salim_B=3.15,
                    initdim_kpc=200, initdim_mass_fraction=0.99, use_precomputed_ssp=True, 
                    ssp_filepath=None, ssp_interpolation_method='nearest', 
                    output_pixel_spectra=False, rest_wave_min=1000.0, rest_wave_max=30000.0, rest_delta_wave=5.0): 
    """
    Generates astrophysical images from HDF5 simulation data with parallelized pixel calculations.
    Allows choice between using pre-computed SSP spectra from an HDF5 file or
    generating them on-the-fly using FSPS.
    Optionally outputs observed-frame spectra (redshifted and IGM-transmitted) for each pixel.

    Parameters:
        sim_file (str): Path to the HDF5 simulation file.
        z (float): Redshift of the galaxy.
        filters (list): List of photometric filters.
        filter_transmission_path (dict): Dictionary of paths to text files containing the transmission function.
                                         Keys are filter names, values are file paths. Each text file has
                                         two columns: wavelength, transmission.
        dim_kpc (float, optional): Dimension of the image in kpc. If None, assigned automatically. Defaults to None.
        pix_arcsec (float, optional): Pixel size in arcseconds. Defaults to 0.02.
        flux_unit (string, optional): Desired flux unit for the generated images. Options are: 'MJy/sr', 'nJy', 'AB magnitude', or 'erg/s/cm2/A'. Default to 'MJy/sr'.
        polar_angle_deg (float, optional): Polar angle for projection. Defaults to 0.
        azimuth_angle_deg (float, optional): Azimuth angle for projection. Defaults to 0.
        name_out_img (str, optional): Output file name for images. Defaults to None.
        n_jobs (int, optional): Number of CPU cores to use for parallel processing. Defaults to -1 (all available).
        ssp_code (str, optional): The SSP code to use ('FSPS' or 'Bagpipes'). Defaults to 'FSPS'.
        imf_type (int, optional): IMF type for FSPS (must match SSP grid if pre-computed). Defaults to 1.
        imf_upper_limit (float, optional): The upper limit of the IMF, in solar masses. Only used if `use_precomputed_ssp` is False or for consistency check if `use_precomputed_ssp` is True. Defaults to 120.0.
        imf_lower_limit (float, optional): The lower limit of the IMF, in solar masses. Only used if `use_precomputed_ssp` is False or for consistency check if `use_precomputed_ssp` is True. Defaults to 0.08.
        imf1 (float, optional): Logarithmic slope of the IMF over the range. Only used if `imf_type=2`. Only used if `use_precomputed_ssp` is False or for consistency check if `use_precomputed_ssp` is True. Defaults to 1.3.
        imf2 (float, optional): Logarithmic slope of the IMF over the range. Only used if `imf_type=2`. Only used if `use_precomputed_ssp` is False or for consistency check if `use_precomputed_ssp` is True. Defaults to 2.3.
        imf3 (float, optional): Logarithmic slope of the IMF over the range. Only used if `imf_type=2`. Only used if `use_precomputed_ssp` is False or for consistency check if `use_precomputed_ssp` is True. Defaults to 2.3.
        vdmc (float, optional): IMF parameter defined in van Dokkum (2008). Only used if `imf_type=3`. Only used if `use_precomputed_ssp` is False or for consistency check if `use_precomputed_ssp` is True. Defaults to 0.08.
        mdave (float, optional): IMF parameter defined in Dave (2008). Only used if `imf_type=4`. Only used if `use_precomputed_ssp` is False or for consistency check if `use_precomputed_ssp` is True. Defaults to 0.5.
        gas_logu (float, optional): Log ionization parameter (must match SSP grid if pre-computed). Defaults to -2.0.
        igm_type (int, optional): IGM absorption model type. Defaults to 0.
        dust_index_bc (float, optional): Dust index for birth clouds. Defaults to -0.7.
        dust_index (float, optional): Dust index for diffuse ISM (if applicable). Defaults to 0.0.
        t_esc (float, optional): Escape time for young stars. Defaults to 0.01.
        dust_eta (float, optional): Ratio of the dust attenuation A_V in the birth clouds and the diffuse ISM. Defaults to 1.0.
        scale_dust_redshift (str or dict, optional): Defines the dust_tau normalization vs redshift relation.
                                                     Can be a string ("Vogelsberger20") or a dictionary with "z" and "tau_dust" keys (1D arrays).
                                                     Defaults to "Vogelsberger20".
        cosmo_str (str, optional): Cosmology string. Defaults to 'Planck18'.
        dust_law (int, optional): Dust attenuation law type. Defaults to 0.
        bump_amp (float, optional): UV bump amplitude. Defaults to 0.85.
        relation_AVslope (str or dict, optional): Defines the A_V vs dust_index relation.
                                                  Can be a string ("Salim18", "Nagaraj22", "Battisti19")
                                                  or a dictionary with "AV" and "dust_index" keys (1D arrays).
                                                  Defaults to "Salim18".
        salim_a0, salim_a1, salim_a2, salim_a3, salim_RV, salim_B (float, optional): Parameters for Salim+2018 dust law.
        initdim_kpc (float, optional): Initial guess for image dimension in kpc. Defaults to 100.
        initdim_mass_fraction (float, optional): Mass fraction to determine initial image dimension. Defaults to 0.99. 
        use_precomputed_ssp (bool, optional): If True, use pre-computed SSP spectra. If False, generate on-the-fly.
                                              Defaults to True.
        ssp_filepath (str, optional): Path to the pre-computed SSP spectra HDF5 file.
                                      Only used if `use_precomputed_ssp` is True. Defaults to "ssp_spectra.hdf5".
        ssp_interpolation_method (str, optional): Method for interpolating SSPs if `use_precomputed_ssp` is True.
                                                  Options: 'nearest', 'linear', 'cubic'. Defaults to 'nearest'.
        output_pixel_spectra (bool, optional): If True, output observed-frame spectra for each pixel. Defaults to False.
        rest_wave_min (float, optional): Minimum rest-frame wavelength for output spectra (Angstrom). Defaults to 1000.0.
        rest_wave_max (float, optional): Maximum rest-frame wavelength for output spectra (Angstrom). Defaults to 30000.0.
        rest_delta_wave (float, optional): Incremental wavelength in rest-frame for output spectra (Angstrom). Defaults to 5.0.
    """

    cosmo = define_cosmo(cosmo_str)

    print ('Processing '+sim_file)
    # --- Data Loading and Initial Calculations (Sequential) ---

    # Load filter transmission and pivot wavelengths once for the main process
    filter_transmission_data_global, filter_wave_pivot_data_global = _load_filter_transmission_from_paths(filters, filter_transmission_path)

    snap_z = z
    snap_a = 1.0/(1.0 + snap_z)
    snap_univ_age = cosmo.age(snap_z).value

    pix_kpc = angular_to_physical(snap_z, pix_arcsec, cosmo)
    pix_area_kpc2 = pix_kpc*pix_kpc
    print ('pixel size: %lf arcsec or %lf kpc' % (pix_arcsec,pix_kpc))

    f = h5py.File(sim_file,'r')

    # get star particles data
    stars_init_mass = f['star']['init_mass'][:]
    stars_form_z = f['star']['form_z'][:]
    stars_mass = f['star']['mass'][:]
    stars_zmet = f['star']['zmet'][:]
    stars_coords = f['star']['coords'][:]   # coordinates (N,3) in units of kpc
    stars_vel = f['star']['vel'][:]         # velocities (N,3) in units km/s

    stars_form_age_univ = interp_age_univ_from_z(stars_form_z, cosmo)
    stars_age = snap_univ_age - stars_form_age_univ                 # age in Gyr

    idx = np.where(stars_age>=0)[0]
    stars_init_mass = stars_init_mass[idx]
    stars_mass = stars_mass[idx]
    stars_zmet = stars_zmet[idx]
    stars_age = stars_age[idx]
    stars_coords = stars_coords[idx,:]
    stars_vel = stars_vel[idx,:]

    # get gas particles data
    gas_mass = f['gas']['mass'][:]
    gas_zmet = f['gas']['zmet'][:]
    gas_sfr_inst = f['gas']['sfr_inst'][:]
    gas_log_temp = np.log10(f['gas']['temp'][:])
    gas_coords = f['gas']['coords'][:]
    gas_vel = f['gas']['vel'][:]
    gas_mass_H = f['gas']['mass_H'][:]

    f.close()

    if dim_kpc is None:
        dim_kpc = determine_image_size(stars_coords, stars_mass, pix_kpc, (initdim_kpc, initdim_kpc), 
                                       polar_angle_deg, azimuth_angle_deg, gas_coords, gas_mass, 
                                       mass_percentage=initdim_mass_fraction, max_img_dim=initdim_kpc)

    output_dimension = (dim_kpc, dim_kpc)
    star_particle_membership, star_mass_density_map, central_pixel_coords, grid_info, gas_particle_membership, gas_mass_density_map, stars_vel_los_proj, gas_vel_los_proj = get_2d_density_projection_no_los_binning(
                                                                                                                                                stars_coords, 
                                                                                                                                                stars_mass, 
                                                                                                                                                pix_kpc, 
                                                                                                                                                output_dimension, 
                                                                                                                                                polar_angle_deg=polar_angle_deg, 
                                                                                                                                                azimuth_angle_deg=azimuth_angle_deg, 
                                                                                                                                                gas_coords=gas_coords, 
                                                                                                                                                gas_masses=gas_mass,
                                                                                                                                                star_vels=stars_vel,
                                                                                                                                                gas_vels=gas_vel)
    
    dimx, dimy = grid_info['num_pixels_x'], grid_info['num_pixels_y']
    print ('Cutout size: %d x %d pix or %d x %d kpc' % (dimx,dimy,dim_kpc,dim_kpc))

    # --- Dust normalization loading ---
    norm_dust_z = None
    norm_dust_tau = None
    if isinstance(scale_dust_redshift, str):
        if scale_dust_redshift == "Vogelsberger20":
            data_file_name = "Vogelsberger20_scale_dust.txt"
            try:
                data_path = str(importlib.resources.files('galsyn.data').joinpath(data_file_name))
                data = np.loadtxt(data_path)
                norm_dust_z = data[:,0]
                norm_dust_tau = data[:,1]
            except Exception as e:
                print(f"Error loading dust normalization data from {data_file_name}: {e}")
                sys.exit(1)
        else:
            print(f"Error: Unknown string option for scale_dust_redshift: {scale_dust_redshift}")
            sys.exit(1)
    elif isinstance(scale_dust_redshift, dict):
        norm_dust_z = np.asarray(scale_dust_redshift["z"])
        norm_dust_tau = np.asarray(scale_dust_redshift["tau_dust"])
    else:
        print("Error: Invalid scale_dust_redshift type passed to generate_images.")
        sys.exit(1)

    # Calculate scale_dust_tau here in the main process
    scale_dust_tau = tau_dust_given_z(snap_z, norm_dust_z, norm_dust_tau)
    # --- End of dust normalization loading and calculation ---

    nbands = len(filters)

    # --- Define the fixed observed-frame wavelength grid for output spectra ---
    fixed_global_output_obs_wave = np.array([])
    if output_pixel_spectra:
        obs_min_boundary = rest_wave_min * (1.0 + snap_z)
        obs_max_boundary = rest_wave_max * (1.0 + snap_z)
        obs_delta_wave = rest_delta_wave * (1.0 + snap_z)
        # Create a linear wavelength grid with 5 Angstrom increment
        fixed_global_output_obs_wave = np.arange(obs_min_boundary, obs_max_boundary + obs_delta_wave, obs_delta_wave)
        if fixed_global_output_obs_wave.size == 0:
            print(f"Warning: Calculated output observed wavelength grid is empty for rest-frame range "
                  f"[{rest_wave_min}-{rest_wave_max}] Angstroms at z={snap_z}. "
                  f"Observed range: [{obs_min_boundary:.1f}-{obs_max_boundary:.1f}] Angstroms. "
                  f"Output spectra will be empty for all pixels.")

    num_obs_wave_points = fixed_global_output_obs_wave.size if output_pixel_spectra else 0

    map_mw_age = np.zeros((dimy,dimx))
    map_stars_mw_zsol = np.zeros((dimy,dimx))
    map_stars_mass = np.zeros((dimy,dimx))
    map_sfr_100 = np.zeros((dimy,dimx))
    map_sfr_30 = np.zeros((dimy,dimx))
    map_sfr_10 = np.zeros((dimy,dimx))
    map_gas_mw_zsol = np.zeros((dimy,dimx))
    map_gas_mass = np.zeros((dimy,dimx))
    map_sfr_inst = np.zeros((dimy,dimx))
    map_dust_mean_tauV = np.zeros((dimy,dimx))
    map_dust_mean_AV = np.zeros((dimy,dimx))
    map_flux = np.zeros((dimy,dimx,nbands))
    map_flux_dust = np.zeros((dimy,dimx,nbands))
    
    # Initialize spectra maps directly with the fixed size
    map_spectra_nodust = np.zeros((dimy, dimx, num_obs_wave_points)) if output_pixel_spectra else None
    map_spectra_dust = np.zeros((dimy, dimx, num_obs_wave_points)) if output_pixel_spectra else None

    # New maps for light-weighted age and metallicity
    map_lw_age_nodust = np.full((dimy, dimx), np.nan, dtype=np.float32)
    map_lw_age_dust = np.full((dimy, dimx), np.nan, dtype=np.float32)
    map_lw_zsol_nodust = np.full((dimy, dimx), np.nan, dtype=np.float32)
    map_lw_zsol_dust = np.full((dimy, dimx), np.nan, dtype=np.float32)

    # New maps for velocity
    map_stars_mw_vel_los = np.full((dimy, dimx), np.nan, dtype=np.float32)
    map_gas_mw_vel_los = np.full((dimy, dimx), np.nan, dtype=np.float32)
    map_stars_vel_disp_los = np.full((dimy, dimx), np.nan, dtype=np.float32)
    map_gas_vel_disp_los = np.full((dimy, dimx), np.nan, dtype=np.float32)
    map_lw_vel_los_nodust = np.full((dimy, dimx), np.nan, dtype=np.float32)
    map_lw_vel_los_dust = np.full((dimy, dimx), np.nan, dtype=np.float32)
    map_lw_vel_los_nebular = np.full((dimy, dimx), np.nan, dtype=np.float32) # New: Nebular light-weighted velocity

    tasks = []
    for ii in range(dimy):
        for jj in range(dimx):
            num_stars_in_pixel = len(star_particle_membership[ii][jj])
            num_gas_in_pixel = len(gas_particle_membership[ii][jj]) if gas_particle_membership is not None else 0
            complexity = num_stars_in_pixel + num_gas_in_pixel
            tasks.append({'coords': (ii, jj), 'complexity': complexity, 
                          'star_part_mem': star_particle_membership[ii][jj],
                          'gas_part_mem': gas_particle_membership[ii][jj] if gas_particle_membership is not None else []})

    tasks.sort(key=lambda x: x['complexity'], reverse=True)

    # Prepare arguments for _process_pixel_data - ONLY task-specific data
    processed_tasks_args = []
    for task in tasks:
        ii, jj = task['coords']
        processed_tasks_args.append((ii, jj, task['star_part_mem'], task['gas_part_mem'])) # Removed large arrays

    num_cores = n_jobs
    if num_cores == -1:
        num_cores = multiprocessing.cpu_count()

    print(f"\nStarting parallel pixel processing on {num_cores} cores...")

    # Pass all large particle data arrays to init_worker
    with tqdm_joblib(total=len(processed_tasks_args), desc="Processing pixels") as progress_bar:
        results = Parallel(n_jobs=num_cores, verbose=0, initializer=init_worker,
                           initargs=(ssp_code, snap_z, pix_area_kpc2,
                                     filters, filter_transmission_path,
                                     imf_type, imf_upper_limit, imf_lower_limit,
                                     imf1, imf2, imf3, vdmc, mdave,
                                     gas_logu, igm_type, dust_index_bc,
                                     dust_index, t_esc, dust_eta, scale_dust_tau,
                                     cosmo_str, dust_law, bump_amp, relation_AVslope, 
                                     salim_a0, salim_a1, salim_a2, salim_a3, salim_RV, salim_B, use_precomputed_ssp, 
                                     stars_mass, stars_age, stars_zmet, stars_init_mass, stars_vel_los_proj, stars_coords,
                                     gas_mass, gas_sfr_inst, gas_zmet, gas_log_temp, gas_mass_H, gas_vel_los_proj, gas_coords,
                                     ssp_filepath, ssp_interpolation_method, 
                                     output_pixel_spectra, fixed_global_output_obs_wave))( 
            delayed(_process_pixel_data)(*task_args) for task_args in processed_tasks_args
        )
    #print("\nFinished parallel pixel processing.")

    for k, pixel_result_tuple in enumerate(results):
        ii, jj, pixel_data = pixel_result_tuple 
        
        original_ii, original_jj = ii, jj 
        
        map_stars_mass[original_ii][original_jj] = pixel_data['map_stars_mass']
        map_mw_age[original_ii][original_jj] = pixel_data['map_mw_age']
        map_stars_mw_zsol[original_ii][original_jj] = pixel_data['map_stars_mw_zsol']
        map_sfr_100[original_ii][original_jj] = pixel_data['map_sfr_100']
        map_sfr_30[original_ii][original_jj] = pixel_data['map_sfr_30']
        map_sfr_10[original_ii][original_jj] = pixel_data['map_sfr_10']

        map_gas_mass[original_ii][original_jj] = pixel_data['map_gas_mass']
        map_sfr_inst[original_ii][original_jj] = pixel_data['map_sfr_inst']
        map_gas_mw_zsol[original_ii][original_jj] = pixel_data['map_gas_mw_zsol']

        map_dust_mean_tauV[original_ii][original_jj] = pixel_data['map_dust_mean_tauV']
        map_dust_mean_AV[original_ii][original_jj] = pixel_data['map_dust_mean_AV']

        map_flux[original_ii][original_jj] = pixel_data['map_flux']
        map_flux_dust[original_ii][original_jj] = pixel_data['map_flux_dust']

        if output_pixel_spectra:
            # Directly assign as map_spectra_nodust is pre-allocated with correct size
            map_spectra_nodust[original_ii][original_jj] = pixel_data['obs_spectra_nodust_igm']
            map_spectra_dust[original_ii][original_jj] = pixel_data['obs_spectra_dust_igm']

        # Populate new light-weighted maps
        map_lw_age_nodust[original_ii][original_jj] = pixel_data['map_lw_age_nodust']
        map_lw_age_dust[original_ii][original_jj] = pixel_data['map_lw_age_dust']
        map_lw_zsol_nodust[original_ii][original_jj] = pixel_data['map_lw_zsol_nodust']
        map_lw_zsol_dust[original_ii][original_jj] = pixel_data['map_lw_zsol_dust']

        # Populate new velocity maps
        map_stars_mw_vel_los[original_ii][original_jj] = pixel_data['map_stars_mw_vel_los']
        map_gas_mw_vel_los[original_ii][original_jj] = pixel_data['map_gas_mw_vel_los']
        map_stars_vel_disp_los[original_ii][original_jj] = pixel_data['map_stars_vel_disp_los']
        map_gas_vel_disp_los[original_ii][original_jj] = pixel_data['map_gas_vel_disp_los']
        map_lw_vel_los_nodust[original_ii][original_jj] = pixel_data['map_lw_vel_los_nodust']
        map_lw_vel_los_dust[original_ii][original_jj] = pixel_data['map_lw_vel_los_dust']
        map_lw_vel_los_nebular[original_ii][original_jj] = pixel_data['map_lw_vel_los_nebular'] # New

    #print("All calculations complete. Maps populated.")

    for i_band in range(len(filters)): 
        map_flux[:,:,i_band] = convert_flux_map(map_flux[:,:,i_band], filter_wave_pivot_data_global[filters[i_band]], to_unit=flux_unit, pixel_scale_arcsec=pix_arcsec)
        map_flux_dust[:,:,i_band] = convert_flux_map(map_flux_dust[:,:,i_band], filter_wave_pivot_data_global[filters[i_band]], to_unit=flux_unit, pixel_scale_arcsec=pix_arcsec)

    if flux_unit == 'erg/s/cm2/A':
        flux_scale = 1e-20
    else:
        flux_scale = 1.0
    
    map_flux = map_flux/flux_scale
    map_flux_dust = map_flux_dust/flux_scale

    if name_out_img is not None:
        try:
            hdul = fits.HDUList()

            if map_flux.shape[2] > 0:
                primary_data = map_flux[:, :, 0]
                prihdr = fits.Header()
                prihdr['COMMENT'] = 'Primary Image: First band (no dust)'
                prihdr['CRPIX1'] = dimx / 2.0 + 0.5
                prihdr['CRPIX2'] = dimy / 2.0 + 0.5
                prihdr['CDELT1'] = pix_kpc
                prihdr['CDELT2'] = pix_kpc
                prihdr['CUNIT1'] = 'kpc'
                prihdr['CUNIT2'] = 'kpc'
                prihdr['REDSHIFT'] = snap_z
                prihdr['POLAR'] = polar_angle_deg
                prihdr['AZIMUTH'] = azimuth_angle_deg
                prihdr['DIM_KPC'] = dim_kpc
                prihdr['PIX_KPC'] = pix_kpc
                prihdr['PIXSIZE'] = pix_arcsec
                prihdr['BUNIT'] = flux_unit
                prihdr['SCALE'] = flux_scale
                prihdr['SSP_CODE'] = ssp_code

                primary_hdu = fits.PrimaryHDU(data=primary_data, header=prihdr)
                hdul.append(primary_hdu)

                for i_band in range(len(filters)):
                    ext_hdr = fits.Header()
                    ext_hdr['EXTNAME'] = 'NODUST_'+filters[i_band].upper() 
                    ext_hdr['FILTER'] = filters[i_band] 
                    ext_hdr['COMMENT'] = f'Flux for filter: {filters[i_band]}' 
                    ext_hdu = fits.ImageHDU(data=map_flux[:, :, i_band], header=ext_hdr)
                    hdul.append(ext_hdu)

                for i_band in range(len(filters)):
                    ext_hdr = fits.Header()
                    ext_hdr['EXTNAME'] = 'DUST_'+filters[i_band].upper() 
                    ext_hdr['FILTER'] = filters[i_band] 
                    ext_hdr['COMMENT'] = f'Flux (with dust) for filter: {filters[i_band]}' 
                    ext_hdu = fits.ImageHDU(data=map_flux_dust[:, :, i_band], header=ext_hdr)
                    hdul.append(ext_hdu)
            else:
                hdul.append(fits.PrimaryHDU())

            map_data_to_save = {
                'STARS_MASS': map_stars_mass,
                'MW_AGE': map_mw_age,
                'STARS_MW_ZSOL': map_stars_mw_zsol,
                'SFR_100MYR': map_sfr_100,
                'SFR_30MYR': map_sfr_30,
                'SFR_10MYR': map_sfr_10,
                'GAS_MASS': map_gas_mass,
                'SFR_INST': map_sfr_inst,
                'GAS_MW_ZSOL': map_gas_mw_zsol,
                'DUST_MEAN_TAUV': map_dust_mean_tauV,
                'DUST_MEAN_AV': map_dust_mean_AV,
                'LW_AGE_NODUST': map_lw_age_nodust,
                'LW_AGE_DUST': map_lw_age_dust,
                'LW_ZSOL_NODUST': map_lw_zsol_nodust,
                'LW_ZSOL_DUST': map_lw_zsol_dust,
                'STARS_MW_VEL_LOS': map_stars_mw_vel_los,
                'GAS_MW_VEL_LOS': map_gas_mw_vel_los,
                'STARS_VEL_DISP_LOS': map_stars_vel_disp_los,
                'GAS_VEL_DISP_LOS': map_gas_vel_disp_los,
                'LW_VEL_LOS_NODUST': map_lw_vel_los_nodust,
                'LW_VEL_LOS_DUST': map_lw_vel_los_dust,
                'LW_VEL_LOS_NEBULAR': map_lw_vel_los_nebular # New
            }

            for map_name, data_array in map_data_to_save.items():
                if data_array is not None:
                    ext_hdr = fits.Header()
                    ext_hdr['EXTNAME'] = map_name
                    ext_hdr['COMMENT'] = f'Map of {map_name.replace("_", " ").title()}'
                    if 'AGE' in map_name:
                        ext_hdr['BUNIT'] = 'Gyr'
                    elif 'ZSOL' in map_name:
                        ext_hdr['BUNIT'] = 'Z/Zsun'
                    elif 'VEL' in map_name:
                        ext_hdr['BUNIT'] = 'km/s'
                    hdul.append(fits.ImageHDU(data=data_array, header=ext_hdr))

            if output_pixel_spectra:
                # Transpose the data cube from (dim_y, dim_x, wavelength) to (wavelength, dim_y, dim_x)
                transposed_map_spectra_nodust = map_spectra_nodust.transpose((2, 0, 1))
                transposed_map_spectra_dust = map_spectra_dust.transpose((2, 0, 1))

                ext_hdr_nodust_spec = fits.Header()
                ext_hdr_nodust_spec['EXTNAME'] = 'OBS_SPEC_NODUST' 
                ext_hdr_nodust_spec['COMMENT'] = 'Observed-frame spectra (no dust attenuation)' 
                
                # Update CRPIX, CDELT, CUNIT to reflect new axis order
                ext_hdr_nodust_spec['CRPIX1'] = 1.0 # Wavelength axis becomes the first axis
                ext_hdr_nodust_spec['CRVAL1'] = fixed_global_output_obs_wave[0] if fixed_global_output_obs_wave.size > 0 else 0.0
                ext_hdr_nodust_spec['CDELT1'] = (fixed_global_output_obs_wave[1] - fixed_global_output_obs_wave[0]) if fixed_global_output_obs_wave.size > 1 else 0.0
                ext_hdr_nodust_spec['CUNIT1'] = 'Angstrom'

                ext_hdr_nodust_spec['CRPIX2'] = dimy / 2.0 + 0.5 # dim_y becomes the second axis
                ext_hdr_nodust_spec['CDELT2'] = pix_kpc
                ext_hdr_nodust_spec['CUNIT2'] = 'kpc'

                ext_hdr_nodust_spec['CRPIX3'] = dimx / 2.0 + 0.5 # dim_x becomes the third axis
                ext_hdr_nodust_spec['CDELT3'] = pix_kpc
                ext_hdr_nodust_spec['CUNIT3'] = 'kpc'
                
                ext_hdr_nodust_spec['BUNIT'] = 'erg/s/cm2/Angstrom' 
                hdul.append(fits.ImageHDU(data=transposed_map_spectra_nodust, header=ext_hdr_nodust_spec))

                ext_hdr_dust_spec = fits.Header()
                ext_hdr_dust_spec['EXTNAME'] = 'OBS_SPEC_DUST' 
                ext_hdr_dust_spec['COMMENT'] = 'Observed-frame spectra (with dust attenuation)' 
                
                # Update CRPIX, CDELT, CUNIT to reflect new axis order
                ext_hdr_dust_spec['CRPIX1'] = 1.0 # Wavelength axis becomes the first axis
                ext_hdr_dust_spec['CRVAL1'] = fixed_global_output_obs_wave[0] if fixed_global_output_obs_wave.size > 0 else 0.0
                ext_hdr_dust_spec['CDELT1'] = (fixed_global_output_obs_wave[1] - fixed_global_output_obs_wave[0]) if fixed_global_output_obs_wave.size > 1 else 0.0
                ext_hdr_dust_spec['CUNIT1'] = 'Angstrom'

                ext_hdr_dust_spec['CRPIX2'] = dimy / 2.0 + 0.5 # dim_y becomes the second axis
                ext_hdr_dust_spec['CDELT2'] = pix_kpc
                ext_hdr_dust_spec['CUNIT2'] = 'kpc'

                ext_hdr_dust_spec['CRPIX3'] = dimx / 2.0 + 0.5 # dim_x becomes the third axis
                ext_hdr_dust_spec['CDELT3'] = pix_kpc
                ext_hdr_dust_spec['CUNIT3'] = 'kpc'

                ext_hdr_dust_spec['BUNIT'] = 'erg/s/cm2/Angstrom'
                hdul.append(fits.ImageHDU(data=transposed_map_spectra_dust, header=ext_hdr_dust_spec))
                
                # --- Add wavelength array as a binary table extension ---
                if fixed_global_output_obs_wave.size > 0:
                    col = fits.Column(name='WAVELENGTH', format='D', array=fixed_global_output_obs_wave)
                    cols = fits.ColDefs([col])
                    wavelength_hdu = fits.BinTableHDU.from_columns(cols, name='WAVELENGTH_GRID')
                    wavelength_hdu.header['BUNIT'] = 'Angstrom'
                    wavelength_hdu.header['COMMENT'] = 'Wavelength grid for OBS_SPEC_NODUST and OBS_SPEC_DUST'
                    hdul.append(wavelength_hdu)
                else:
                    print("Warning: No wavelength grid data to save for WAVELENGTH_GRID extension.")


            output_dir = os.path.dirname(name_out_img)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)

            hdul.writeto(name_out_img, overwrite=True, output_verify='fix')
            print(f"Galaxy image synthesis completed successfully and results saved to FITS file: {name_out_img}")

        except Exception as e:
            print(f"Error saving FITS file {name_out_img}: {e}")