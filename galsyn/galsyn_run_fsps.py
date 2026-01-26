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
from scipy.ndimage import zoom
import warnings

import fsps

FSPS_Z_SUN = 0.019

# Global variables for SSP data
ssp_wave = None
ssp_ages_gyr = None
ssp_logzsol_grid = None
ssp_stellar_mass_grid = None
ssp_code_z_sun = None
ssp_stellar_continuum_grid = None 
ssp_nebular_emission_grid = None    

_global_ssp_stellar_mass_interpolator = None
_global_ssp_stellar_continuum_interpolator = None
_global_ssp_nebular_emission_interpolator = None

sp_instance = None

# Global worker variables
igm_trans = None
snap_z = None
pix_area_kpc2 = None
gas_logu = None
igm_type = None
dust_index_bc = None
t_esc = None
dust_eta = None
dust_law = None

func_interp_dust_index = None
func_interp_bump_amp = None
func_interp_bump_dwave = None
dust_Alambda_per_AV = None

use_precomputed_ssp = False
ssp_interpolation_method = 'nearest'

_worker_scale_dust_tau = None
output_pixel_spectra_flag = False
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
_worker_stars_mass = None
_worker_stars_age = None
_worker_stars_zmet = None
_worker_stars_init_mass = None
_worker_stars_vel_los_proj = None
_worker_stars_coords = None 
_worker_gas_mass = None
_worker_gas_sfr_inst = None
_worker_gas_zmet = None
_worker_gas_log_temp = None
_worker_gas_mass_H = None
_worker_gas_vel_los_proj = None
_worker_gas_coords = None 

_lw_wave_min_rest = 1000.0 
_lw_wave_max_rest = 30000.0 


def rebin_map(data, factor, mode='sum'):
    """
    Rebins a 2D or 3D array with float factors and robust NaN/zero handling.
    Updated to prevent dilution of physical properties by empty pixels.
    
    'sum'  : Preserves total flux/mass (flux-conserving).
    'mean' : Preserves average property values (Age, Temperature, Velocity) 
             by averaging only pixels that actually contain data.
    """
    if factor == 1.0:
        return data

    zoom_rate = 1.0 / factor
    # Handle spatial scaling (assumes Y, X are first two dims)
    zoom_factors = (zoom_rate, zoom_rate, 1) if data.ndim == 3 else (zoom_rate, zoom_rate)

    if mode == 'mean':
        # Create a mask of pixels containing valid physical data (non-zero and non-NaN)
        # This prevents dilution of properties like Velocity or Z by empty pixels.
        mask = (data != 0) & (~np.isnan(data))
        
        # Prepare weighted data (zero out invalid pixels)
        data_weighted = np.where(mask, data, 0.0)
        
        # Resample both the data and the mask using bilinear interpolation
        # grid_mode=True ensures we treat pixels as areas, which is standard in astronomy.
        sum_val = zoom(data_weighted, zoom_factors, order=1, prefilter=False, 
                       grid_mode=True, mode='grid-constant')
        count_val = zoom(mask.astype(float), zoom_factors, order=1, prefilter=False, 
                         grid_mode=True, mode='grid-constant')
        
        # Calculate the local mean: divide resampled values by resampled mask
        # This effectively calculates: (Sum of valid sub-pixels) / (Number of valid sub-pixels)
        rescaled = np.divide(sum_val, count_val, out=np.zeros_like(sum_val), where=count_val > 0)

    elif mode == 'sum':
        # Use order=1 (bilinear) as it is the most stable for physical data.
        rescaled = zoom(data, zoom_factors, order=1, prefilter=False, 
                        grid_mode=True, mode='grid-constant')
        
        # Apply the area-scaling factor to preserve total flux/mass
        rescaled = rescaled * (factor**2)
        
        # Enforce exact flux conservation (ignoring NaNs)
        if data.ndim == 3:
            sum_in = np.nansum(data, axis=(0, 1))
            sum_out = np.nansum(rescaled, axis=(0, 1))
            nonzero = (sum_out != 0) & (~np.isnan(sum_out))
            rescaled[..., nonzero] *= (sum_in[nonzero] / sum_out[nonzero])
        else:
            sum_in = np.nansum(data)
            sum_out = np.nansum(rescaled)
            if sum_out != 0 and not np.isnan(sum_out):
                rescaled *= (sum_in / sum_out)

    return rescaled


def _load_filter_transmission_from_paths(filters_list, filter_transmission_path_dict):
    filter_transmission_data = {}
    filter_wave_pivot_data = {}
    for f_name in filters_list:
        file_path = filter_transmission_path_dict[f_name]
        data = np.loadtxt(file_path)
        wave, trans = data[:, 0], data[:, 1]
        filter_transmission_data[f_name] = {'wave': wave, 'trans': trans}
        num = simpson(wave * trans, wave)
        den = simpson(np.where(wave != 0, trans / wave, 0), wave)
        filter_wave_pivot_data[f_name] = np.sqrt(num / den) if den > 0 else np.nan
    return filter_transmission_data, filter_wave_pivot_data

def init_worker(ssp_code_val, snap_z_val, pix_area_kpc2_val,  
                filters_list_val, filter_transmission_path_val,
                imf_type_val, imf_upper_limit_val, imf_lower_limit_val, 
                imf1_val, imf2_val, imf3_val, vdmc_val, mdave_val,     
                gas_logu_val, igm_type_val, dust_index_bc_val, 
                dust_index_val, t_esc_val, dust_eta_val, precomputed_scale_dust_tau_val,
                cosmo_str_val, dust_law_val, bump_amp_val, bump_dwave_val,
                salim_a0_val, salim_a1_val, salim_a2_val, salim_a3_val, salim_RV_val, salim_B_val, 
                use_precomputed_ssp_val, stars_mass_arr, stars_age_arr, stars_zmet_arr, stars_init_mass_arr, 
                stars_vel_los_proj_arr, stars_coords_arr, gas_mass_arr, gas_sfr_inst_arr, gas_zmet_arr, 
                gas_log_temp_arr, gas_mass_H_arr, gas_vel_los_proj_arr, gas_coords_arr, ssp_filepath_val=None, 
                ssp_interpolation_method_val='nearest', output_pixel_spectra_val=False, 
                output_obs_wave_grid_val=None, dust_method_val='los', av_sfrden_relation_val=None, max_dist_neb_val=0.5): 
    
    global ssp_wave, ssp_ages_gyr, ssp_logzsol_grid, ssp_stellar_mass_grid, ssp_code_z_sun
    global ssp_stellar_continuum_grid, ssp_nebular_emission_grid 
    global _global_ssp_stellar_continuum_interpolator, _global_ssp_nebular_emission_interpolator, _global_ssp_stellar_mass_interpolator
    global sp_instance, igm_trans, snap_z, pix_area_kpc2
    global gas_logu, igm_type, dust_index_bc, t_esc, dust_eta, dust_law
    global salim_a0, salim_a1, salim_a2, salim_a3, salim_RV, salim_B, dust_Alambda_per_AV
    global func_interp_dust_index, func_interp_bump_amp, func_interp_bump_dwave
    global use_precomputed_ssp, ssp_interpolation_method, output_pixel_spectra_flag, _worker_output_obs_wave_grid 
    global _worker_filters, _worker_filter_transmission, _worker_filter_wave_eff, _worker_imf_type, _worker_cosmo
    global _worker_imf_upper_limit, _worker_imf_lower_limit, _worker_imf1, _worker_imf2, _worker_imf3, _worker_vdmc, _worker_mdave
    global _worker_scale_dust_tau, _worker_stars_mass, _worker_stars_age, _worker_stars_zmet, _worker_stars_init_mass, _worker_stars_vel_los_proj, _worker_stars_coords
    global _worker_gas_mass, _worker_gas_sfr_inst, _worker_gas_zmet, _worker_gas_log_temp, _worker_gas_mass_H, _worker_gas_vel_los_proj, _worker_gas_coords
    global _worker_dust_method, func_interp_av_sfrden, _worker_max_dist_neb

    # Map worker arrays
    _worker_stars_mass, _worker_stars_age, _worker_stars_zmet = stars_mass_arr, stars_age_arr, stars_zmet_arr
    _worker_stars_init_mass, _worker_stars_vel_los_proj, _worker_stars_coords = stars_init_mass_arr, stars_vel_los_proj_arr, stars_coords_arr
    _worker_gas_mass, _worker_gas_sfr_inst, _worker_gas_zmet = gas_mass_arr, gas_sfr_inst_arr, gas_zmet_arr
    _worker_gas_log_temp, _worker_gas_mass_H, _worker_gas_vel_los_proj, _worker_gas_coords = gas_log_temp_arr, gas_mass_H_arr, gas_vel_los_proj_arr, gas_coords_arr
    _worker_max_dist_neb = max_dist_neb_val

    # Basic setup
    snap_z, pix_area_kpc2, _worker_imf_type = snap_z_val, pix_area_kpc2_val, imf_type_val
    dust_law = dust_law_val
    dust_index_bc, t_esc, dust_eta, _worker_scale_dust_tau = dust_index_bc_val, t_esc_val, dust_eta_val, precomputed_scale_dust_tau_val
    _worker_cosmo = define_cosmo(cosmo_str_val)
    use_precomputed_ssp, ssp_interpolation_method = use_precomputed_ssp_val, ssp_interpolation_method_val 
    _worker_filters = filters_list_val
    _worker_filter_transmission, _worker_filter_wave_eff = _load_filter_transmission_from_paths(_worker_filters, filter_transmission_path_val)
    output_pixel_spectra_flag = output_pixel_spectra_val
    #_worker_output_obs_wave_grid = np.asarray(output_obs_wave_grid_val) if isinstance(output_obs_wave_grid_val, tuple) else output_obs_wave_grid_val
    _worker_output_obs_wave_grid = np.asarray(output_obs_wave_grid_val)

    # Dust Framework Setup
    _worker_dust_method = dust_method_val
    if _worker_dust_method == 'sfr_AV' and isinstance(av_sfrden_relation_val, dict):
        func_interp_av_sfrden = interp1d(av_sfrden_relation_val['log_SFR_density'], 
                                         av_sfrden_relation_val['log_AV'], 
                                         bounds_error=False, fill_value='extrapolate')

    if use_precomputed_ssp:
        with h5py.File(ssp_filepath_val, 'r') as f_ssp:
            ssp_wave = f_ssp['wavelength'][:]
            ssp_ages_gyr, ssp_logzsol_grid = f_ssp['ages_gyr'][:], f_ssp['logzsol'][:]
            ssp_stellar_mass_grid = f_ssp['stellar_mass'][:]
            ssp_stellar_continuum_grid, ssp_nebular_emission_grid = f_ssp['stellar_continuum_spectra'][:], f_ssp['nebular_emission_spectra'][:]
            ssp_code_z_sun = f_ssp.attrs['z_sun'] 
            method = ssp_interpolation_method if ssp_interpolation_method in ['linear', 'cubic'] else None
            if method:
                _global_ssp_stellar_continuum_interpolator = RegularGridInterpolator((ssp_ages_gyr, ssp_logzsol_grid), ssp_stellar_continuum_grid, method=method, bounds_error=False, fill_value=0.0)
                _global_ssp_nebular_emission_interpolator = RegularGridInterpolator((ssp_ages_gyr, ssp_logzsol_grid), ssp_nebular_emission_grid, method=method, bounds_error=False, fill_value=0.0)
                _global_ssp_stellar_mass_interpolator = RegularGridInterpolator((ssp_ages_gyr, ssp_logzsol_grid), ssp_stellar_mass_grid, method=method, bounds_error=False, fill_value=0.0)
    else:
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

    # Unified Option 0 Dust Interpolators
    if dust_law == 0:
        # Dust Index (Slope)
        if isinstance(dust_index_val, dict):
            func_interp_dust_index = interp1d(dust_index_val['AV'], dust_index_val['dust_index'], bounds_error=False, fill_value='extrapolate')
        else:
            func_interp_dust_index = lambda av: dust_index_val

        # Bump Amplitude
        if isinstance(bump_amp_val, dict):
            func_interp_bump_amp = interp1d(bump_amp_val['AV'], bump_amp_val['bump_amp'], bounds_error=False, fill_value='extrapolate')
        else:
            func_interp_bump_amp = lambda av: bump_amp_val

        # Bump Width (micron)
        if isinstance(bump_dwave_val, dict):
            func_interp_bump_dwave = interp1d(bump_dwave_val['AV'], bump_dwave_val['bump_dwave'], bounds_error=False, fill_value='extrapolate')
        else:
            func_interp_bump_dwave = lambda av: bump_dwave_val

    # Tabulated Dust Laws (4-9)
    elif dust_law == 1: dust_Alambda_per_AV = salim18_dust_Alambda_per_AV(ssp_wave, salim_a0_val, salim_a1_val, salim_a2_val, salim_a3_val, salim_B_val, salim_RV_val)
    elif dust_law == 2: dust_Alambda_per_AV = calzetti_dust_Alambda_per_AV(ssp_wave)
    elif dust_law == 3: dust_Alambda_per_AV = smc_gordon2003_dust_Alambda_per_AV(ssp_wave)
    elif dust_law == 4: dust_Alambda_per_AV = lmc_gordon2003_dust_Alambda_per_AV(ssp_wave)
    elif dust_law == 5: dust_Alambda_per_AV = ccm89_dust_Alambda_per_AV(ssp_wave)
    elif dust_law == 6: dust_Alambda_per_AV = fitzpatrick99_dust_Alambda_per_AV(ssp_wave)

    if igm_type == 0: igm_trans = igm_att_madau(ssp_wave * (1.0+snap_z), snap_z)
    else: igm_trans = igm_att_inoue(ssp_wave * (1.0+snap_z), snap_z)

def dust_reddening_diffuse_ism(dust_AV, wave, dust_law):
    """Calculates attenuation for Option 0 using interpolators or static laws."""
    if dust_law == 0:
        d_idx = func_interp_dust_index(dust_AV)
        b_amp = func_interp_bump_amp(dust_AV)
        b_w   = func_interp_bump_dwave(dust_AV)
        return modified_calzetti_dust_Alambda_per_AV(wave, dust_index=d_idx, bump_amp=b_amp, bump_dwave=b_w) * dust_AV
    else:
        return dust_Alambda_per_AV * dust_AV

def _process_pixel_data(ii, jj, star_particle_membership_list, gas_particle_membership_list):
    """
    ii=y jj=x
    Worker function to process calculations for a single pixel (ii, jj).
    This function will be executed in parallel.
    
    star_particle_membership_list: List of (original_particle_index, line_of_sight_distance) for THIS pixel.
    gas_particle_membership_list: List of (original_particle_index, line_of_sight_distance) for THIS pixel.
    """
    current_num_obs_wave_points = len(_worker_output_obs_wave_grid) if output_pixel_spectra_flag else 0

    pixel_results = {
        'map_stars_mass': 0.0, 'map_mw_age': 0.0, 'map_stars_mw_zsol': 0.0, 'map_sfr_100': 0.0,
        'map_sfr_30': 0.0, 'map_sfr_10': 0.0, 'map_gas_mass': 0.0, 'map_sfr_inst': 0.0,
        'map_gas_mw_zsol': 0.0, 'map_dust_mean_tauV': 0.0, 'map_dust_mean_AV': 0.0,
        'map_flux': np.zeros(len(_worker_filters)), 'map_flux_dust': np.zeros(len(_worker_filters)),
        'obs_spectra_nodust_igm': np.zeros(current_num_obs_wave_points),
        'obs_spectra_dust_igm': np.zeros(current_num_obs_wave_points),
        'map_lw_age_nodust': np.nan, 'map_lw_age_dust': np.nan, 'map_lw_zsol_nodust': np.nan, 'map_lw_zsol_dust': np.nan,
        'map_stars_mw_vel_los': np.nan, 'map_gas_mw_vel_los': np.nan, 'map_stars_vel_disp_los': np.nan,
        'map_gas_vel_disp_los': np.nan, 'map_lw_vel_los_nodust': np.nan, 'map_lw_vel_los_dust': np.nan, 'map_lw_vel_los_nebular': np.nan 
    }

    star_ids0 = np.asarray([x[0] for x in star_particle_membership_list], dtype=int)
    star_los_dist0 = np.asarray([x[1] for x in star_particle_membership_list])
    idx_valid_stars = np.where(~np.isnan(_worker_stars_mass[star_ids0]) & ~np.isnan(_worker_stars_age[star_ids0]) & ~np.isnan(_worker_stars_zmet[star_ids0]) & ~np.isnan(_worker_stars_vel_los_proj[star_ids0]))[0]
    star_ids, star_los_dist = star_ids0[idx_valid_stars], star_los_dist0[idx_valid_stars]

    gas_ids0 = np.asarray([x[0] for x in gas_particle_membership_list], dtype=int)
    gas_los_dist0 = np.asarray([x[1] for x in gas_particle_membership_list])
    idxg = np.where(~np.isnan(_worker_gas_mass[gas_ids0]) & ~np.isnan(_worker_gas_vel_los_proj[gas_ids0]))[0]
    gas_ids, gas_los_dist = gas_ids0[idxg], gas_los_dist0[idxg]

    m_sum = np.nansum(_worker_stars_mass[star_ids])
    pixel_results['map_stars_mass'] = m_sum
    if m_sum > 0:
        pixel_results['map_mw_age'] = np.nansum(_worker_stars_mass[star_ids] * _worker_stars_age[star_ids]) / m_sum
        pixel_results['map_stars_mw_zsol'] = np.nansum(_worker_stars_mass[star_ids] * _worker_stars_zmet[star_ids] / FSPS_Z_SUN) / m_sum
        pixel_results['map_stars_mw_vel_los'] = np.nansum(_worker_stars_mass[star_ids] * _worker_stars_vel_los_proj[star_ids]) / m_sum
        pixel_results['map_stars_vel_disp_los'] = np.sqrt(np.nansum(_worker_stars_mass[star_ids] * (_worker_stars_vel_los_proj[star_ids] - pixel_results['map_stars_mw_vel_los'])**2) / m_sum)

    pixel_results['map_sfr_100'] = np.nansum(_worker_stars_init_mass[star_ids[_worker_stars_age[star_ids] <= 0.1]]) / 1e8
    pixel_results['map_sfr_30'] = np.nansum(_worker_stars_init_mass[star_ids[_worker_stars_age[star_ids] <= 0.03]]) / 3e7
    pixel_results['map_sfr_10'] = np.nansum(_worker_stars_init_mass[star_ids[_worker_stars_age[star_ids] <= 0.01]]) / 1e7
    
    g_sum = np.nansum(_worker_gas_mass[gas_ids])
    pixel_results['map_gas_mass'], pixel_results['map_sfr_inst'] = g_sum, np.nansum(_worker_gas_sfr_inst[gas_ids])
    if g_sum > 0:
        pixel_results['map_gas_mw_zsol'] = np.nansum(_worker_gas_mass[gas_ids] * _worker_gas_zmet[gas_ids] / FSPS_Z_SUN) / g_sum
        pixel_results['map_gas_mw_vel_los'] = np.nansum(_worker_gas_mass[gas_ids] * _worker_gas_vel_los_proj[gas_ids]) / g_sum
        pixel_results['map_gas_vel_disp_los'] = np.sqrt(np.nansum(_worker_gas_mass[gas_ids] * (_worker_gas_vel_los_proj[gas_ids] - pixel_results['map_gas_mw_vel_los'])**2) / g_sum)

    # Effective AV calculation for sfr_AV method
    effective_av = 0.0
    if _worker_dust_method == 'sfr_AV' and func_interp_av_sfrden is not None:
        sfr_density = pixel_results['map_sfr_inst'] / pix_area_kpc2
        with np.errstate(divide='ignore'):
            effective_av = np.power(10.0, func_interp_av_sfrden(np.log10(sfr_density)))

    if len(star_ids) > 0:
        array_spec, array_spec_dust, array_AV, array_tauV, array_L_nodust, array_L_dust = [], [], [], [], [], []
        if output_pixel_spectra_flag: array_vel_los, array_L_nebular, array_vel_los_nebular_weighted = [], [], []
        
        lw_wave_idx = np.where((ssp_wave >= _lw_wave_min_rest) & (ssp_wave <= _lw_wave_max_rest))[0]

        for i_sid in range(len(star_ids)):
            star_id = star_ids[i_sid]
            if use_precomputed_ssp:
                particle_logzsol = np.log10(_worker_stars_zmet[star_id]/ssp_code_z_sun)
                pts = np.array([[ _worker_stars_age[star_id], particle_logzsol]])
                if ssp_interpolation_method in ['linear', 'cubic']:
                    s_cont = _global_ssp_stellar_continuum_interpolator(pts)[0]
                    n_em = _global_ssp_nebular_emission_interpolator(pts)[0]
                    s_mass = _global_ssp_stellar_mass_interpolator(pts)[0]
                else:
                    a_idx, z_idx = np.argmin(np.abs(ssp_ages_gyr - _worker_stars_age[star_id])), np.argmin(np.abs(ssp_logzsol_grid - particle_logzsol))
                    s_cont, n_em, s_mass = ssp_stellar_continuum_grid[a_idx, z_idx, :], ssp_nebular_emission_grid[a_idx, z_idx, :], ssp_stellar_mass_grid[a_idx, z_idx]
            else:
                # FSPS on-the-fly logic
                logzsol = np.log10(_worker_stars_zmet[star_id]/FSPS_Z_SUN)
                sp_instance.params["logzsol"], sp_instance.params['gas_logz'] = logzsol, logzsol
                sp_instance.params["add_neb_emission"] = 1
                _, s_tot = sp_instance.get_spectrum(peraa=True, tage=_worker_stars_age[star_id])
                sp_instance.params["add_neb_emission"] = 0
                _, s_cont = sp_instance.get_spectrum(peraa=True, tage=_worker_stars_age[star_id])
                n_em, s_mass = s_tot - s_cont, sp_instance.stellar_mass

            norm = _worker_stars_mass[star_id] / s_mass
            if output_pixel_spectra_flag:
                w_d_s, s_c_d = doppler_shift_spectrum(ssp_wave, s_cont, _worker_stars_vel_los_proj[star_id])
                s_c_i = interp1d(w_d_s, s_c_d, kind='linear', bounds_error=False, fill_value=0.0)(ssp_wave)
                g_v_neb, idxg_f = 0.0, np.where(gas_los_dist < star_los_dist[i_sid])[0]
                sf_g = gas_ids[idxg_f][_worker_gas_sfr_inst[gas_ids[idxg_f]] > 0.0]
                if sf_g.size > 0:
                    d3d = np.linalg.norm(_worker_stars_coords[star_id] - _worker_gas_coords[sf_g], axis=1)
                    sf_g_near = sf_g[d3d < _worker_max_dist_neb]
                    if sf_g_near.size > 0 and np.nansum(_worker_gas_mass[sf_g_near]) > 0:
                        g_v_neb = np.nansum(_worker_gas_mass[sf_g_near] * _worker_gas_vel_los_proj[sf_g_near]) / np.nansum(_worker_gas_mass[sf_g_near])
                w_d_n, n_e_d = doppler_shift_spectrum(ssp_wave, n_em, g_v_neb)
                n_e_i = interp1d(w_d_n, n_e_d, kind='linear', bounds_error=False, fill_value=0.0)(ssp_wave)
                spec = s_c_i + n_e_i
                array_vel_los.append(_worker_stars_vel_los_proj[star_id])
                array_vel_los_nebular_weighted.append(g_v_neb)
                array_L_nebular.append(simpson(n_e_i[lw_wave_idx]*norm, ssp_wave[lw_wave_idx]) if lw_wave_idx.size > 1 else 0.0)
            else:
                spec = s_cont + n_em

            spec_dust, d_AV = spec.copy(), 0.0

            # Dust Method Branching
            if _worker_dust_method == 'sfr_AV':
                d_AV = effective_av
            else:
                # Default LOS method
                idx_c = np.where((gas_los_dist < star_los_dist[i_sid]) & ((_worker_gas_sfr_inst[gas_ids] > 0.0) | (_worker_gas_log_temp[gas_ids] < 3.9)))[0]
                if idx_c.size > 0:
                    g_m_s = np.nansum(_worker_gas_mass[gas_ids[idx_c]])
                    if g_m_s > 0:
                        g_z = np.nansum(_worker_gas_mass[gas_ids[idx_c]]*_worker_gas_zmet[gas_ids[idx_c]]/FSPS_Z_SUN)/g_m_s
                        nH = np.nansum(_worker_gas_mass_H[gas_ids[idx_c]])*1.247914e+14/pix_area_kpc2
                        tauV = np.clip(_worker_scale_dust_tau * g_z * nH / 2.1e+21, 1e-10, None)
                        d_AV = -2.5*np.log10((1.0 - np.exp(-1.0*tauV))/tauV)
            
            if d_AV > 0:
                al = dust_reddening_diffuse_ism(d_AV, ssp_wave, dust_law)
                spec_dust *= 10.0**(-0.4*al)
                array_tauV.append(d_AV * 0.921); array_AV.append(d_AV)
            
            if _worker_stars_age[star_id] <= t_esc and dust_eta != 0.0:
                al_bc = unresolved_dust_birth_cloud_Alambda_per_AV(ssp_wave, dust_index_bc=dust_index_bc) * d_AV * dust_eta
                spec_dust *= 10.0**(-0.4*al_bc)

            array_spec.append(spec*norm)
            array_spec_dust.append(spec_dust*norm)
            array_L_nodust.append(simpson(spec[lw_wave_idx]*norm, ssp_wave[lw_wave_idx]) if lw_wave_idx.size > 1 else 0.0)
            array_L_dust.append(simpson(spec_dust[lw_wave_idx]*norm, ssp_wave[lw_wave_idx]) if lw_wave_idx.size > 1 else 0.0)
            
        pixel_results['map_dust_mean_AV'] = np.nanmean(array_AV) if array_AV else np.nan
        pixel_results['map_dust_mean_tauV'] = np.nanmean(array_tauV) if array_tauV else np.nan

        if array_spec:
            s_lum, s_lum_d = np.nansum(array_spec, axis=0), np.nansum(array_spec_dust, axis=0)
            w_o, f_o = cosmo_redshifting(ssp_wave, s_lum, snap_z, _worker_cosmo)
            _, f_o_d = cosmo_redshifting(ssp_wave, s_lum_d, snap_z, _worker_cosmo)
            f_o_i, f_o_d_i = f_o * igm_trans, f_o_d * igm_trans

            if output_pixel_spectra_flag and _worker_output_obs_wave_grid.size > 0:
                pixel_results['obs_spectra_nodust_igm'] = interp1d(w_o, f_o_i, kind='linear', bounds_error=False, fill_value=0.0)(_worker_output_obs_wave_grid)
                pixel_results['obs_spectra_dust_igm'] = interp1d(w_o, f_o_d_i, kind='linear', bounds_error=False, fill_value=0.0)(_worker_output_obs_wave_grid)

            pixel_results['map_flux'] = [filtering(w_o, f_o_i, _worker_filter_transmission[f]['wave'], _worker_filter_transmission[f]['trans']) for f in _worker_filters]
            pixel_results['map_flux_dust'] = [filtering(w_o, f_o_d_i, _worker_filter_transmission[f]['wave'], _worker_filter_transmission[f]['trans']) for f in _worker_filters]

            L_n, L_d = np.nansum(array_L_nodust), np.nansum(array_L_dust)
            if L_n > 0:
                pixel_results['map_lw_age_nodust'] = np.nansum(np.asarray(array_L_nodust) * _worker_stars_age[star_ids]) / L_n
                pixel_results['map_lw_zsol_nodust'] = np.nansum(np.asarray(array_L_nodust) * _worker_stars_zmet[star_ids] / FSPS_Z_SUN) / L_n
            if L_d > 0:
                pixel_results['map_lw_age_dust'] = np.nansum(np.asarray(array_L_dust) * _worker_stars_age[star_ids]) / L_d
                pixel_results['map_lw_zsol_dust'] = np.nansum(np.asarray(array_L_dust) * _worker_stars_zmet[star_ids] / FSPS_Z_SUN) / L_d
            
            if output_pixel_spectra_flag:
                v_los = np.asarray(array_vel_los)
                arr_L_nodust = np.asarray(array_L_nodust)
                arr_L_dust = np.asarray(array_L_dust)
                arr_L_nebular = np.asarray(array_L_nebular) if array_L_nebular else np.asarray([])
                arr_v_neb_weight = np.asarray(array_vel_los_nebular_weighted) if array_vel_los_nebular_weighted else np.asarray([])

                if L_n > 0: pixel_results['map_lw_vel_los_nodust'] = np.nansum(arr_L_nodust * v_los) / L_n
                if L_d > 0: pixel_results['map_lw_vel_los_dust'] = np.nansum(arr_L_dust * v_los) / L_d
                if arr_L_nebular.size > 0 and np.nansum(arr_L_nebular) > 0:
                    pixel_results['map_lw_vel_los_nebular'] = np.nansum(arr_L_nebular * arr_v_neb_weight) / np.nansum(arr_L_nebular)

    return ii, jj, pixel_results

def generate_images(sim_file, z, filters, filter_transmission_path, dim_kpc=None, smoothing_length=0.15, 
                    pix_arcsec=0.1, pix_kpc=None, flux_unit='MJy/sr', polar_angle_deg=0, azimuth_angle_deg=0,
                    name_out_img=None, n_jobs=-1, ssp_code='FSPS', imf_type=1, imf_upper_limit=120.0, 
                    imf_lower_limit=0.08, imf1=1.3, imf2=2.3, imf3=2.3, vdmc=0.08, mdave=0.5, gas_logu=-2.0,
                    igm_type=0, dust_index_bc=-0.7, dust_index=0.0, t_esc=0.01, dust_eta=1.0, 
                    scale_dust_redshift="Vogelsberger20", cosmo_str='Planck18', dust_method='los', 
                    av_sfrden_relation=None, dust_law=0, bump_amp=0.85, bump_dwave=0.035, salim_a0=-4.30, 
                    salim_a1=2.71, salim_a2= -0.191, salim_a3=0.0121, salim_RV=3.15, salim_B=3.15, 
                    initdim_kpc=200, initdim_mass_fraction=0.99, use_precomputed_ssp=True, ssp_filepath=None, 
                    ssp_interpolation_method='nearest', output_pixel_spectra=False, rest_wave_min=1000.0, 
                    rest_wave_max=30000.0, rest_delta_wave=5.0, max_dist_neb=0.5):
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
        smoothing_length (float, optional): Smoothing length of the simulation in kpc. Defaults to 0.15.
        pix_arcsec (float, optional): Pixel size in arcseconds. Defaults to 0.1. If pix_arcsec is provided, pix_kpc input will be ignored but converted from pix_arcsec (given z) instead.
        pix_kpc (float, optional): Pixel size in physical unit kpc. Defaults to None.
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

        dust_method (str, optional): The framework used to calculate diffuse ISM dust attenuation. Options: 
                                        
                                        - 'los': (Default) Integration of the optical depth along the line-of-sight for each star particle based on local gas column density and metallicity. 
                                        
                                        - 'sfr_AV': Calculates an effective A_V for the entire pixel/grid cell based on the instantaneous SFR surface density (Msun/yr/kpc^2).

        av_sfrden_relation (dict, optional): A dictionary defining the proportionality between SFR surface density and A_V in logarithmic scale. Required if dust_method='sfr_AV'. Expected format: {'log_AV': [values], 'log_SFR_density': [values]}. The units for 'SFR_density' should be Msun/yr/kpc^2. In the simulation, the SFR density is calculated with 100 Myr timescale SFR. Defaults to None.
        dust_law (int, optional): Dust attenuation law type. Defaults to 0.
        bump_amp (float, optional): UV bump amplitude. Defaults to 0.85.
        bump_dwave (float, optional): Width (FWHM) of the dust attenuation bump in units of micron, as parameterized with the Drude profile. Defaults to 0.035 micron.
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
        max_dist_neb (float, optional): Max distance (kpc) to search for gas particles for nebular Doppler shifting. Default to 0.5.
    """ 
    cosmo = define_cosmo(cosmo_str)
    print ('Processing '+sim_file)
    f_t_g, f_w_p_g = _load_filter_transmission_from_paths(filters, filter_transmission_path)
    snap_z, snap_a, snap_univ_age = z, 1.0/(1.0 + z), cosmo.age(z).value

    if pix_arcsec is not None:
        pix_kpc = angular_to_physical(snap_z, pix_arcsec, cosmo)
    else:
        if pix_kpc is None: 
            print ('Both pix_arcsec and pix_kpc cannot be None!')
            sys.exit()
        else:
            pix_arcsec = physical_to_angular(snap_z, pix_kpc, cosmo)  

    # Initial gridding based on smoothing length
    working_pix_kpc = smoothing_length
    pix_area_kpc2_working = working_pix_kpc**2
    rebin_factor = pix_kpc / smoothing_length

    with h5py.File(sim_file,'r') as f:
        s_im, s_fz, s_m, s_z, s_c, s_v = f['star/init_mass'][:], f['star/form_z'][:], f['star/mass'][:], f['star/zmet'][:], f['star/coords'][:], f['star/vel'][:]
        s_age = snap_univ_age - interp_age_univ_from_z(s_fz, cosmo)
        idx = np.where(s_age>=0)[0]
        s_im, s_m, s_z, s_age, s_c, s_v = s_im[idx], s_m[idx], s_z[idx], s_age[idx], s_c[idx,:], s_v[idx,:]
        g_m, g_z, g_sfr, g_lt, g_c, g_v, g_mh = f['gas/mass'][:], f['gas/zmet'][:], f['gas/sfr_inst'][:], np.log10(f['gas/temp'][:]), f['gas/coords'][:], f['gas/vel'][:], f['gas/mass_H'][:]

    if dim_kpc is None: 
        dim_kpc = determine_image_size(s_c, s_m, working_pix_kpc, (initdim_kpc, initdim_kpc), 
                                       polar_angle_deg, azimuth_angle_deg, g_c, g_m, 
                                       mass_percentage=initdim_mass_fraction, max_img_dim=initdim_kpc)
    
    # Grid info for high-res working grid
    s_mem, _, _, g_inf, g_mem, _, s_v_los, g_v_los = get_2d_density_projection_no_los_binning(s_c, s_m, working_pix_kpc, 
                                                                                              (dim_kpc, dim_kpc), 
                                                                                              polar_angle_deg=polar_angle_deg, 
                                                                                              azimuth_angle_deg=azimuth_angle_deg, 
                                                                                              gas_coords=g_c, gas_masses=g_m, 
                                                                                              star_vels=s_v, gas_vels=g_v)
    dimx, dimy = g_inf['num_pixels_x'], g_inf['num_pixels_y']
    #print ('Cutout size: %d x %d pix or %d x %d kpc' % (dimx,dimy,dim_kpc,dim_kpc))

    # Dust normalization loading
    if isinstance(scale_dust_redshift, str):
        data = np.loadtxt(str(importlib.resources.files('galsyn.data').joinpath("Vogelsberger20_scale_dust.txt")))
        n_d_z, n_d_t = data[:,0], data[:,1]
    else: n_d_z, n_d_t = np.asarray(scale_dust_redshift["z"]), np.asarray(scale_dust_redshift["tau_dust"])
    s_d_t = tau_dust_given_z(snap_z, n_d_z, n_d_t)

    # Define the fixed observed-frame wavelength grid for output spectra
    f_g_o_w = np.arange(rest_wave_min*(1+z), (rest_wave_max+rest_delta_wave)*(1+z), rest_delta_wave*(1+z)) if output_pixel_spectra else np.array([])
    num_w = f_g_o_w.size

    # Allocation for working grid maps
    w_map_mw_age, w_map_stars_mw_zsol, w_map_stars_mass = np.zeros((dimy,dimx)), np.zeros((dimy,dimx)), np.zeros((dimy,dimx))
    w_map_sfr_100, w_map_sfr_30, w_map_sfr_10 = np.zeros((dimy,dimx)), np.zeros((dimy,dimx)), np.zeros((dimy,dimx))
    w_map_gas_mw_zsol, w_map_gas_mass, w_map_sfr_inst = np.zeros((dimy,dimx)), np.zeros((dimy,dimx)), np.zeros((dimy,dimx))
    w_map_dust_mean_tauV, w_map_dust_mean_AV = np.zeros((dimy,dimx)), np.zeros((dimy,dimx))
    w_map_flux, w_map_flux_dust = np.zeros((dimy,dimx,len(filters))), np.zeros((dimy,dimx,len(filters)))
    w_map_spec_n, w_map_spec_d = (np.zeros((dimy, dimx, num_w)), np.zeros((dimy, dimx, num_w))) if output_pixel_spectra else (None, None)
    w_map_lw = {k: np.full((dimy, dimx), np.nan) for k in ['age_n', 'age_d', 'z_n', 'z_d', 'v_n', 'v_d', 'v_neb', 'v_s_mw', 'v_g_mw', 'v_s_disp', 'v_g_disp']}

    tasks = [(ii, jj, s_mem[ii][jj], g_mem[ii][jj]) for ii in range(dimy) for jj in range(dimx)]
    tasks.sort(key=lambda x: len(x[2]) + len(x[3]), reverse=True)

    with tqdm_joblib(total=len(tasks), desc="Processing pixels") as pb:
        results = Parallel(n_jobs=n_jobs, initializer=init_worker, 
                           initargs=(ssp_code, snap_z, pix_area_kpc2_working, 
                                     filters, filter_transmission_path, 
                                     imf_type, imf_upper_limit, imf_lower_limit, 
                                     imf1, imf2, imf3, vdmc, mdave, gas_logu, 
                                     igm_type, dust_index_bc, dust_index, 
                                     t_esc, dust_eta, s_d_t, cosmo_str, 
                                     dust_law, bump_amp, bump_dwave, 
                                     salim_a0, salim_a1, salim_a2, salim_a3, 
                                     salim_RV, salim_B, use_precomputed_ssp, 
                                     s_m, s_age, s_z, s_im, s_v_los, s_c, g_m, 
                                     g_sfr, g_z, g_lt, g_mh, g_v_los, g_c, 
                                     ssp_filepath, ssp_interpolation_method, 
                                     output_pixel_spectra, f_g_o_w.tolist(),
                                     dust_method, av_sfrden_relation,
                                     max_dist_neb))(delayed(_process_pixel_data)(*t) for t in tasks)

    for ii, jj, pd in results:
        w_map_stars_mass[ii,jj], w_map_mw_age[ii,jj], w_map_stars_mw_zsol[ii,jj] = pd['map_stars_mass'], pd['map_mw_age'], pd['map_stars_mw_zsol']
        w_map_sfr_100[ii,jj], w_map_sfr_30[ii,jj], w_map_sfr_10[ii,jj] = pd['map_sfr_100'], pd['map_sfr_30'], pd['map_sfr_10']
        w_map_gas_mass[ii,jj], w_map_sfr_inst[ii,jj], w_map_gas_mw_zsol[ii,jj] = pd['map_gas_mass'], pd['map_sfr_inst'], pd['map_gas_mw_zsol']
        w_map_dust_mean_tauV[ii,jj], w_map_dust_mean_AV[ii,jj] = pd['map_dust_mean_tauV'], pd['map_dust_mean_AV']
        w_map_flux[ii,jj], w_map_flux_dust[ii,jj] = pd['map_flux'], pd['map_flux_dust']
        if output_pixel_spectra: 
            w_map_spec_n[ii,jj], w_map_spec_d[ii,jj] = pd['obs_spectra_nodust_igm'], pd['obs_spectra_dust_igm']
        w_map_lw['age_n'][ii,jj], w_map_lw['age_d'][ii,jj], w_map_lw['z_n'][ii,jj], w_map_lw['z_d'][ii,jj] = pd['map_lw_age_nodust'], pd['map_lw_age_dust'], pd['map_lw_zsol_nodust'], pd['map_lw_zsol_dust']
        w_map_lw['v_s_mw'][ii,jj], w_map_lw['v_g_mw'][ii,jj], w_map_lw['v_s_disp'][ii,jj], w_map_lw['v_g_disp'][ii,jj] = pd['map_stars_mw_vel_los'], pd['map_gas_mw_vel_los'], pd['map_stars_vel_disp_los'], pd['map_gas_vel_disp_los']
        w_map_lw['v_n'][ii,jj], w_map_lw['v_d'][ii,jj], w_map_lw['v_neb'][ii,jj] = pd['map_lw_vel_los_nodust'], pd['map_lw_vel_los_dust'], pd['map_lw_vel_los_nebular']


    # FLUX CONSERVING REBINNING TO USER PIXEL SIZE
    # The raw values in w_map_flux are in erg/s/cm2/A (linear flux density).
    # To conserve flux, we sum the sub-pixels within each target pixel.
    map_flux_summed = rebin_map(w_map_flux, rebin_factor, mode='sum')
    map_flux_dust_summed = rebin_map(w_map_flux_dust, rebin_factor, mode='sum')
    
    # Rebin properties using mean (physical quantities like age/metallicity)
    map_stars_mass = rebin_map(w_map_stars_mass, rebin_factor, mode='sum')
    map_mw_age = rebin_map(w_map_mw_age, rebin_factor, mode='mean')
    map_stars_mw_zsol = rebin_map(w_map_stars_mw_zsol, rebin_factor, mode='mean')
    map_sfr_100 = rebin_map(w_map_sfr_100, rebin_factor, mode='sum')
    map_sfr_30 = rebin_map(w_map_sfr_30, rebin_factor, mode='sum')
    map_sfr_10 = rebin_map(w_map_sfr_10, rebin_factor, mode='sum')
    map_gas_mass = rebin_map(w_map_gas_mass, rebin_factor, mode='sum')
    map_sfr_inst = rebin_map(w_map_sfr_inst, rebin_factor, mode='sum')
    map_gas_mw_zsol = rebin_map(w_map_gas_mw_zsol, rebin_factor, mode='mean')
    map_dust_mean_tauV = rebin_map(w_map_dust_mean_tauV, rebin_factor, mode='mean')
    map_dust_mean_AV = rebin_map(w_map_dust_mean_AV, rebin_factor, mode='mean')

    map_lw_final = {k: rebin_map(v, rebin_factor, mode='mean') for k, v in w_map_lw.items()}
    
    if output_pixel_spectra:
        # Sum the spectra cubes to preserve total flux per pixel
        map_spec_nodust = rebin_map(w_map_spec_n, rebin_factor, mode='sum')
        map_spec_dust = rebin_map(w_map_spec_d, rebin_factor, mode='sum')

    # FINAL UNIT CONVERSION
    # Now convert the summed erg/s/cm2/A into the user's requested unit.
    # We use the FINAL pix_arcsec to handle MJy/sr correctly.
    map_flux = np.zeros((map_flux_summed.shape[0], map_flux_summed.shape[1], len(filters)))
    map_flux_dust = np.zeros_like(map_flux)

    for i in range(len(filters)): 
        map_flux[:,:,i] = convert_flux_map(map_flux_summed[:,:,i], f_w_p_g[filters[i]], to_unit=flux_unit, pixel_scale_arcsec=pix_arcsec)
        map_flux_dust[:,:,i] = convert_flux_map(map_flux_dust_summed[:,:,i], f_w_p_g[filters[i]], to_unit=flux_unit, pixel_scale_arcsec=pix_arcsec)

    # SAVE TO FITS
    if name_out_img is not None:
        try:
            hdul = fits.HDUList()
            final_dimy, final_dimx = map_flux.shape[0], map_flux.shape[1]

            # 1. Primary HDU & Multi-band Photometry
            if map_flux.shape[2] > 0:
                primary_data = map_flux[:, :, 0]
                prihdr = fits.Header()
                prihdr['COMMENT'] = 'Primary Image: First band (no dust)'
                prihdr['CRPIX1'] = final_dimx / 2.0 + 0.5
                prihdr['CRPIX2'] = final_dimy / 2.0 + 0.5
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
                prihdr['SSP_CODE'] = ssp_code
                prihdr['SM_LEN'] = smoothing_length
                prihdr['DUST_MET'] = dust_method
                primary_hdu = fits.PrimaryHDU(data=primary_data, header=prihdr)
                hdul.append(primary_hdu)

                for i_band in range(len(filters)):
                    ext_hdr = fits.Header()
                    ext_hdr['EXTNAME'] = 'NODUST_'+filters[i_band].upper() 
                    ext_hdr['FILTER'] = filters[i_band] 
                    ext_hdr['COMMENT'] = f'Flux for filter: {filters[i_band]}' 
                    ext_hdr['BUNIT'] = flux_unit
                    hdul.append(fits.ImageHDU(data=map_flux[:, :, i_band], header=ext_hdr))

                for i_band in range(len(filters)):
                    ext_hdr = fits.Header()
                    ext_hdr['EXTNAME'] = 'DUST_'+filters[i_band].upper() 
                    ext_hdr['FILTER'] = filters[i_band] 
                    ext_hdr['COMMENT'] = f'Flux (with dust) for filter: {filters[i_band]}' 
                    ext_hdr['BUNIT'] = flux_unit
                    hdul.append(fits.ImageHDU(data=map_flux_dust[:, :, i_band], header=ext_hdr))
            else:
                hdul.append(fits.PrimaryHDU())

            # 2. Property Maps (Mass, Age, Z, Velocity)
            # Map the rebinned data to their original naming convention
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
                'LW_AGE_NODUST': map_lw_final['age_n'],
                'LW_AGE_DUST': map_lw_final['age_d'],
                'LW_ZSOL_NODUST': map_lw_final['z_n'],
                'LW_ZSOL_DUST': map_lw_final['z_d'],
                'STARS_MW_VEL_LOS': map_lw_final['v_s_mw'],
                'GAS_MW_VEL_LOS': map_lw_final['v_g_mw'],
                'STARS_VEL_DISP_LOS': map_lw_final['v_s_disp'],
                'GAS_VEL_DISP_LOS': map_lw_final['v_g_disp'],
                'LW_VEL_LOS_NODUST': map_lw_final['v_n'],
                'LW_VEL_LOS_DUST': map_lw_final['v_d'],
                'LW_VEL_LOS_NEBULAR': map_lw_final['v_neb']
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
                    elif 'MASS' in map_name: 
                        ext_hdr['BUNIT'] = 'Msun'
                    elif 'SFR' in map_name: 
                        ext_hdr['BUNIT'] = 'Msun/yr'
                    
                    hdul.append(fits.ImageHDU(data=data_array, header=ext_hdr))

            # 3. Spectral Cubes
            if output_pixel_spectra:
                # Transpose the rebinned data cube from (y, x, wave) to (wave, y, x)
                trans_nodust = map_spec_nodust.transpose((2, 0, 1))
                trans_dust = map_spec_dust.transpose((2, 0, 1))

                for ext_name, data_cube in [('OBS_SPEC_NODUST', trans_nodust), 
                                            ('OBS_SPEC_DUST', trans_dust)]:
                    h = fits.Header()
                    h['EXTNAME'] = ext_name
                    h['COMMENT'] = 'Observed-frame spectra'
                    h['CRPIX1'] = 1.0
                    h['CRVAL1'] = f_g_o_w[0] if f_g_o_w.size > 0 else 0.0
                    h['CDELT1'] = (f_g_o_w[1] - f_g_o_w[0]) if f_g_o_w.size > 1 else 0.0
                    h['CUNIT1'] = 'Angstrom'
                    h['CRPIX2'] = final_dimy / 2.0 + 0.5
                    h['CDELT2'] = pix_kpc
                    h['CUNIT2'] = 'kpc'
                    h['CRPIX3'] = final_dimx / 2.0 + 0.5
                    h['CDELT3'] = pix_kpc
                    h['CUNIT3'] = 'kpc'
                    h['BUNIT'] = 'erg/s/cm2/Angstrom'
                    hdul.append(fits.ImageHDU(data=data_cube, header=h))
                
                # Wavelength Binary Table
                if f_g_o_w.size > 0:
                    col = fits.Column(name='WAVELENGTH', format='D', array=f_g_o_w)
                    hdul.append(fits.BinTableHDU.from_columns([col], name='WAVELENGTH_GRID'))

            # Write to file
            output_dir = os.path.dirname(name_out_img)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)

            hdul.writeto(name_out_img, overwrite=True, output_verify='fix')
            print(f"Galaxy image synthesis completed successfully and results saved to FITS file: {name_out_img}")

        except Exception as e:
            print(f"Error saving FITS file {name_out_img}: {e}")