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
import importlib.resources
from scipy.ndimage import zoom

import bagpipes as pipes

# Constants for solar metallicity
BAGPIPES_Z_SUN = 0.02
L_SUN_ERG_S = 3.828e33 # Solar luminosity in erg/s

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

# Global variables for Bagpipes instance components
_ssp_worker_bagpipes_model_components = None

# Other global worker variables
igm_trans = None
snap_z = None
pix_area_kpc2 = None
gas_logu = None
igm_type = None
dust_index_bc = None
t_esc = None
dust_eta = None
dust_law = None

# New Global variables for Dust Frameworks
_worker_dust_method = 'los' 
func_interp_av_sfrden = None

# Unified Dust Model Globals for Option 0
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
_worker_cosmo = None

# Global variables for particle data
_worker_stars_mass = _worker_stars_age = _worker_stars_zmet = _worker_stars_init_mass = _worker_stars_vel_los_proj = _worker_stars_coords = None
_worker_gas_mass = _worker_gas_sfr_inst = _worker_gas_zmet = _worker_gas_log_temp = _worker_gas_mass_H = _worker_gas_vel_los_proj = _worker_gas_coords = None

_child_wave_min_rest = 1000.0 
_child_wave_max_rest = 30000.0 

def rebin_map(data, factor, mode='sum'):
    """
    Rebins a 2D or 3D array with float factors and robust NaN handling.
    Updated to use grid_mode=True for astronomical pixel area conservation.
    
    'sum' preserves total flux/mass.
    'mean' preserves average property values (Age, Temperature).
    """
    if factor == 1.0:
        return data

    zoom_rate = 1.0 / factor
    # Handle spatial scaling (assumes Y, X are first two dims)
    zoom_factors = (zoom_rate, zoom_rate, 1) if data.ndim == 3 else (zoom_rate, zoom_rate)

    # Use order=1 (bilinear) as it is the most stable for physical data.
    # grid_mode=True treats pixels as having area, which is correct for fluxes and masses.
    # mode='grid-constant' ensures proper boundary handling without pixel shrinkage.
    rescaled = zoom(data, zoom_factors, order=1, prefilter=False, 
                    grid_mode=True, mode='grid-constant')

    if mode == 'sum':
        # Apply the area-scaling factor
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

    elif mode == 'mean':
        # Enforce property conservation
        # Ensures the average of the new map matches the average of the old map
        if data.ndim == 3:
            mean_in = np.nanmean(data, axis=(0, 1))
            mean_out = np.nanmean(rescaled, axis=(0, 1))
            nonzero = (mean_out != 0) & (~np.isnan(mean_out))
            rescaled[..., nonzero] *= (mean_in[nonzero] / mean_out[nonzero])
        else:
            mean_in = np.nanmean(data)
            mean_out = np.nanmean(rescaled)
            if mean_out != 0 and not np.isnan(mean_out):
                rescaled *= (mean_in / mean_out)

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
                gas_logu_val, igm_type_val, dust_index_bc_val, 
                dust_index_val, t_esc_val, dust_eta_val, precomputed_scale_dust_tau_val,
                cosmo_str_val, dust_law_val, bump_amp_val, bump_dwave_val, salim_a0_val, 
                salim_a1_val, salim_a2_val, salim_a3_val, salim_RV_val, salim_B_val, use_precomputed_ssp_val, 
                stars_mass_arr, stars_age_arr, stars_zmet_arr, stars_init_mass_arr, stars_vel_los_proj_arr, stars_coords_arr,
                gas_mass_arr, gas_sfr_inst_arr, gas_zmet_arr, gas_log_temp_arr, gas_mass_H_arr, gas_vel_los_proj_arr, gas_coords_arr, 
                ssp_filepath_val=None, ssp_interpolation_method_val='nearest', 
                output_pixel_spectra_val=False, output_obs_wave_grid_val=None,
                dust_method_val='los', av_sfrden_relation_val=None):
    
    global ssp_wave, ssp_ages_gyr, ssp_logzsol_grid, ssp_stellar_mass_grid, ssp_code_z_sun
    global ssp_stellar_continuum_grid, ssp_nebular_emission_grid 
    global _global_ssp_stellar_continuum_interpolator, _global_ssp_nebular_emission_interpolator, _global_ssp_stellar_mass_interpolator
    global _ssp_worker_bagpipes_model_components, igm_trans, snap_z, pix_area_kpc2
    global gas_logu, igm_type, dust_index_bc, t_esc, dust_eta, dust_law
    global salim_a0, salim_a1, salim_a2, salim_a3, salim_RV, salim_B, dust_Alambda_per_AV
    global func_interp_dust_index, func_interp_bump_amp, func_interp_bump_dwave
    global use_precomputed_ssp, ssp_interpolation_method, output_pixel_spectra_flag, _worker_output_obs_wave_grid 
    global _worker_filters, _worker_filter_transmission, _worker_filter_wave_eff, _worker_cosmo
    global _worker_scale_dust_tau, _worker_stars_mass, _worker_stars_age, _worker_stars_zmet, _worker_stars_init_mass, _worker_stars_vel_los_proj, _worker_stars_coords
    global _worker_gas_mass, _worker_gas_sfr_inst, _worker_gas_zmet, _worker_gas_log_temp, _worker_gas_mass_H, _worker_gas_vel_los_proj, _worker_gas_coords
    global _worker_dust_method, func_interp_av_sfrden

    _worker_stars_mass, _worker_stars_age, _worker_stars_zmet = stars_mass_arr, stars_age_arr, stars_zmet_arr
    _worker_stars_init_mass, _worker_stars_vel_los_proj, _worker_stars_coords = stars_init_mass_arr, stars_vel_los_proj_arr, stars_coords_arr
    _worker_gas_mass, _worker_gas_sfr_inst, _worker_gas_zmet = gas_mass_arr, gas_sfr_inst_arr, gas_zmet_arr
    _worker_gas_log_temp, _worker_gas_mass_H, _worker_gas_vel_los_proj, _worker_gas_coords = gas_log_temp_arr, gas_mass_H_arr, gas_vel_los_proj_arr, gas_coords_arr

    snap_z, pix_area_kpc2 = snap_z_val, pix_area_kpc2_val
    gas_logu, igm_type = gas_logu_val, igm_type_val
    dust_index_bc, t_esc, dust_eta, _worker_scale_dust_tau = dust_index_bc_val, t_esc_val, dust_eta_val, precomputed_scale_dust_tau_val
    _worker_cosmo = define_cosmo(cosmo_str_val)
    dust_law = dust_law_val
    salim_a0, salim_a1, salim_a2, salim_a3, salim_RV, salim_B = salim_a0_val, salim_a1_val, salim_a2_val, salim_a3_val, salim_RV_val, salim_B_val
    use_precomputed_ssp, ssp_interpolation_method = use_precomputed_ssp_val, ssp_interpolation_method_val 
    _worker_filters = filters_list_val
    _worker_filter_transmission, _worker_filter_wave_eff = _load_filter_transmission_from_paths(_worker_filters, filter_transmission_path_val)
    output_pixel_spectra_flag = output_pixel_spectra_val
    _worker_output_obs_wave_grid = np.asarray(output_obs_wave_grid_val)

    # Dust Framework Setup
    _worker_dust_method = dust_method_val
    if _worker_dust_method == 'sfr_AV' and isinstance(av_sfrden_relation_val, dict):
        func_interp_av_sfrden = interp1d(av_sfrden_relation_val['SFR_density'], 
                                         av_sfrden_relation_val['AV'], 
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
        # Generate on-the-fly using Bagpipes
        dust = {"type": "Calzetti", "Av": 0.0, "eta": 1.0}
        nebular = {"logU": gas_logu_val}
        _ssp_worker_bagpipes_model_components = {"redshift": 0.0, "veldisp": 0, "dust": dust, "nebular": nebular, "sfh": "delta"}
        dummy_model = pipes.model_galaxy({"burst": {"age": 0.01, "massformed": 1.0, "metallicity": 1.0}, **_ssp_worker_bagpipes_model_components}, spec_wavs=np.arange(100., 30000., 5.))
        ssp_wave = dummy_model.wavelengths
        ssp_code_z_sun = BAGPIPES_Z_SUN

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
    elif dust_law == 1: dust_Alambda_per_AV = salim18_dust_Alambda_per_AV(ssp_wave, salim_a0, salim_a1, salim_a2, salim_a3, salim_B, salim_RV)
    elif dust_law == 2: dust_Alambda_per_AV = calzetti_dust_Alambda_per_AV(ssp_wave)
    elif dust_law == 3: dust_Alambda_per_AV = smc_gordon2003_dust_Alambda_per_AV(ssp_wave)
    elif dust_law == 4: dust_Alambda_per_AV = lmc_gordon2003_dust_Alambda_per_AV(ssp_wave)
    elif dust_law == 5: dust_Alambda_per_AV = ccm89_dust_Alambda_per_AV(ssp_wave)
    elif dust_law == 6: dust_Alambda_per_AV = fitzpatrick99_dust_Alambda_per_AV(ssp_wave)

    if igm_type == 0: igm_trans = igm_att_madau(ssp_wave * (1.0+snap_z), snap_z)
    else: igm_trans = igm_att_inoue(ssp_wave * (1.0+snap_z), snap_z)

def dust_reddening_diffuse_ism(dust_AV, wave, dust_law):
    if dust_law == 0:
        d_idx, b_amp, b_w = func_interp_dust_index(dust_AV), func_interp_bump_amp(dust_AV), func_interp_bump_dwave(dust_AV)
        return modified_calzetti_dust_Alambda_per_AV(wave, dust_index=d_idx, bump_amp=b_amp, bump_dwave=b_w) * dust_AV
    else:
        return dust_Alambda_per_AV * dust_AV
    
def _process_pixel_data(ii, jj, star_particle_membership_list, gas_particle_membership_list):
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
    idx_valid_stars = np.where(~np.isnan(_worker_stars_mass[star_ids0]) & ~np.isnan(_worker_stars_age[star_ids0]) & ~np.isnan(_worker_stars_zmet[star_ids0]))[0]
    star_ids, star_los_dist = star_ids0[idx_valid_stars], star_los_dist0[idx_valid_stars]

    gas_ids0 = np.asarray([x[0] for x in gas_particle_membership_list], dtype=int)
    gas_los_dist0 = np.asarray([x[1] for x in gas_particle_membership_list])
    idxg = np.where(~np.isnan(_worker_gas_mass[gas_ids0]))[0]
    gas_ids, gas_los_dist = gas_ids0[idxg], gas_los_dist0[idxg]

    m_sum = np.nansum(_worker_stars_mass[star_ids])
    pixel_results['map_stars_mass'] = m_sum
    if m_sum > 0:
        pixel_results['map_mw_age'] = np.nansum(_worker_stars_mass[star_ids] * _worker_stars_age[star_ids]) / m_sum
        pixel_results['map_stars_mw_zsol'] = np.nansum(_worker_stars_mass[star_ids] * _worker_stars_zmet[star_ids] / BAGPIPES_Z_SUN) / m_sum
        pixel_results['map_stars_mw_vel_los'] = np.nansum(_worker_stars_mass[star_ids] * _worker_stars_vel_los_proj[star_ids]) / m_sum
        pixel_results['map_stars_vel_disp_los'] = np.sqrt(np.nansum(_worker_stars_mass[star_ids] * (_worker_stars_vel_los_proj[star_ids] - pixel_results['map_stars_mw_vel_los'])**2) / m_sum)

    pixel_results['map_sfr_100'] = np.nansum(_worker_stars_init_mass[star_ids[_worker_stars_age[star_ids] <= 0.1]]) / 1e8
    pixel_results['map_sfr_30'] = np.nansum(_worker_stars_init_mass[star_ids[_worker_stars_age[star_ids] <= 0.03]]) / 3e7
    pixel_results['map_sfr_10'] = np.nansum(_worker_stars_init_mass[star_ids[_worker_stars_age[star_ids] <= 0.01]]) / 1e7
    
    g_sum = np.nansum(_worker_gas_mass[gas_ids])
    pixel_results['map_gas_mass'], pixel_results['map_sfr_inst'] = g_sum, np.nansum(_worker_gas_sfr_inst[gas_ids])
    if g_sum > 0:
        pixel_results['map_gas_mw_zsol'] = np.nansum(_worker_gas_mass[gas_ids] * _worker_gas_zmet[gas_ids] / BAGPIPES_Z_SUN) / g_sum
        pixel_results['map_gas_mw_vel_los'] = np.nansum(_worker_gas_mass[gas_ids] * _worker_gas_vel_los_proj[gas_ids]) / g_sum
        pixel_results['map_gas_vel_disp_los'] = np.sqrt(np.nansum(_worker_gas_mass[gas_ids] * (_worker_gas_vel_los_proj[gas_ids] - pixel_results['map_gas_mw_vel_los'])**2) / g_sum)

    # Effective AV calculation for sfr_AV method
    effective_av = 0.0
    if _worker_dust_method == 'sfr_AV' and func_interp_av_sfrden is not None:
        sfr_density = pixel_results['map_sfr_inst'] / pix_area_kpc2
        effective_av = float(func_interp_av_sfrden(sfr_density))

    if len(star_ids) > 0:
        array_spec, array_spec_dust, array_AV, array_tauV, array_L_nodust, array_L_dust = [], [], [], [], [], []
        if output_pixel_spectra_flag: array_vel_los, array_L_nebular, array_vel_los_nebular_weighted = [], [], []
        lw_wave_idx = np.where((ssp_wave >= _child_wave_min_rest) & (ssp_wave <= _child_wave_max_rest))[0]

        for i_sid in range(len(star_ids)):
            star_id = star_ids[i_sid]
            if use_precomputed_ssp:
                particle_logzsol = np.log10(_worker_stars_zmet[star_id]/ssp_code_z_sun)
                pts = np.array([[ _worker_stars_age[star_id], particle_logzsol]])
                if ssp_interpolation_method in ['linear', 'cubic']:
                    s_cont, n_em, s_mass = _global_ssp_stellar_continuum_interpolator(pts)[0], _global_ssp_nebular_emission_interpolator(pts)[0], _global_ssp_stellar_mass_interpolator(pts)[0]
                else:
                    a_idx, z_idx = np.argmin(np.abs(ssp_ages_gyr - _worker_stars_age[star_id])), np.argmin(np.abs(ssp_logzsol_grid - particle_logzsol))
                    s_cont, n_em, s_mass = ssp_stellar_continuum_grid[a_idx, z_idx, :], ssp_nebular_emission_grid[a_idx, z_idx, :], ssp_stellar_mass_grid[a_idx, z_idx]
            else:
                metallicity_z_zsun = _worker_stars_zmet[star_id] / BAGPIPES_Z_SUN
                burst = {"age": _worker_stars_age[star_id], "massformed": 1.0, "metallicity": metallicity_z_zsun}
                m_total = pipes.model_galaxy({**_ssp_worker_bagpipes_model_components, "burst": burst, "nebular": {"logU": gas_logu}}, spec_wavs=ssp_wave)
                s_tot = m_total.spectrum_full / L_SUN_ERG_S
                m_stellar = pipes.model_galaxy({**_ssp_worker_bagpipes_model_components, "burst": burst, "nebular": None}, spec_wavs=ssp_wave)
                s_cont = m_stellar.spectrum_full / L_SUN_ERG_S
                n_em, s_mass = s_tot - s_cont, m_total.sfh.stellar_mass

            norm = _worker_stars_mass[star_id] / s_mass

            if output_pixel_spectra_flag:
                w_d_s, s_c_d = doppler_shift_spectrum(ssp_wave, s_cont, _worker_stars_vel_los_proj[star_id])
                s_c_i = interp1d(w_d_s, s_c_d, kind='linear', bounds_error=False, fill_value=0.0)(ssp_wave)
                g_v_neb, idxg_f = 0.0, np.where(gas_los_dist < star_los_dist[i_sid])[0]
                sf_g = gas_ids[idxg_f][_worker_gas_sfr_inst[gas_ids[idxg_f]] > 0.0]

                if sf_g.size > 0:
                    d3d = np.linalg.norm(_worker_stars_coords[star_id] - _worker_gas_coords[sf_g], axis=1)
                    sf_g_near = sf_g[d3d < 0.3]
                    if sf_g_near.size > 0 and np.nansum(_worker_gas_mass[sf_g_near]) > 0:
                        g_v_neb = np.nansum(_worker_gas_mass[sf_g_near] * _worker_gas_vel_los_proj[sf_g_near]) / np.nansum(_worker_gas_mass[sf_g_near])
                
                w_d_n, n_e_d = doppler_shift_spectrum(ssp_wave, n_em, g_v_neb)
                n_e_i = interp1d(w_d_n, n_e_d, kind='linear', bounds_error=False, fill_value=0.0)(ssp_wave)
                spec = s_c_i + n_e_i
                array_vel_los.append(_worker_stars_vel_los_proj[star_id]); array_vel_los_nebular_weighted.append(g_v_neb)
                array_L_nebular.append(simpson(n_e_i[lw_wave_idx]*norm, ssp_wave[lw_wave_idx]) if lw_wave_idx.size > 1 else 0.0)
            else: spec = s_cont + n_em

            spec_dust, d_AV = spec.copy(), 0.0

            # Dust Method Branching
            if _worker_dust_method == 'sfr_AV':
                d_AV = effective_av
            else:
                idx_c = np.where((gas_los_dist < star_los_dist[i_sid]) & ((_worker_gas_sfr_inst[gas_ids] > 0.0) | (_worker_gas_log_temp[gas_ids] < 3.9)))[0]
                if idx_c.size > 0:
                    g_m_s = np.nansum(_worker_gas_mass[gas_ids[idx_c]])
                    if g_m_s > 0:
                        g_z = np.nansum(_worker_gas_mass[gas_ids[idx_c]]*_worker_gas_zmet[gas_ids[idx_c]]/BAGPIPES_Z_SUN)/g_m_s
                        nH = np.nansum(_worker_gas_mass_H[gas_ids[idx_c]])*1.247914e+14/pix_area_kpc2
                        tauV = np.clip(_worker_scale_dust_tau * g_z * nH / 2.1e+21, 1e-10, None)
                        d_AV = -2.5*np.log10((1.0 - np.exp(-1.0*tauV))/tauV)
            
            if d_AV > 0:
                al = dust_reddening_diffuse_ism(d_AV, ssp_wave, dust_law)
                spec_dust *= 10.0**(-0.4*al)
                array_tauV.append(d_AV * 0.921); array_AV.append(d_AV)
            
            if _worker_stars_age[star_id] <= t_esc:
                al_bc = unresolved_dust_birth_cloud_Alambda_per_AV(ssp_wave, dust_index_bc=dust_index_bc) * d_AV * dust_eta
                spec_dust *= 10.0**(-0.4*al_bc)

            array_spec.append(spec*norm); array_spec_dust.append(spec_dust*norm)
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
            pixel_results['map_lw_zsol_nodust'] = np.nansum(np.asarray(array_L_nodust) * _worker_stars_zmet[star_ids] / BAGPIPES_Z_SUN) / L_n
        if L_d > 0:
            pixel_results['map_lw_age_dust'] = np.nansum(np.asarray(array_L_dust) * _worker_stars_age[star_ids]) / L_d
            pixel_results['map_lw_zsol_dust'] = np.nansum(np.asarray(array_L_dust) * _worker_stars_zmet[star_ids] / BAGPIPES_Z_SUN) / L_d
        if output_pixel_spectra_flag:
            v_los = np.asarray(array_vel_los)
            if L_n > 0: pixel_results['map_lw_vel_los_nodust'] = np.nansum(array_L_nodust * v_los) / L_n
            if L_d > 0: pixel_results['map_lw_vel_los_dust'] = np.nansum(array_L_dust * v_los) / L_d
            if np.nansum(array_L_nebular) > 0: pixel_results['map_lw_vel_los_nebular'] = np.nansum(array_L_nebular * array_vel_los_nebular_weighted) / np.nansum(array_L_nebular)
    
    return ii, jj, pixel_results

def generate_images(sim_file, z, filters, filter_transmission_path, smoothing_length=0.15, dim_kpc=None,
                    pix_arcsec=None, pix_kpc=0.1, flux_unit='MJy/sr', polar_angle_deg=0, azimuth_angle_deg=0,
                    name_out_img=None, n_jobs=-1, ssp_code='Bagpipes', gas_logu=-2.0,
                    igm_type=0, dust_index_bc=-0.7, dust_index=0.0, t_esc=0.01, dust_eta=1.0, 
                    scale_dust_redshift="Vogelsberger20", cosmo_str='Planck18', dust_method='los',
                    av_sfrden_relation=None, dust_law=0, bump_amp=0.85, bump_dwave=0.035, salim_a0=-4.30, 
                    salim_a1=2.71, salim_a2= -0.191, salim_a3=0.0121, salim_RV=3.15, salim_B=1.57, 
                    initdim_kpc=200, initdim_mass_fraction=0.99, use_precomputed_ssp=True, 
                    ssp_filepath=None, ssp_interpolation_method='nearest', 
                    output_pixel_spectra=False, rest_wave_min=1000.0, rest_wave_max=30000.0, 
                    rest_delta_wave=5.0): 
    
    """
    Generates astrophysical images from HDF5 simulation data using Bagpipes.

    This function orchestrates the image synthesis pipeline with parallelized pixel
    calculations. It allows the choice between using pre-computed SSP spectra
    from an HDF5 file or generating them on-the-fly using Bagpipes. It can
    optionally output observed-frame spectra for each pixel.

    Parameters:
        sim_file (str): Path to the HDF5 simulation file.
        z (float): Redshift of the galaxy.
        filters (list): List of photometric filters.
        filter_transmission_path (dict): Dictionary of paths to text files containing
                                         the transmission function. Keys are filter names,
                                         values are file paths.
        smoothing_length (float, optional): Smoothing length in kpc. Defaults to 0.15.
        dim_kpc (float, optional): Dimension of the image in kpc. If None, it is
                                   assigned automatically. Defaults to None.
        pix_arcsec (float, optional): Pixel size in arcseconds. Defaults to None.
        pix_kpc (float, optional): Pixel size in physical unit kpc. Defaults to 0.1. 
                                   If pix_kpc is provided, pix_arcsec input will be 
                                   ignored but converted from pix_kpc (given z) instead.
        flux_unit (str, optional): Desired flux unit for the generated images.
                                   Options: 'MJy/sr', 'nJy', 'AB magnitude',
                                   or 'erg/s/cm2/A'. Defaults to 'MJy/sr'.
        polar_angle_deg (float, optional): Polar angle for projection. Defaults to 0.
        azimuth_angle_deg (float, optional): Azimuth angle for projection. Defaults to 0.
        name_out_img (str, optional): Output file name for images. Defaults to None.
        n_jobs (int, optional): Number of CPU cores for parallel processing.
                                Defaults to -1 (all available).
        ssp_code (str, optional): The SSP code to use. Defaults to 'Bagpipes'.
        gas_logu (float, optional): Log ionization parameter for nebular emission.
                                    Defaults to -2.0.
        igm_type (int, optional): IGM absorption model type (0: Madau+95, 1: Inoue+14).
                                  Defaults to 0.
        dust_index_bc (float, optional): Dust index for birth clouds. Defaults to -0.7.
        dust_index (float, optional): Dust index for the diffuse ISM. Defaults to 0.0.
        t_esc (float, optional): Escape time for young stars from birth clouds in Gyr.
                                 Defaults to 0.01.
        dust_eta (float, optional): Ratio of A_V in birth clouds to the diffuse ISM.
                                    Defaults to 1.0.
        scale_dust_redshift (str or dict, optional): Defines the dust_tau normalization
                                                     vs. redshift relation. Can be a string
                                                     ("Vogelsberger20") or a dictionary.
                                                     Defaults to "Vogelsberger20".
        cosmo_str (str, optional): Cosmology string. Defaults to 'Planck18'.

        dust_method (str, optional): The framework used to calculate diffuse ISM dust attenuation. Options: 
                                        
                                        - 'los': (Default) Integration of the optical depth along the line-of-sight for each star particle based on local gas column density and metallicity. 
                                        
                                        - 'sfr_AV': Calculates an effective A_V for the entire pixel/grid cell based on the instantaneous SFR surface density (Msun/yr/kpc^2).

        av_sfrden_relation (dict, optional): Dictionary defining proportionality between 
                                             SFR surface density and A_V. Required if 
                                             dust_method='sfr_AV'. 
                                             Format: {'AV': [values], 'SFR_density': [values]}.
        dust_law (int, optional): Dust attenuation law type. Defaults to 0.
        bump_amp (float, optional): UV bump amplitude for the dust curve. Defaults to 0.85.
        bump_dwave (float, optional): Width (FWHM) of the dust attenuation bump in units of micron, as parameterized with the Drude profile. Defaults to 0.035 micron.
        salim_a0, salim_a1, salim_a2, salim_a3, salim_RV, salim_B (float, optional):
                                                  Parameters for the Salim+18 dust law.
        initdim_kpc (float, optional): Initial guess for image dimension in kpc for
                                       auto-sizing. Defaults to 200.
        initdim_mass_fraction (float, optional): Mass fraction to enclose when determining
                                                 image dimension. Defaults to 0.99.
        use_precomputed_ssp (bool, optional): If True, use pre-computed SSP spectra.
                                              If False, generate on-the-fly.
                                              Defaults to True.
        ssp_filepath (str, optional): Path to the pre-computed SSP spectra HDF5 file.
                                      Used if `use_precomputed_ssp` is True.
                                      Defaults to None.
        ssp_interpolation_method (str, optional): Method for interpolating SSPs ('nearest',
                                                  'linear', 'cubic'). Defaults to 'nearest'.
        output_pixel_spectra (bool, optional): If True, output observed-frame spectra
                                               for each pixel. Defaults to False.
        rest_wave_min (float, optional): Minimum rest-frame wavelength for output
                                         spectra (Angstroms). Defaults to 1000.0.
        rest_wave_max (float, optional): Maximum rest-frame wavelength for output
                                         spectra (Angstroms). Defaults to 30000.0.
        rest_delta_wave (float, optional): Wavelength step in rest-frame for output
                                           spectra (Angstroms). Defaults to 5.0.
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
        s_m, s_fz, s_z, s_c, s_v = f['star/mass'][:], f['star/form_z'][:], f['star/zmet'][:], f['star/coords'][:], f['star/vel'][:]
        s_im = f['star/init_mass'][:]
        s_age = snap_univ_age - interp_age_univ_from_z(s_fz, cosmo)
        idx = np.where(s_age>=0)[0]
        s_im, s_m, s_z, s_age, s_c, s_v = s_im[idx], s_m[idx], s_z[idx], s_age[idx], s_c[idx,:], s_v[idx,:]
        g_m, g_z, g_sfr, g_lt, g_c, g_v, g_mh = f['gas/mass'][:], f['gas/zmet'][:], f['gas/sfr_inst'][:], np.log10(f['gas/temp'][:]), f['gas/coords'][:], f['gas/vel'][:], f['gas/mass_H'][:]

    if dim_kpc is None:
        dim_kpc = determine_image_size(s_c, s_m, working_pix_kpc, (initdim_kpc, initdim_kpc), 
                                       polar_angle_deg, azimuth_angle_deg, g_c, g_m, 
                                       mass_percentage=initdim_mass_fraction, max_img_dim=initdim_kpc)

    s_mem, _, _, g_inf, g_mem, _, s_v_los, g_v_los = get_2d_density_projection_no_los_binning(s_c, s_m, working_pix_kpc, (dim_kpc, dim_kpc), 
                                                                                              polar_angle_deg=polar_angle_deg, azimuth_angle_deg=azimuth_angle_deg, 
                                                                                              gas_coords=g_c, gas_masses=g_m, star_vels=s_v, gas_vels=g_v)
    dimx, dimy = g_inf['num_pixels_x'], g_inf['num_pixels_y']
    print ('Working Cutout size: %d x %d pix or %d x %d kpc' % (dimx,dimy,dim_kpc,dim_kpc))

    if isinstance(scale_dust_redshift, str):
        data = np.loadtxt(str(importlib.resources.files('galsyn.data').joinpath("Vogelsberger20_scale_dust.txt")))
        n_d_z, n_d_t = data[:,0], data[:,1]
    else: n_d_z, n_d_t = np.asarray(scale_dust_redshift["z"]), np.asarray(scale_dust_redshift["tau_dust"])
    s_d_t = tau_dust_given_z(snap_z, n_d_z, n_d_t)

    fixed_global_output_obs_wave = np.arange(rest_wave_min*(1+z), (rest_wave_max+rest_delta_wave)*(1+z), rest_delta_wave*(1+z)) if output_pixel_spectra else np.array([])
    num_w = fixed_global_output_obs_wave.size

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
        results = Parallel(n_jobs=n_jobs, initializer=init_worker, initargs=(ssp_code, snap_z, pix_area_kpc2_working, 
                                                                             filters, filter_transmission_path, 
                                                                             gas_logu, igm_type, dust_index_bc, dust_index, 
                                                                             t_esc, dust_eta, s_d_t, cosmo_str, 
                                                                             dust_law, bump_amp, bump_dwave, # New 4-param dust model
                                                                             salim_a0, salim_a1, salim_a2, salim_a3, 
                                                                             salim_RV, salim_B, use_precomputed_ssp, 
                                                                             s_m, s_age, s_z, s_im, s_v_los, s_c, g_m, 
                                                                             g_sfr, g_z, g_lt, g_mh, g_v_los, g_c, 
                                                                             ssp_filepath, ssp_interpolation_method, 
                                                                             output_pixel_spectra,
                                                                             fixed_global_output_obs_wave.tolist() if isinstance(fixed_global_output_obs_wave, np.ndarray) else fixed_global_output_obs_wave, 
                                                                             dust_method, av_sfrden_relation))(delayed(_process_pixel_data)(*t) for t in tasks) 
                                                                             #tuple(fixed_global_output_obs_wave) if fixed_global_output_obs_wave.size==0 else fixed_global_output_obs_wave))(delayed(_process_pixel_data)(*t) for t in tasks)

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

    # Final Rebinning to target pixel size
    map_flux_summed = rebin_map(w_map_flux, rebin_factor, mode='sum')
    map_flux_dust_summed = rebin_map(w_map_flux_dust, rebin_factor, mode='sum')
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
    
    map_lw_age_nodust = rebin_map(w_map_lw['age_n'], rebin_factor, mode='mean')
    map_lw_age_dust = rebin_map(w_map_lw['age_d'], rebin_factor, mode='mean')
    map_lw_zsol_nodust = rebin_map(w_map_lw['z_n'], rebin_factor, mode='mean')
    map_lw_zsol_dust = rebin_map(w_map_lw['z_d'], rebin_factor, mode='mean')
    map_stars_mw_vel_los = rebin_map(w_map_lw['v_s_mw'], rebin_factor, mode='mean')
    map_gas_mw_vel_los = rebin_map(w_map_lw['v_g_mw'], rebin_factor, mode='mean')
    map_stars_vel_disp_los = rebin_map(w_map_lw['v_s_disp'], rebin_factor, mode='mean')
    map_gas_vel_disp_los = rebin_map(w_map_lw['v_g_disp'], rebin_factor, mode='mean')
    map_lw_vel_los_nodust = rebin_map(w_map_lw['v_n'], rebin_factor, mode='mean')
    map_lw_vel_los_dust = rebin_map(w_map_lw['v_d'], rebin_factor, mode='mean')
    map_lw_vel_los_nebular = rebin_map(w_map_lw['v_neb'], rebin_factor, mode='mean')

    if output_pixel_spectra:
        map_spectra_nodust = rebin_map(w_map_spec_n, rebin_factor, mode='sum')
        map_spectra_dust = rebin_map(w_map_spec_d, rebin_factor, mode='sum')

    # Unit conversion based on final rebinned pixels
    map_flux = np.zeros_like(map_flux_summed)
    map_flux_dust = np.zeros_like(map_flux_dust_summed)
    for i in range(len(filters)): 
        map_flux[:,:,i] = convert_flux_map(map_flux_summed[:,:,i], f_w_p_g[filters[i]], to_unit=flux_unit, pixel_scale_arcsec=pix_arcsec)
        map_flux_dust[:,:,i] = convert_flux_map(map_flux_dust_summed[:,:,i], f_w_p_g[filters[i]], to_unit=flux_unit, pixel_scale_arcsec=pix_arcsec)

    # SAVE TO FITS
    if name_out_img is not None:
        try:
            hdul = fits.HDUList()
            final_dimy, final_dimx = map_flux.shape[0], map_flux.shape[1]

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
                    ext_hdu = fits.ImageHDU(data=map_flux[:, :, i_band], header=ext_hdr)
                    hdul.append(ext_hdu)

                for i_band in range(len(filters)):
                    ext_hdr = fits.Header()
                    ext_hdr['EXTNAME'] = 'DUST_'+filters[i_band].upper()
                    ext_hdr['FILTER'] = filters[i_band]
                    ext_hdr['COMMENT'] = f'Flux (with dust) for filter: {filters[i_band]}'
                    ext_hdr['BUNIT'] = flux_unit
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
                'LW_VEL_LOS_NEBULAR': map_lw_vel_los_nebular
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

            if output_pixel_spectra:
                # Transpose the data cube from (dim_y, dim_x, wavelength) to (wavelength, dim_y, dim_x)
                transposed_map_spectra_nodust = map_spectra_nodust.transpose((2, 0, 1))
                transposed_map_spectra_dust = map_spectra_dust.transpose((2, 0, 1))

                ext_hdr_nodust_spec = fits.Header()
                ext_hdr_nodust_spec['EXTNAME'] = 'OBS_SPEC_NODUST'
                ext_hdr_nodust_spec['COMMENT'] = 'Observed-frame spectra (no dust attenuation)'
                
                ext_hdr_nodust_spec['CRPIX1'] = 1.0 # Wavelength axis becomes the first axis
                ext_hdr_nodust_spec['CRVAL1'] = fixed_global_output_obs_wave[0] if fixed_global_output_obs_wave.size > 0 else 0.0
                ext_hdr_nodust_spec['CDELT1'] = (fixed_global_output_obs_wave[1] - fixed_global_output_obs_wave[0]) if fixed_global_output_obs_wave.size > 1 else 0.0
                ext_hdr_nodust_spec['CUNIT1'] = 'Angstrom'

                ext_hdr_nodust_spec['CRPIX2'] = final_dimy / 2.0 + 0.5 # dim_y becomes the second axis
                ext_hdr_nodust_spec['CDELT2'] = pix_kpc
                ext_hdr_nodust_spec['CUNIT2'] = 'kpc'

                ext_hdr_nodust_spec['CRPIX3'] = final_dimx / 2.0 + 0.5 # dim_x becomes the third axis
                ext_hdr_nodust_spec['CDELT3'] = pix_kpc
                ext_hdr_nodust_spec['CUNIT3'] = 'kpc'

                ext_hdr_nodust_spec['BUNIT'] = 'erg/s/cm2/Angstrom'
                hdul.append(fits.ImageHDU(data=transposed_map_spectra_nodust, header=ext_hdr_nodust_spec))

                ext_hdr_dust_spec = fits.Header()
                ext_hdr_dust_spec['EXTNAME'] = 'OBS_SPEC_DUST'
                ext_hdr_dust_spec['COMMENT'] = 'Observed-frame spectra (with dust attenuation)'
                
                ext_hdr_dust_spec['CRPIX1'] = 1.0 # Wavelength axis becomes the first axis
                ext_hdr_dust_spec['CRVAL1'] = fixed_global_output_obs_wave[0] if fixed_global_output_obs_wave.size > 0 else 0.0
                ext_hdr_dust_spec['CDELT1'] = (fixed_global_output_obs_wave[1] - fixed_global_output_obs_wave[0]) if fixed_global_output_obs_wave.size > 1 else 0.0
                ext_hdr_dust_spec['CUNIT1'] = 'Angstrom'

                ext_hdr_dust_spec['CRPIX2'] = final_dimy / 2.0 + 0.5 # dim_y becomes the second axis
                ext_hdr_dust_spec['CDELT2'] = pix_kpc
                ext_hdr_dust_spec['CUNIT2'] = 'kpc'

                ext_hdr_dust_spec['CRPIX3'] = final_dimx / 2.0 + 0.5 # dim_x becomes the third axis
                ext_hdr_dust_spec['CDELT3'] = pix_kpc
                ext_hdr_dust_spec['CUNIT3'] = 'kpc'

                ext_hdr_dust_spec['BUNIT'] = 'erg/s/cm2/Angstrom'
                hdul.append(fits.ImageHDU(data=transposed_map_spectra_dust, header=ext_hdr_dust_spec))
                
                if fixed_global_output_obs_wave.size > 0:
                    col = fits.Column(name='WAVELENGTH', format='D', array=fixed_global_output_obs_wave)
                    cols = fits.ColDefs([col])
                    wavelength_hdu = fits.BinTableHDU.from_columns(cols, name='WAVELENGTH_GRID')
                    wavelength_hdu.header['BUNIT'] = 'Angstrom'
                    wavelength_hdu.header['COMMENT'] = 'Wavelength grid for OBS_SPEC_NODUST and OBS_SPEC_DUST'
                    hdul.append(wavelength_hdu)

            output_dir = os.path.dirname(name_out_img)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)

            hdul.writeto(name_out_img, overwrite=True, output_verify='fix')
            print(f"Galaxy image synthesis completed successfully and results saved to FITS file: {name_out_img}")

        except Exception as e:
            print(f"Error saving FITS file {name_out_img}: {e}")