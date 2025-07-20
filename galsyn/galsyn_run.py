import h5py
import os, sys 
import numpy as np 
from astropy.io import fits
from .imgutils import *
from .utils import *
from .dust import *
from joblib import Parallel, delayed
import multiprocessing
from scipy.interpolate import interp1d

from tqdm.auto import tqdm
from tqdm_joblib import tqdm_joblib

import fsps
sp_instance = None  # Global variable to avoid reloading fsps per model

def init_worker(snap_z_val, pix_area_kpc2_val, mean_AV_unres_val, filters_val, 
                filter_transmission_val, imf_type_val, add_neb_emission_val, 
                gas_logu_val, add_igm_absorption_val, igm_type_val, dust_index_bc_val, 
                dust_index_val, t_esc_val, scale_dust_tau_val, cosmo_val, dust_law_val,
                bump_amp_val, dustindexAV_AV_val, dustindexAV_dust_index_val, salim_a0_val, 
                salim_a1_val, salim_a2_val, salim_a3_val, salim_RV_val, salim_B_val):
    
    global sp_instance, igm_trans, snap_z, pix_area_kpc2
    global mean_AV_unres, filters, filter_transmission, imf_type
    global add_neb_emission, gas_logu, add_igm_absorption, igm_type
    global dust_index_bc, dust_index, t_esc, scale_dust_tau, cosmo
    global dust_law, bump_amp, salim_a0, salim_a1, salim_a2, salim_a3 
    global salim_RV, salim_B, dust_Alambda_per_AV

    snap_z = snap_z_val
    pix_area_kpc2 = pix_area_kpc2_val
    mean_AV_unres = mean_AV_unres_val
    filters = filters_val
    filter_transmission = filter_transmission_val
    imf_type = imf_type_val
    add_neb_emission = add_neb_emission_val
    gas_logu = gas_logu_val
    add_igm_absorption = add_igm_absorption_val
    igm_type = igm_type_val
    dust_index_bc = dust_index_bc_val
    t_esc = t_esc_val
    scale_dust_tau = scale_dust_tau_val
    cosmo = cosmo_val
    dust_law = dust_law_val
    salim_a0 = salim_a0_val
    salim_a1 = salim_a1_val
    salim_a2 = salim_a2_val
    salim_a3 = salim_a3_val
    salim_RV = salim_RV_val
    salim_B = salim_B_val
    dust_index = dust_index_val
    bump_amp = bump_amp_val

    sp_instance = fsps.StellarPopulation(zcontinuous=1)
    sp_instance.params['imf_type'] = imf_type
    sp_instance.params["add_dust_emission"] = False
    sp_instance.params["add_neb_emission"] = add_neb_emission
    sp_instance.params['gas_logu'] = gas_logu
    sp_instance.params["fagn"] = 0
    sp_instance.params["sfh"] = 0   # SSP
    sp_instance.params["dust1"] = 0.0
    sp_instance.params["dust2"] = 0.0   # optical depth

    wave, spec = sp_instance.get_spectrum(peraa=True, tage=1.0)

    if dust_law <= 1:
        global func_interp_dust_index
        dustindexAV_AV = dustindexAV_AV_val
        dustindexAV_dust_index = dustindexAV_dust_index_val
        func_interp_dust_index = interp1d(dustindexAV_AV, dustindexAV_dust_index, bounds_error=False, fill_value='extrapolate')

    elif dust_law == 2:  # modified Calzetti+20 with Bump strength tied to dust_index and dust_index is free. Dust index is single value applied to all star particles irrespective of A_V
        bump_amp = bump_amp_from_dust_index(dust_index)
        dust_Alambda_per_AV = modified_calzetti_dust_Alambda_per_AV(wave, dust_index=dust_index, bump_amp=bump_amp)

    elif dust_law == 3:  # modified Calzetti+20 with free Bump strengh and dust_index.  Dust index is single value applied to all star particles irrespective of A_V
        dust_Alambda_per_AV = modified_calzetti_dust_Alambda_per_AV(wave, dust_index=dust_index, bump_amp=bump_amp)

    elif dust_law == 4:  # Salim et al. (2018)
        dust_Alambda_per_AV = salim18_dust_Alambda_per_AV(wave, salim_a0, salim_a1, salim_a2, salim_a3, salim_B, salim_RV)

    elif dust_law == 5:  # Calzetti et al. (2000)
        dust_Alambda_per_AV = calzetti_dust_Alambda_per_AV(wave)

    if add_igm_absorption == 1:
        if igm_type == 0:
            igm_trans = igm_att_madau(wave * (1.0+snap_z), snap_z)
        elif igm_type == 1:
            igm_trans = igm_att_inoue(wave * (1.0+snap_z), snap_z)
        else:
            print ('igm_type is not recognized! options are: 1 for Madau+1995 and Inoue+2014')
            sys.exit()
    else:
        igm_trans = 1


def dust_reddening_diffuse_ism(dust_AV, wave, dust_law):
    if dust_law == 0:  # modified Calzetti+20 with Bump strength tied to dust_index and dust_index is dependent on A_V
        dust_index1 = func_interp_dust_index(dust_AV)
        bump_amp1 = bump_amp_from_dust_index(dust_index1)
        Alambda = modified_calzetti_dust_Alambda_per_AV(wave, dust_index=dust_index1, bump_amp=bump_amp1) * dust_AV

    elif dust_law == 1:  # modified Calzetti+20 with free Bump strengh and dust_index is dependent on A_V
        dust_index1 = func_interp_dust_index(dust_AV)
        Alambda = modified_calzetti_dust_Alambda_per_AV(wave, dust_index=dust_index1, bump_amp=bump_amp) * dust_AV

    #elif dust_law == 2:  # modified Calzetti+20 with Bump strength tied to dust_index and dust_index is free. Dust index is single value applied to all star particles irrespective of A_V
    #    bump_amp1 = bump_amp_from_dust_index(dust_index)
    #    Alambda = modified_calzetti_dust_Alambda_per_AV(wave, dust_index=dust_index, bump_amp=bump_amp1) * dust_AV
    #elif dust_law == 3:  # modified Calzetti+20 with free Bump strengh and dust_index.  Dust index is single value applied to all star particles irrespective of A_V
    #    Alambda = modified_calzetti_dust_Alambda_per_AV(wave, dust_index=dust_index, bump_amp=bump_amp) * dust_AV
    #elif dust_law == 4:  # Salim et al. (2018)
    #    Alambda = salim18_dust_Alambda_per_AV(wave, salim_a0, salim_a1, salim_a2, salim_a3, salim_B, salim_RV) * dust_AV
    #elif dust_law == 5:  # Calzetti et al. (2000)
    #    Alambda = calzetti_dust_Alambda_per_AV(wave) * dust_AV
    #else:
    #    print ('dust_law choice not recognized!')
    #    sys.exit()

    return Alambda


def _process_pixel_data(ii, jj, star_particle_membership, gas_particle_membership, 
                        stars_mass, stars_age, stars_zsol, stars_init_mass, 
                        gas_mass, gas_sfr_inst, gas_zsol, gas_log_temp, gas_mass_H):
    """
    ii=y jj=x
    Worker function to process calculations for a single pixel (ii, jj).
    This function will be executed in parallel for each pixel.
    """

    # Initialize a dictionary to store results for this pixel
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
        'map_flux': np.zeros(len(filters)),
        'map_flux_dust': np.zeros(len(filters))
    }

    star_ids0 = np.asarray([x[0] for x in star_particle_membership[ii][jj]], dtype=int)
    idxs = np.where((np.isnan(stars_mass[star_ids0]) == False) &
                     (np.isnan(stars_age[star_ids0]) == False) &
                     (np.isnan(stars_zsol[star_ids0]) == False))[0]
    star_ids = star_ids0[idxs]
    
    star_los_dist0 = np.asarray([x[1] for x in star_particle_membership[ii][jj]])
    star_los_dist = star_los_dist0[idxs]

    gas_ids0 = np.asarray([x[0] for x in gas_particle_membership[ii][jj]], dtype=int)
    idxg = np.where(np.isnan(gas_mass[gas_ids0]) == False)[0]
    gas_ids = gas_ids0[idxg]
    
    gas_los_dist0 = np.asarray([x[1] for x in gas_particle_membership[ii][jj]])
    gas_los_dist = gas_los_dist0[idxg]

    current_stars_mass_sum = np.nansum(stars_mass[star_ids])
    pixel_results['map_stars_mass'] = current_stars_mass_sum

    if current_stars_mass_sum > 0:
        pixel_results['map_mw_age'] = np.nansum(stars_mass[star_ids] * stars_age[star_ids]) / current_stars_mass_sum
        pixel_results['map_stars_mw_zsol'] = np.nansum(stars_mass[star_ids] * stars_zsol[star_ids]) / current_stars_mass_sum
    else:
        pixel_results['map_mw_age'] = np.nan
        pixel_results['map_stars_mw_zsol'] = np.nan

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
        pixel_results['map_gas_mw_zsol'] = np.nansum(gas_mass[gas_ids] * gas_zsol[gas_ids]) / current_gas_mass_sum
    else:
        pixel_results['map_gas_mw_zsol'] = np.nan

    # Model spectra of SSP, get total spectrum and photometric fluxes
    if len(star_ids) > 0:

        array_spec = []
        array_spec_dust = []
        array_AV = []
        array_tauV = []

        for i_sid in range(len(star_ids)):
            star_id = star_ids[i_sid]

            logzsol = np.log10(stars_zsol[star_id])
            sp_instance.params["logzsol"] = logzsol   
            sp_instance.params['gas_logz'] = logzsol

            wave, spec = sp_instance.get_spectrum(peraa=True, tage=stars_age[star_id])        # spectrum in L_sun/AA

            # calculate total cold hydrogen gas column density in front of the particle along the line of sight
            idxg_front = np.where(gas_los_dist < star_los_dist[i_sid])[0]
            front_gas_ids = gas_ids[idxg_front]
            idxg1 = np.where((gas_sfr_inst[front_gas_ids] > 0.0) | (gas_log_temp[front_gas_ids] < 3.9))[0]
            cold_front_gas_ids = front_gas_ids[idxg1]

            if len(cold_front_gas_ids) > 0:
                temp_mw_gas_zsol = np.nansum(gas_mass[cold_front_gas_ids]*gas_zsol[cold_front_gas_ids])/np.nansum(gas_mass[cold_front_gas_ids])
                nH = np.nansum(gas_mass_H[cold_front_gas_ids])*1.247914e+14/pix_area_kpc2      # number of hydrogen atom per cm^2
                tauV = scale_dust_tau * temp_mw_gas_zsol * nH / 2.1e+21
                dust_AV = -2.5*np.log10((1.0 - np.exp(-1.0*tauV))/tauV)

                if np.isnan(dust_AV)==True or dust_AV==0.0:
                    spec_dust = spec
                else:
                    # attenuation by resolved dust in the diffuse ISM
                    if dust_law <= 1:
                        Alambda = dust_reddening_diffuse_ism(dust_AV, wave, dust_law)
                    else:
                        Alambda = dust_Alambda_per_AV * dust_AV
                    #Alambda = modified_calzetti_dust_curve(dust_AV, wave, dust_index=dust_index)
                    
                    spec_dust = spec*np.power(10.0, -0.4*Alambda)
                    array_tauV.append(tauV)
                    array_AV.append(dust_AV)
            else:
                spec_dust = spec

            if stars_age[star_id] <= t_esc:    # age criterion defining young stars associated with birth clouds 
                # attenuation by unresolved dust in the birth cloud
                Alambda = unresolved_dust_birth_cloud_Alambda_per_AV(wave, dust_index_bc=dust_index_bc) * mean_AV_unres
                #A_lambda = unresolved_dust_birth_cloud(mean_AV_unres, wave, dust_index_bc=dust_index_bc)
                spec_dust = spec_dust*np.power(10.0, -0.4*Alambda)
    
            norm = stars_mass[star_id]/sp_instance.stellar_mass

            if len(np.asarray(spec_dust).shape) == 1:
                array_spec.append(spec*norm)
                array_spec_dust.append(spec_dust*norm)

        # average AV:
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

            # redshifting:
            spec_wave, spec_flux = cosmo_redshifting(wave, spec_lum, snap_z, cosmo)              # in erg/s/cm^2/Ang.
            spec_wave, spec_flux_dust = cosmo_redshifting(wave, spec_lum_dust, snap_z, cosmo)    # in erg/s/cm^2/Ang.

            # filtering
            nbands = len(filters)
            redshift_flux = np.zeros(nbands)
            redshift_flux_dust = np.zeros(nbands)

            for i_band in range(nbands):
                redshift_flux[i_band] = filtering(spec_wave, spec_flux*igm_trans, filter_transmission[filters[i_band]]['wave'], filter_transmission[filters[i_band]]['trans'])
                redshift_flux_dust[i_band] = filtering(spec_wave, spec_flux_dust*igm_trans, filter_transmission[filters[i_band]]['wave'], filter_transmission[filters[i_band]]['trans'])

        if len(redshift_flux) > 0:
            pixel_results['map_flux'] = redshift_flux
            pixel_results['map_flux_dust'] = redshift_flux_dust

            pixel_results['map_dust_mean_tauV'] = mean_tauV
            pixel_results['map_dust_mean_AV'] = mean_AV

    return ii, jj, pixel_results


def generate_images(sim_file, z, filters, filter_transmission, filter_wave_eff, dim_kpc=5.0, 
                    pix_arcsec=0.02, flux_unit='MJy/sr', polar_angle_deg=0, azimuth_angle_deg=0,
                    name_out_img=None, n_jobs=-1, imf_type=1, add_neb_emission=1, gas_logu=-2.0, 
                    add_igm_absorption=1, igm_type=0, dust_index_bc=-0.7, dust_index=0.0, t_esc=0.01, 
                    norm_dust_z=[], norm_dust_tau=[], cosmo_str='Planck18', cosmo_h=0.6774, XH=0.76, 
                    dust_law=0, bump_amp=0.85, dustindexAV_AV=[], dustindexAV_dust_index=[], salim_a0=-4.30, 
                    salim_a1=2.71, salim_a2=-0.191, salim_a3=0.0121, salim_RV=3.15, salim_B=1.57):
    """
    Generates astrophysical images from HDF5 simulation data with parallelized pixel calculations.

    Parameters:
        sim_file (str): Path to the HDF5 simulation file.
        snap_number (int): Snapshot number.
        filters (list): List of photometric filters.
        filter_transmission: [filter string]['wave' or 'trans']
        filter_wave_eff: [filter string]
        z (float, optional): Redshift. If None, derived from snap_number. Defaults to None.
        dim_kpc (float, optional): Dimension of the image in kpc. If None, assigned automatically. Defaults to None.
        pix_arcsec (float, optional): Pixel size in arcseconds. Defaults to 0.02.
        flux_unit (string, optional): Desired flux unit for the generated images. Options are: 'MJy/sr', 'nJy', 'AB magnitude', or 'erg/s/cm2/A'. Default to 'MJy/sr'.
        name_out_img (str, optional): Output file name for images. Defaults to None.
        ncpu (int, optional): Number of CPU cores to use for parallel processing.
    """

    cosmo = define_cosmo(cosmo_str)

    print ('Processing '+sim_file)
    # --- Data Loading and Initial Calculations (Sequential) ---

    snap_z = z
    snap_a = 1.0/(1.0 + snap_z)

    pix_kpc = angular_to_physical(snap_z, pix_arcsec, cosmo)
    pix_area_kpc2 = pix_kpc*pix_kpc
    print ('pixel size: %lf arcsec or %lf kpc' % (pix_arcsec,pix_kpc))

    f = h5py.File(sim_file,'r')

    stars_form_a = f['PartType4']['GFM_StellarFormationTime'][:]
    stars_form_z = (1.0/stars_form_a) - 1.0

    stars_init_mass = f['PartType4']['GFM_InitialMass'][:]*1e+10/cosmo_h
    stars_mass = f['PartType4']['Masses'][:]*1e+10/cosmo_h
    stars_zsol = f['PartType4']['GFM_Metallicity'][:]/0.0127         # in solar metallicity

    coords = f['PartType4']['Coordinates'][:]
    coords_x = coords[:,0]*snap_a/cosmo_h             # x in kpc
    coords_y = coords[:,1]*snap_a/cosmo_h             # y in kpc
    coords_z = coords[:,2]*snap_a/cosmo_h             # z in kpc

    snap_univ_age = cosmo.age(snap_z).value
    stars_form_age_univ = interp_age_univ_from_z(stars_form_z, cosmo)
    stars_age = snap_univ_age - stars_form_age_univ                 # age in Gyr
    # select only stellar particles
    idx = np.where((stars_form_a>0) & (stars_age>=0))[0]
    stars_form_a = stars_form_a[idx]
    stars_form_z = stars_form_z[idx]
    stars_init_mass = stars_init_mass[idx]
    stars_mass = stars_mass[idx]
    stars_zsol = stars_zsol[idx]
    stars_coords_x = coords_x[idx]
    stars_coords_y = coords_y[idx]
    stars_coords_z = coords_z[idx]
    stars_age = stars_age[idx]

    # read gas particles data
    gas_mass = f['PartType0']['Masses'][:]*1e+10/cosmo_h
    gas_zsol = f['PartType0']['GFM_Metallicity'][:]/0.0127  # in solar metallicity
    gas_coords = f['PartType0']['Coordinates'][:]
    gas_coords_x = gas_coords[:,0]*snap_a/cosmo_h                 # in kpc
    gas_coords_y = gas_coords[:,1]*snap_a/cosmo_h
    gas_coords_z = gas_coords[:,2]*snap_a/cosmo_h
    gas_sfr_inst = f['PartType0']['StarFormationRate'][:]   # in Msun/yr 
    gas_mass_H = gas_mass*XH                     # mass of hydrogen 
    # calculate gas temperature
    u = f['PartType0']['InternalEnergy'][:]      #  the Internal Energy
    Xe = f['PartType0']['ElectronAbundance'][:]  # xe (=ne/nH)  the electron abundance                
    gamma = 5.0/3.0          # the adiabatic index
    KB = 1.3807e-16          # the Boltzmann constant in CGS units  [cm^2 g s^-2 K^-1]
    mp = 1.6726e-24          # the proton mass  [g]
    mu = (4*mp)/(1+3*XH+4*XH*Xe)
    gas_log_temp = np.log10((gamma-1.0)*(u/KB)*mu*1e+10)  # log temperature in Kelvin

    f.close()

    if dim_kpc is None:
        dim_kpc = assign_cutout_size(snap_z, np.log10(np.nansum(stars_mass)))

    star_coords = np.column_stack((stars_coords_x, stars_coords_y, stars_coords_z))
    gas_coords = np.column_stack((gas_coords_x, gas_coords_y, gas_coords_z))

    output_dimension = (dim_kpc, dim_kpc)
    star_particle_membership, star_mass_density_map, central_pixel_coords, grid_info, gas_particle_membership, gas_mass_density_map = get_2d_density_projection_no_los_binning(
                                                                                                                                                star_coords, 
                                                                                                                                                stars_mass, 
                                                                                                                                                pix_kpc, 
                                                                                                                                                output_dimension, 
                                                                                                                                                polar_angle_deg=polar_angle_deg, 
                                                                                                                                                azimuth_angle_deg=azimuth_angle_deg, 
                                                                                                                                                gas_coords=gas_coords, 
                                                                                                                                                gas_masses=gas_mass)
    
    dimx, dimy = grid_info['num_pixels_x'], grid_info['num_pixels_y']
    print ('Cutout size: %d x %d pix or %d x %d kpc' % (dimx,dimy,dim_kpc,dim_kpc))

    # ------ estimate unresolved dust AV associated with the birth cloud ------- #
    idxg_global = np.where((gas_sfr_inst>0.0) | (gas_log_temp<3.9))[0]
    # calculate Hydrogen column density
    if np.nansum(gas_mass[idxg_global]) > 0:
        temp_mw_gas_zsol = np.nansum(gas_mass[idxg_global]*gas_zsol[idxg_global])/np.nansum(gas_mass[idxg_global])
    else:
        temp_mw_gas_zsol = 0.0 # Default if no gas mass
    nH = np.nansum(gas_mass_H[idxg_global])*1.247914e+14/dim_kpc/dim_kpc      # number of hydrogen atom per cm^2

    scale_dust_tau = tau_dust_given_z(snap_z, norm_dust_z, norm_dust_tau)
    mean_tauV_res = scale_dust_tau*temp_mw_gas_zsol*nH/2.1e+21 

    global mean_AV_unres
    if np.isnan(mean_tauV_res)==True or np.isinf(mean_tauV_res)==True:
        mean_tauV_res, mean_AV_unres = 0.0, 0.0
    else:
        mean_AV_unres = -2.5*np.log10(np.exp(-2.0*mean_tauV_res))                 # assumed twice of average tauV resolved (Vogelsberger+20)
    print ('mean_tauV_res=%lf mean_AV_unres=%lf' % (mean_tauV_res,mean_AV_unres))
    # -----------

    nbands = len(filters)

    # Initialize all result maps to zeros
    map_mw_age = np.zeros((dimy,dimx))
    map_stars_mw_zsol = np.zeros((dimy,dimx))
    map_stars_mass = np.zeros((dimy,dimx))
    map_sfr_100 = np.zeros((dimy,dimx))
    map_sfr_10 = np.zeros((dimy,dimx))
    map_sfr_30 = np.zeros((dimy,dimx))

    map_gas_mw_zsol = np.zeros((dimy,dimx))
    map_gas_mass = np.zeros((dimy,dimx))
    map_sfr_inst = np.zeros((dimy,dimx))

    map_dust_mean_tauV = np.zeros((dimy,dimx))
    map_dust_mean_AV = np.zeros((dimy,dimx))

    map_flux = np.zeros((dimy,dimx,nbands))
    map_flux_dust = np.zeros((dimy,dimx,nbands))

    # --- Parallel Calculation Section ---
    tasks = []
    for ii in range(dimy): # Iterate over rows (y-axis)
        for jj in range(dimx): # Iterate over columns (x-axis)
            tasks.append((ii, jj, star_particle_membership, gas_particle_membership, 
                            stars_mass, stars_age, stars_zsol, stars_init_mass, 
                            gas_mass, gas_sfr_inst, gas_zsol, gas_log_temp, gas_mass_H))

    # Determine the number of CPU cores to use
    num_cores = n_jobs
    if num_cores == -1:
        num_cores = multiprocessing.cpu_count() # Use all available cores

    print(f"\nStarting parallel pixel processing on {num_cores} cores...")

    with tqdm_joblib(total=len(tasks), desc="Processing pixels") as progress_bar:
        results = Parallel(n_jobs=num_cores, verbose=0, initializer=init_worker,
                           initargs=(snap_z, pix_area_kpc2, mean_AV_unres, filters, filter_transmission, imf_type, add_neb_emission, 
                                     gas_logu, add_igm_absorption, igm_type, dust_index_bc, dust_index, t_esc, scale_dust_tau, cosmo, 
                                     dust_law, bump_amp, dustindexAV_AV, dustindexAV_dust_index, salim_a0, salim_a1, salim_a2, salim_a3, 
                                     salim_RV, salim_B))(
            delayed(_process_pixel_data)(*task_args) for task_args in tasks
        )
    print("\nFinished parallel pixel processing.")

    # --- Aggregate Results (Sequential) ---
    # Populate the pre-initialized maps with results from parallel processing
    for ii, jj, pixel_data in results:
        map_stars_mass[ii][jj] = pixel_data['map_stars_mass']
        map_mw_age[ii][jj] = pixel_data['map_mw_age']
        map_stars_mw_zsol[ii][jj] = pixel_data['map_stars_mw_zsol']
        map_sfr_100[ii][jj] = pixel_data['map_sfr_100']
        map_sfr_30[ii][jj] = pixel_data['map_sfr_30']
        map_sfr_10[ii][jj] = pixel_data['map_sfr_10']

        map_gas_mass[ii][jj] = pixel_data['map_gas_mass']
        map_sfr_inst[ii][jj] = pixel_data['map_sfr_inst']
        map_gas_mw_zsol[ii][jj] = pixel_data['map_gas_mw_zsol']

        map_dust_mean_tauV[ii][jj] = pixel_data['map_dust_mean_tauV']
        map_dust_mean_AV[ii][jj] = pixel_data['map_dust_mean_AV']

        map_flux[ii][jj] = pixel_data['map_flux']
        map_flux_dust[ii][jj] = pixel_data['map_flux_dust']

    print("All calculations complete. Maps populated.")

    # convert flux units
    for i_band in range(nbands):
        map_flux[:,:,i_band] = convert_flux_map(map_flux[:,:,i_band], filter_wave_eff[filters[i_band]], to_unit=flux_unit, pixel_scale_arcsec=pix_arcsec)
        map_flux_dust[:,:,i_band] = convert_flux_map(map_flux_dust[:,:,i_band], filter_wave_eff[filters[i_band]], to_unit=flux_unit, pixel_scale_arcsec=pix_arcsec)

    # rescaling maps if flux_unit is 'erg/s/cm2/A'
    if flux_unit == 'erg/s/cm2/A':
        flux_scale = 1e-20
    else:
        flux_scale = 1.0
    
    map_flux = map_flux/flux_scale
    map_flux_dust = map_flux_dust/flux_scale

    # --- Save results to FITS file (New Block) ---
    if name_out_img is not None:
        try:
            # Create a list of HDUs
            hdul = fits.HDUList()

            # Primary HDU: Can be empty or hold a basic header
            if map_flux.shape[2] > 0:
                primary_data = map_flux[:, :, 0] # First band for primary
                prihdr = fits.Header()
                prihdr['COMMENT'] = 'Primary Image: First band (no dust)'
                prihdr['CRPIX1'] = dimx / 2.0 + 0.5 # Reference pixel X (center)
                prihdr['CRPIX2'] = dimy / 2.0 + 0.5 # Reference pixel Y (center)
                prihdr['CDELT1'] = pix_kpc # Pixel scale X (kpc/pixel)
                prihdr['CDELT2'] = pix_kpc # Pixel scale Y (kpc/pixel)
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

                primary_hdu = fits.PrimaryHDU(data=primary_data, header=prihdr)
                hdul.append(primary_hdu)

                # Add extensions for other bands (no dust)
                for i_band in range(nbands):
                    ext_hdr = fits.Header()
                    ext_hdr['EXTNAME'] = 'NODUST_'+filters[i_band].upper()
                    ext_hdr['FILTER'] = filters[i_band]
                    ext_hdr['COMMENT'] = f'Flux for filter: {filters[i_band]}'
                    ext_hdu = fits.ImageHDU(data=map_flux[:, :, i_band], header=ext_hdr)
                    hdul.append(ext_hdu)

                # Add extensions for other bands (with dust)
                for i_band in range(nbands):
                    ext_hdr = fits.Header()
                    ext_hdr['EXTNAME'] = 'DUST_'+filters[i_band].upper()
                    ext_hdr['FILTER'] = filters[i_band]
                    ext_hdr['COMMENT'] = f'Flux (with dust) for filter: {filters[i_band]}'
                    ext_hdu = fits.ImageHDU(data=map_flux_dust[:, :, i_band], header=ext_hdr)
                    hdul.append(ext_hdu)
            else:
                # If no flux bands, create an empty primary HDU
                hdul.append(fits.PrimaryHDU())

            # Add other maps as ImageHDU extensions
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
            }

            for map_name, data_array in map_data_to_save.items():
                if data_array is not None: # Ensure the array exists
                    ext_hdr = fits.Header()
                    ext_hdr['EXTNAME'] = map_name
                    ext_hdr['COMMENT'] = f'Map of {map_name.replace("_", " ").title()}'
                    hdul.append(fits.ImageHDU(data=data_array, header=ext_hdr))

            # Ensure the directory exists
            output_dir = os.path.dirname(name_out_img)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # Write the HDUList to the FITS file
            hdul.writeto(name_out_img, overwrite=True, output_verify='fix')
            print(f"Galaxy image synthesis completed successfully and results saved to FITS file: {name_out_img}")

        except Exception as e:
            print(f"Error saving FITS file {name_out_img}: {e}")



