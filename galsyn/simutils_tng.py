import os, sys 
import numpy as np
from astropy.io import fits 
from .config import API_KEY
from .imgutils import *
from .utils import *

baseUrl_tng = 'http://www.tng-project.org/api/'
headers = {"api-key":API_KEY}

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

def cosmic_times_snapshots(sim='TNG50-1',cosmo=None, snaps_info=None):

    if cosmo is None:
        from astropy.cosmology import Planck15 as cosmo

    if snaps_info is None:
        snaps_info = get_tng_snaps_info(sim)

    nsnaps = 100
    cosmic_times = np.zeros(nsnaps)
    for ii in range(nsnaps):
        snap_z = get_snap_z(ii, sim=sim, snaps_info=snaps_info)
        cosmic_times[ii] = cosmo.age(snap_z).value

    return cosmic_times

def cosmic_times_of_snapshots(snaps, sim='TNG50-1', cosmo=None, snaps_info=None):

    if cosmo is None:
        from astropy.cosmology import Planck15 as cosmo

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

def assign_cutout_size(z, log_totmstar):
    
    if z<=0.5:
        if log_totmstar >= 9.0:
            dim_kpc = 100 
        else:
            dim_kpc = 50
    elif z<=2.0:
        if log_totmstar >= 9.0:
            dim_kpc = 90 
        else:
            dim_kpc = 44
    elif z<=3.0:
        if log_totmstar >= 9.0:
            dim_kpc = 80 
        else:
            dim_kpc = 40 
    elif z<=4.0:
        if log_totmstar >= 9.0:
            dim_kpc = 50 
        else:
            dim_kpc = 24 
    else:
        if log_totmstar >= 9.0:
            dim_kpc = 40 
        else:
            dim_kpc = 20 

    return dim_kpc


def tau_dust_given_z(z):
    from scipy.interpolate import interp1d
    # based on Vogelsberger+2020 (Table 3)
    data_z = [0, 2, 3, 4, 5, 6, 7, 8, 12]
    data_tau_dust = [0.46, 0.46, 0.20, 0.13, 0.08, 0.06, 0.04, 0.03, 0.03]

    f = interp1d(data_z, data_tau_dust, fill_value="extrapolate")
    return f(z)


def construct_SFH_TNG(stars_form_lbt, stars_init_mass, del_t=0.3, max_lbt=18.0):
    """
    Constructs the Star Formation History (SFH) from stellar formation times
    and initial masses using vectorized NumPy operations for efficiency.

    Args:
        stars_form_lbt (array-like): Array of stellar formation lookback times in Gyr.
        stars_init_mass (array-like): Array of initial stellar masses in solar mass,
                                       corresponding to stars_form_lbt.
        del_t (float, optional): Time bin width in Gyr. Defaults to 0.3 Gyr.
        max_lbt (float, optional): Maximum lookback time to consider for binning.
                                   Defaults to 18.0 Gyr.

    Returns:
        tuple: A tuple containing two dictionaries:
               - sfh (dict): Contains 'lbt' (midpoints of bins), 'sfr' (star formation rate),
                             'nstars' (number of stars), and 'smgh' (stellar mass growth history).
               - sfh_ladder (dict): Contains 'lbt', 'sfr', 'nstars', and 'smgh' for a ladder plot,
                                    with duplicated entries for step-like visualization.
    """
    stars_form_lbt = np.asarray(stars_form_lbt)
    stars_init_mass = np.asarray(stars_init_mass)

    # Define the bins for lookback time.
    # The bins are inclusive of the lower bound and exclusive of the upper bound [t, t + del_t).
    # We add a small epsilon to max_lbt to ensure that max_lbt itself is included in a bin
    # if it falls exactly on a bin edge.
    bins = np.arange(0, max_lbt + del_t, del_t)

    # Calculate total initial mass for SMGH calculation
    total_initial_mass = np.nansum(stars_init_mass)

    # Use np.histogram to efficiently bin the data
    # mass_in_bins: Sum of initial masses in each time bin
    # nstars_in_bins: Count of stars in each time bin
    mass_in_bins, _ = np.histogram(stars_form_lbt, bins=bins, weights=stars_init_mass)
    nstars_in_bins, _ = np.histogram(stars_form_lbt, bins=bins)

    # Identify valid bins (where at least one star was formed)
    valid_bins_mask = nstars_in_bins > 0

    # Extract data for valid bins
    valid_masses = mass_in_bins[valid_bins_mask]
    valid_nstars = nstars_in_bins[valid_bins_mask]

    # Calculate lookback time midpoints for the sfh dictionary
    # bins[:-1] gives the start times of the bins
    sfh_lbt_midpoints = bins[:-1][valid_bins_mask] + 0.5 * del_t

    # Calculate Star Formation Rate (SFR)
    # SFR is mass formed per unit time, converted to solar mass per year (1e9 for Gyr to year)
    sfh_sfr = valid_masses / del_t / 1e9

    # Calculate Stellar Mass Growth History (SMGH)
    # SMGH represents the total initial mass of stars that have *not yet formed*
    # at the start of each bin.
    # First, calculate the cumulative mass formed up to the end of each bin (including empty ones)
    cumulative_mass_formed_at_end_of_bin = np.cumsum(mass_in_bins)
    # Then, subtract this from the total initial mass.
    # For the first bin, the mass not yet formed is the total_initial_mass.
    # For subsequent bins, it's total_initial_mass - (mass formed up to previous bin's end).
    smgh_all_bins = total_initial_mass - np.concatenate(([0], cumulative_mass_formed_at_end_of_bin[:-1]))
    # Filter for only the valid bins
    sfh_smgh = smgh_all_bins[valid_bins_mask]

    # Construct the sfh dictionary
    sfh = {
        'lbt': sfh_lbt_midpoints,
        'sfr': sfh_sfr,
        'nstars': valid_nstars,
        'smgh': sfh_smgh
    }

    # --- Constructing the sfh_ladder dictionary for step-like plots ---
    # These arrays will have duplicated values to create the "steps"

    # Get the start and end times for the valid bins
    valid_lbt_starts = bins[:-1][valid_bins_mask]
    valid_lbt_ends = bins[1:][valid_bins_mask]

    # Interleave start and end times for the ladder's LBT
    sfh_ladder_lbt = np.empty(2 * len(valid_lbt_starts))
    sfh_ladder_lbt[0::2] = valid_lbt_starts
    sfh_ladder_lbt[1::2] = valid_lbt_ends

    # Duplicate SFR, Nstars, and SMGH values for the ladder
    sfh_ladder_sfr = np.repeat(sfh_sfr, 2)
    sfh_ladder_nstars = np.repeat(valid_nstars, 2)
    sfh_ladder_smgh = np.repeat(sfh_smgh, 2)

    # Construct the sfh_ladder dictionary
    sfh_ladder = {
        'lbt': sfh_ladder_lbt,
        'sfr': sfh_ladder_sfr,
        'nstars': sfh_ladder_nstars,
        'smgh': sfh_ladder_smgh
    }

    return sfh, sfh_ladder


def modified_calzetti_dust_curve(AV, wave, dust_index=0.0):
    wave = wave/1e+4     # in micron
    idx = np.where(wave <= 0.63)[0]
    k_lambda1 = 4.05 + (2.659*(-2.156 + (1.509/wave[idx]) - (0.198/wave[idx]/wave[idx]) + (0.011/wave[idx]/wave[idx]/wave[idx])))

    idx = np.where(wave > 0.63)[0]
    k_lambda2 = 4.05 + (2.659*(-1.857 + (1.040/wave[idx]))) 

    k_lambda = k_lambda1.tolist() + k_lambda2.tolist()
    k_lambda = np.asarray(k_lambda)

    wave_V = 0.5477
    wave_02 = 0.2175*0.2175
    dwave = 0.0350
    Eb = 0.85 - 1.9*dust_index
    top = wave*dwave*wave*dwave
    low = (wave*wave - wave_02)*(wave*wave - wave_02)
    D_lambda = Eb*top/(low + top)

    A_lambda = AV*(k_lambda + D_lambda)*np.power(wave/wave_V, dust_index)/4.05

    return A_lambda

def unresolved_dust_birth_cloud(AV, wave, dust_index_bc=-0.7):
    wave_V = 5477.0
    A_lambda = AV*np.power(wave/wave_V, dust_index_bc)

    return A_lambda


def calc_csp_fluxes_modified_Cal20_with_unresbc_detailed(sp=None, z=0.001, filters=[], pix_area_kpc2=1.0, stars_age=[], stars_zsol=[], stars_mass=[], 
    stars_coords_z=[], gas_mass_H=[], gas_coords_z=[], gas_mass=[], gas_zsol=[], imf_type=1, add_neb_emission=True, 
    mean_dust_AV_unres=0.3, dust_index=0.0, dust_index_bc=-0.7, cosmo=None, filter_transmission={}): 
    # filter_transmission: [filter string]['wave' or 'trans']

    array_spec = []
    array_spec_dust = []
    array_AV = []
    array_tauV = []

    for ii in range(len(stars_age)):
        logzsol = np.log10(stars_zsol[ii])
        sp.params["logzsol"] = logzsol   
        sp.params['gas_logz'] = logzsol
        sp.params['tage'] = stars_age[ii]

        wave, spec = sp.get_spectrum(peraa=True, tage=stars_age[ii])        # spectrum in L_sun/AA

        # calculate Hydrogen column density
        idxg = np.where(gas_coords_z < stars_coords_z[ii])[0]        # look for gas in front of the stellar particle
        if len(idxg) > 0:
            temp_mw_gas_zsol = np.nansum(gas_mass[idxg]*gas_zsol[idxg])/np.nansum(gas_mass[idxg])
            nH = np.nansum(gas_mass_H[idxg])*1.247914e+14/pix_area_kpc2      # number of hydrogen atom per cm^2
            tauV = tau_dust_given_z(z)*temp_mw_gas_zsol*nH/2.1e+21
            dust_AV = -2.5*np.log10((1.0 - np.exp(-1.0*tauV))/tauV)

            if np.isnan(dust_AV)==True or dust_AV==0.0:
                spec_dust = spec
            else:
                # attenuation by resolved dust in the diffuse ISM
                A_lambda = modified_calzetti_dust_curve(dust_AV, wave, dust_index=dust_index)
                spec_dust = spec*np.power(10.0, -0.4*A_lambda)
                array_tauV.append(tauV)
                array_AV.append(dust_AV)

        else:
            spec_dust = spec

        if stars_age[ii] <= 0.01:    # 10 Myr 
            # attenuation by unresolved dust in the birth cloud
            A_lambda = unresolved_dust_birth_cloud(mean_dust_AV_unres, wave, dust_index_bc=dust_index_bc)
            spec_dust = spec_dust*np.power(10.0, -0.4*A_lambda)
 
        norm = stars_mass[ii]/sp.stellar_mass

        if len(np.asarray(spec_dust).shape) == 1:
            array_spec.append(spec*norm)
            array_spec_dust.append(spec_dust*norm)

    # average AV:
    mean_AV = np.nanmean(np.asarray(array_AV))
    mean_tauV = np.nanmean(np.asarray(array_tauV))

    redshift_flux, redshift_flux_dust = [], []
    
    if len(array_spec) > 0:
        spec_lum = np.nansum(array_spec, axis=0)
        spec_lum_dust = np.nansum(array_spec_dust, axis=0)

        # redshifting:
        spec_wave, spec_flux = cosmo_redshifting(wave, spec_lum, z, cosmo=None)   # in erg/s/cm^2/Ang.
        spec_wave, spec_flux_dust = cosmo_redshifting(wave, spec_lum_dust, z, cosmo=None)    # in erg/s/cm^2/Ang.

        # IGM absorption:
        trans = igm_att_madau(spec_wave, z)

        # filtering
        nbands = len(filters)
        redshift_flux = np.zeros(nbands)
        redshift_flux_dust = np.zeros(nbands)

        for i_band in range(nbands):
            redshift_flux[i_band] = filtering(spec_wave, spec_flux*trans, filter_transmission[filters[i_band]]['wave'], filter_transmission[filters[i_band]]['trans'])
            redshift_flux_dust[i_band] = filtering(spec_wave, spec_flux_dust*trans, filter_transmission[filters[i_band]]['wave'], filter_transmission[filters[i_band]]['trans'])

    return redshift_flux, redshift_flux_dust, mean_AV, mean_tauV
