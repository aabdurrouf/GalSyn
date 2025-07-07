import h5py
import os, sys 
import numpy as np 
from astropy.io import fits
import matplotlib.pyplot as plt
from .simutils_tng import *
from .utils import *
from operator import itemgetter
from scipy.interpolate import interp1d
from joblib import Parallel, delayed
import multiprocessing

from tqdm.auto import tqdm # Import tqdm for general use
from tqdm_joblib import tqdm_joblib # Specific tqdm integration for joblib

import fsps
sp_instance = None  # Global variable to avoid reloading fsps per model

def generate_images_single_cpu(hdf5_file, snap_number, subhalo_id, filters, filter_transmission={}, filter_wave_eff={}, projection='yxz', z=None, dim_kpc=None, 
                    pix_arcsec=0.02, flux_unit='MJy/sr', cosmo=None, h=0.704, imf_type=1, add_neb_emission=True, name_out_props=None, name_out_img=None):
    # filter_transmission: [filter string]['wave' or 'trans']

    if cosmo is None:
        from astropy.cosmology import Planck15 as cosmo

    snap_z = get_snap_z(snap_number)
    if z is not None:
        snap_z = z

    snap_a = 1.0/(1.0 + snap_z)
    print ('Redshift: %lf' % snap_z)

    pix_kpc = angular_to_physical(snap_z, pix_arcsec)

    pix_area_kpc2 = pix_kpc*pix_kpc
    print ('pixel size: %lf kpc' % pix_kpc)

    f = h5py.File(hdf5_file,'r')

    stars_form_a = f['PartType4']['GFM_StellarFormationTime'][:]
    stars_form_z = (1.0/stars_form_a) - 1.0

    stars_init_mass = f['PartType4']['GFM_InitialMass'][:]*1e+10/h
    stars_mass = f['PartType4']['Masses'][:]*1e+10/h
    stars_zsol = f['PartType4']['GFM_Metallicity'][:]/0.0127         # in solar metallicity

    coords = f['PartType4']['Coordinates'][:]
    coords_x = coords[:,0]*snap_a/h             # x in kpc
    coords_y = coords[:,1]*snap_a/h             # y in kpc
    coords_z = coords[:,2]*snap_a/h             # z in kpc

    snap_univ_age = cosmo.age(snap_z).value
    stars_form_age_univ = interp_age_univ_from_z(stars_form_z)
    stars_age = snap_univ_age - stars_form_age_univ                 # age in Gyr
    # select only stellar particles
    idx = np.where((stars_form_a>0) & (stars_age>=0))[0]
    stars_form_a = stars_form_a[idx]
    stars_form_z = stars_form_z[idx]
    stars_init_mass = stars_init_mass[idx]
    stars_mass = stars_mass[idx]
    stars_zsol = stars_zsol[idx]
    coords_x0 = coords_x[idx]
    coords_y0 = coords_y[idx]
    coords_z0 = coords_z[idx]
    stars_age = stars_age[idx]

    # read gas particles data
    gas_mass = f['PartType0']['Masses'][:]*1e+10/h
    gas_zsol = f['PartType0']['GFM_Metallicity'][:]/0.0127  # in solar metallicity
    gas_coords = f['PartType0']['Coordinates'][:]
    gas_coords_x0 = gas_coords[:,0]*snap_a/h                 # in kpc
    gas_coords_y0 = gas_coords[:,1]*snap_a/h
    gas_coords_z0 = gas_coords[:,2]*snap_a/h
    gas_sfr_inst = f['PartType0']['StarFormationRate'][:]   # in Msun/yr
    XH = 0.76                                    # the hydrogen mass fraction  
    gas_mass_H = gas_mass*XH                     # mass of hydrogen 
    # calculate gas temperature
    u = f['PartType0']['InternalEnergy'][:]      #  the Internal Energy
    Xe = f['PartType0']['ElectronAbundance'][:]  # xe (=ne/nH)  the electron abundance                
    gamma = 5.0/3.0          # the adiabatic index
    KB = 1.3807e-16          # the Boltzmann constant in CGS units  [cm^2 g s^-2 K^-1]
    mp = 1.6726e-24          # the proton mass  [g]
    little_h = 0.704         # NOTE: 0.6775 for all TNG simulations
    mu = (4*mp)/(1+3*XH+4*XH*Xe)
    gas_log_temp = np.log10((gamma-1.0)*(u/KB)*mu*1e+10)  # log temperature in Kelvin

    f.close()

    if dim_kpc is None:
        dim_kpc = assign_cutout_size(snap_z, np.log10(np.nansum(stars_mass)))

    # note: new z coordinates are modified such that values increases as distance increases from the observer
    if projection == 'yxz':
        if max(coords_z0)>=max(gas_coords_z0):
            maxz = max(coords_z0)
        else:
            maxz = max(gas_coords_z0)

        coords_x, coords_y, coords_z = coords_x0, coords_y0, np.absolute(coords_z0-maxz)
        gas_coords_x, gas_coords_y, gas_coords_z = gas_coords_x0, gas_coords_y0, np.absolute(gas_coords_z0-maxz)

    elif projection == 'xyz':
        coords_x, coords_y, coords_z = coords_y0, coords_x0, coords_z0
        gas_coords_x, gas_coords_y, gas_coords_z = gas_coords_y0, gas_coords_x0, gas_coords_z0

    elif projection == 'zyx':
        if max(coords_x0)>=max(gas_coords_x0):
            maxz = max(coords_x0)
        else:
            maxz = max(gas_coords_x0)

        coords_x, coords_y, coords_z = coords_y0, coords_z0, np.absolute(coords_x0-maxz)
        gas_coords_x, gas_coords_y, gas_coords_z = gas_coords_y0, gas_coords_z0, np.absolute(gas_coords_x0-maxz)

    elif projection == 'yzx':
        coords_x, coords_y, coords_z = coords_z0, coords_y0, coords_x0 
        gas_coords_x, gas_coords_y, gas_coords_z = gas_coords_z0, gas_coords_y0, gas_coords_x0

    else:
        print ('projection is not recognized!')
        sys.exit()

    #=> estimate central coordinate and determine size
    cent_x, cent_y = np.nanmedian(coords_x), np.nanmedian(coords_y)
    xmin, xmax, ymin, ymax = cent_x - 0.6*dim_kpc, cent_x + 0.6*dim_kpc, cent_y - 0.6*dim_kpc, cent_y + 0.6*dim_kpc
    nbins_x, nbins_y = int(np.rint((xmax-xmin)/pix_kpc)), int(np.rint((ymax-ymin)/pix_kpc))

    map2d_tot_mass, xedges, yedges, image = plt.hist2d(coords_x, coords_y, range=[[xmin,xmax], [ymin,ymax]],
                                                        weights=stars_mass, bins=[nbins_x,nbins_y])
    plt.close()

    pix_rows, pix_cols = np.where(map2d_tot_mass > 0.0)
    arr_mass = map2d_tot_mass[pix_rows,pix_cols]
    idx0, val = max(enumerate(arr_mass), key=itemgetter(1))

    # central coordinate:
    cent_x, cent_y = 0.5*(xedges[pix_cols[idx0]] + xedges[pix_cols[idx0]+1]), 0.5*(yedges[pix_rows[idx0]] + yedges[pix_rows[idx0]+1])
    print ('Central coordinate: x=%lf y=%lf' % (cent_x,cent_y))
    print ('Cutout size: %d x %d kpc' % (dim_kpc,dim_kpc))

    # calculate maps with the new defined cutout size
    xmin, xmax, ymin, ymax = cent_x-0.5*dim_kpc, cent_x+0.5*dim_kpc, cent_y-0.5*dim_kpc, cent_y+0.5*dim_kpc
    nbins_x, nbins_y = int(np.rint((xmax-xmin)/pix_kpc)), int(np.rint((ymax-ymin)/pix_kpc))

    map2d_tot_mass, xedges, yedges, image = plt.hist2d(coords_x, coords_y, range=[[xmin,xmax], [ymin,ymax]],
                                                        weights=stars_mass, bins=[nbins_x,nbins_y])
    plt.close()

    # calculate average AV of the diffuse ISM over the whole galaxy's region
    idxg1 = np.where((gas_coords_x>=xmin) & (gas_coords_x<xmax) & (gas_coords_y>=ymin) & (gas_coords_y<ymax))[0]
    idxg2 = np.where((gas_sfr_inst[idxg1]>0.0) | (gas_log_temp[idxg1]<4.0))[0]
    idxg = idxg1[idxg2]

    # calculate Hydrogen column density
    temp_mw_gas_zsol = np.nansum(gas_mass[idxg]*gas_zsol[idxg])/np.nansum(gas_mass[idxg])
    nH = np.nansum(gas_mass_H[idxg])*1.247914e+14/dim_kpc/dim_kpc      # number of hydrogen atom per cm^2
    mean_tauV_res = tau_dust_given_z(snap_z)*temp_mw_gas_zsol*nH/2.1e+21 
    mean_AV_unres = -2.5*np.log10(np.exp(-2.0*mean_tauV_res))          # assumed twice of average tauV resolved (Vogelsberger+20)
    if np.isnan(mean_tauV_res)==True or np.isinf(mean_tauV_res)==True:
        mean_tauV_res, mean_AV_unres = 0.0, 0.0
    print ('mean_tauV_res=%lf mean_AV_unres=%lf' % (mean_tauV_res,mean_AV_unres))

    # initialize FSPS
    sp = fsps.StellarPopulation(zcontinuous=1, imf_type=imf_type, add_neb_emission=add_neb_emission)
    sp.params['imf_type'] = imf_type
    sp.params["add_dust_emission"] = False
    sp.params["add_neb_emission"] = add_neb_emission
    sp.params['gas_logu'] = -2.0
    sp.params["fagn"] = 0
    sp.params["sfh"] = 0   # SSP
    sp.params["dust_type"] = 2
    sp.params["dust1"] = 0.0
    sp.params["dust2"] = 0.0   # optical depth

    nbands = len(filters)

    map_mw_age = np.zeros((nbins_y,nbins_x))
    map_stars_mw_zsol = np.zeros((nbins_y,nbins_x))
    map_stars_mass = np.zeros((nbins_y,nbins_x))
    map_sfr_100 = np.zeros((nbins_y,nbins_x))
    map_sfr_10 = np.zeros((nbins_y,nbins_x))
    map_sfr_30 = np.zeros((nbins_y,nbins_x))

    map_gas_mw_zsol = np.zeros((nbins_y,nbins_x))
    map_gas_mass = np.zeros((nbins_y,nbins_x))
    map_sfr_inst = np.zeros((nbins_y,nbins_x))

    map_dust_mean_tauV = np.zeros((nbins_y,nbins_x))
    map_dust_mean_AV = np.zeros((nbins_y,nbins_x))

    map_flux = np.zeros((nbins_y,nbins_x,nbands))
    map_flux_dust = np.zeros((nbins_y,nbins_x,nbands))

    dust_index_bc = -0.7 

    for ii in range(len(yedges)-1):
        for jj in range(len(xedges)-1):

            ##=> stellar particles
            idxs1 = np.where((coords_x>=xedges[jj]) & (coords_x<xedges[jj+1]) & (coords_y>=yedges[ii]) & (coords_y<yedges[ii+1]))[0]
            idxs2 = np.where((np.isnan(stars_mass[idxs1])==False) & (np.isnan(stars_age[idxs1])==False) & (np.isnan(stars_zsol[idxs1])==False))[0]
            idxs = idxs1[idxs2]
            
            map_stars_mass[ii][jj] = np.nansum(stars_mass[idxs])
            map_mw_age[ii][jj] = np.nansum(stars_mass[idxs]*stars_age[idxs])/map_stars_mass[ii][jj]
            map_stars_mw_zsol[ii][jj] = np.nansum(stars_mass[idxs]*stars_zsol[idxs])/map_stars_mass[ii][jj]

            idxs3 = np.where((np.isnan(stars_init_mass[idxs1])==False) & (stars_age[idxs1]<=0.1))[0]
            map_sfr_100[ii][jj] = np.nansum(stars_init_mass[idxs1[idxs3]])/0.1/1e+9      # SFR in unit of Msun/yr

            idxs3 = np.where((np.isnan(stars_init_mass[idxs1])==False) & (stars_age[idxs1]<=0.03))[0]
            map_sfr_30[ii][jj] = np.nansum(stars_init_mass[idxs1[idxs3]])/0.03/1e+9      # SFR in unit of Msun/yr

            idxs3 = np.where((np.isnan(stars_init_mass[idxs1])==False) & (stars_age[idxs1]<=0.01))[0]
            map_sfr_10[ii][jj] = np.nansum(stars_init_mass[idxs1[idxs3]])/0.01/1e+9      # SFR in unit of Msun/yr

            ##=> gas particles
            idxg1 = np.where((gas_coords_x>=xedges[jj]) & (gas_coords_x<xedges[jj+1]) & (gas_coords_y>=yedges[ii]) & (gas_coords_y<yedges[ii+1]))[0]
            idxg2 = np.where(np.isnan(gas_mass[idxg1])==False)[0]
            map_gas_mass[ii][jj] = np.nansum(gas_mass[idxg1[idxg2]])
            map_sfr_inst[ii][jj] = np.nansum(gas_sfr_inst[idxg1[idxg2]])

            idxg3 = np.where((np.isnan(gas_mass[idxg1])==False) & (np.isnan(gas_zsol[idxg1])==False))[0]
            map_gas_mw_zsol[ii][jj] = np.nansum(gas_mass[idxg1[idxg3]]*gas_zsol[idxg1[idxg3]])/map_gas_mass[ii][jj]

            # calculate dust optical depth: based on cold star-forming gas
            idxg2 = np.where((gas_sfr_inst[idxg1]>0.0) | (gas_log_temp[idxg1]<4.0))[0]
            idxg = idxg1[idxg2]

            ##=> get fluxes
            if len(idxs) > 0:

                dust_index = 0.0
                redshift_flux, redshift_flux_dust, mean_AV, mean_tauV = calc_csp_fluxes_modified_Cal20_with_unresbc_detailed(sp=sp, z=snap_z, 
                                                    filters=filters, pix_area_kpc2=pix_area_kpc2, stars_age=stars_age[idxs], stars_zsol=stars_zsol[idxs], stars_mass=stars_mass[idxs], 
                                                    stars_coords_z=coords_z[idxs], gas_mass_H=gas_mass_H[idxg], gas_coords_z=gas_coords_z[idxg], gas_mass=gas_mass[idxg], 
                                                    gas_zsol=gas_zsol[idxg], imf_type=imf_type, add_neb_emission=add_neb_emission, mean_dust_AV_unres=mean_AV_unres, 
                                                    dust_index=dust_index, dust_index_bc=dust_index_bc, cosmo=cosmo, filter_transmission=filter_transmission)

                if len(redshift_flux) > 0:
                    map_flux[ii][jj] = redshift_flux
                    map_flux_dust[ii][jj] = redshift_flux_dust

                    map_dust_mean_tauV[ii][jj] = mean_tauV
                    map_dust_mean_AV[ii][jj] = mean_AV

            sys.stdout.write('\r')
            sys.stdout.write('progress: x (%d of %d) and y (%d of %d)' % (jj,len(xedges)-1,ii,len(yedges)-1))
            sys.stdout.flush()

    # convert flux units
    for i_band in range(nbands):
        map_flux[:,:,i_band] = convert_flux_map(map_flux[:,:,i_band], filter_wave_eff[filters[i_band]], to_unit=flux_unit, pixel_scale_arcsec=pix_arcsec)
        map_flux_dust[:,:,i_band] = convert_flux_map(map_flux_dust[:,:,i_band], filter_wave_eff[filters[i_band]], to_unit=flux_unit, pixel_scale_arcsec=pix_arcsec)
    
    ## save results into FITS file
    # properties
    hdul = fits.HDUList()
    hdr = fits.Header()
    hdr['cent_x'] = cent_x
    hdr['cent_y'] = cent_y
    hdr['pix_kpc'] = pix_kpc
    hdr['tauV_unres'] = mean_tauV_res
    hdr['AV_unres'] = mean_AV_unres
    hdr['subhaloid'] = subhalo_id
    hdr['z'] = snap_z

    hdul.append(fits.ImageHDU(data=map_stars_mass, header=hdr, name='stars_mass'))
    hdul.append(fits.ImageHDU(data=map_stars_mw_zsol, name='stars_zsol'))
    hdul.append(fits.ImageHDU(data=map_mw_age, name='stars_age'))
    hdul.append(fits.ImageHDU(data=map_gas_mass, name='gas_mass'))
    hdul.append(fits.ImageHDU(data=map_gas_mw_zsol, name='gas_zsol'))
    hdul.append(fits.ImageHDU(data=map_sfr_inst, name='sfr_inst'))
    hdul.append(fits.ImageHDU(data=map_dust_mean_tauV, name='mean_tauV'))
    hdul.append(fits.ImageHDU(data=map_dust_mean_AV, name='mean_AV'))
    hdul.append(fits.ImageHDU(data=map_sfr_10, name='sfr10'))
    hdul.append(fits.ImageHDU(data=map_sfr_30, name='sfr30'))
    hdul.append(fits.ImageHDU(data=map_sfr_100, name='sfr100'))

    if name_out_props is None:
        name_out_props='maps_props.fits'
    hdul.writeto(name_out_props, overwrite=True)

    # images
    map_flux_trans = np.transpose(map_flux, axes=(2,0,1))
    map_flux_dust_trans = np.transpose(map_flux_dust, axes=(2,0,1))

    hdul = fits.HDUList()
    hdr = fits.Header()
    hdr['cent_x'] = cent_x
    hdr['cent_y'] = cent_y
    hdr['pix_kpc'] = pix_kpc
    hdr['tauV_unres'] = mean_tauV_res
    hdr['AV_unres'] = mean_AV_unres
    hdr['subhaloid'] = subhalo_id
    hdr['z'] = snap_z

    hdul.append(fits.ImageHDU(data=map_flux_trans[0], header=hdr, name='nodust_'+filters[0]))
    hdul.append(fits.ImageHDU(data=map_flux_dust_trans[0], name='dust_'+filters[0]))
    
    # multiband flux maps
    for bb in range(1,nbands):
        hdul.append(fits.ImageHDU(data=map_flux_trans[bb], name='nodust_'+filters[bb]))
        hdul.append(fits.ImageHDU(data=map_flux_dust_trans[bb], name='dust_'+filters[bb]))

    if name_out_img is None:
        name_out_img='maps_images.fits'
    hdul.writeto(name_out_img, overwrite=True)


def init_worker():
    global sp_instance
    sp_instance = fsps.StellarPopulation(zcontinuous=1, add_neb_emission=1)

def _process_pixel_data(ii, jj, xedges, yedges, coords_x, coords_y, stars_mass,
                        stars_age, stars_zsol, stars_init_mass, coords_z,
                        gas_coords_x, gas_coords_y, gas_mass, gas_sfr_inst,
                        gas_zsol, gas_log_temp, gas_mass_H, gas_coords_z, 
                        pix_area_kpc2, mean_AV_unres, filters, imf_type,
                        snap_z, dust_index_bc, add_neb_emission, cosmo, 
                        filter_transmission):
    """
    Worker function to process calculations for a single pixel (ii, jj).
    This function will be executed in parallel for each pixel.
    """

    global sp_instance # Declare that we're using the global variable

    sp_instance.params['imf_type'] = imf_type
    sp_instance.params["add_dust_emission"] = False
    sp_instance.params["add_neb_emission"] = add_neb_emission
    sp_instance.params['gas_logu'] = -2.0
    sp_instance.params["fagn"] = 0
    sp_instance.params["sfh"] = 0   # SSP
    sp_instance.params["dust_type"] = 2
    sp_instance.params["dust1"] = 0.0
    sp_instance.params["dust2"] = 0.0   # optical depth

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

    ##=> stellar particles
    idxs1 = np.where((coords_x >= xedges[jj]) & (coords_x < xedges[jj+1]) &
                     (coords_y >= yedges[ii]) & (coords_y < yedges[ii+1]))[0]
    idxs2 = np.where((np.isnan(stars_mass[idxs1]) == False) &
                     (np.isnan(stars_age[idxs1]) == False) &
                     (np.isnan(stars_zsol[idxs1]) == False))[0]
    idxs = idxs1[idxs2]

    current_stars_mass_sum = np.nansum(stars_mass[idxs])
    pixel_results['map_stars_mass'] = current_stars_mass_sum

    if current_stars_mass_sum > 0:
        pixel_results['map_mw_age'] = np.nansum(stars_mass[idxs] * stars_age[idxs]) / current_stars_mass_sum
        pixel_results['map_stars_mw_zsol'] = np.nansum(stars_mass[idxs] * stars_zsol[idxs]) / current_stars_mass_sum
    else:
        pixel_results['map_mw_age'] = np.nan
        pixel_results['map_stars_mw_zsol'] = np.nan

    idxs3_100 = np.where((np.isnan(stars_init_mass[idxs1]) == False) & (stars_age[idxs1] <= 0.1))[0]
    pixel_results['map_sfr_100'] = np.nansum(stars_init_mass[idxs1[idxs3_100]]) / 0.1 / 1e+9

    idxs3_30 = np.where((np.isnan(stars_init_mass[idxs1]) == False) & (stars_age[idxs1] <= 0.03))[0]
    pixel_results['map_sfr_30'] = np.nansum(stars_init_mass[idxs1[idxs3_30]]) / 0.03 / 1e+9

    idxs3_10 = np.where((np.isnan(stars_init_mass[idxs1]) == False) & (stars_age[idxs1] <= 0.01))[0]
    pixel_results['map_sfr_10'] = np.nansum(stars_init_mass[idxs1[idxs3_10]]) / 0.01 / 1e+9

    ##=> gas particles
    idxg1 = np.where((gas_coords_x >= xedges[jj]) & (gas_coords_x < xedges[jj+1]) &
                     (gas_coords_y >= yedges[ii]) & (gas_coords_y < yedges[ii+1]))[0]
    idxg2_mass = np.where(np.isnan(gas_mass[idxg1]) == False)[0]
    current_gas_mass_sum = np.nansum(gas_mass[idxg1[idxg2_mass]])
    pixel_results['map_gas_mass'] = current_gas_mass_sum
    pixel_results['map_sfr_inst'] = np.nansum(gas_sfr_inst[idxg1[idxg2_mass]])

    idxg3_zsol = np.where((np.isnan(gas_mass[idxg1]) == False) & (np.isnan(gas_zsol[idxg1]) == False))[0]
    if current_gas_mass_sum > 0:
        pixel_results['map_gas_mw_zsol'] = np.nansum(gas_mass[idxg1[idxg3_zsol]] * gas_zsol[idxg1[idxg3_zsol]]) / current_gas_mass_sum
    else:
        pixel_results['map_gas_mw_zsol'] = np.nan

    # calculate dust optical depth: based on cold star-forming gas
    idxg2_dust = np.where((gas_sfr_inst[idxg1] > 0.0) | (gas_log_temp[idxg1] < 4.0))[0]
    idxg_dust = idxg1[idxg2_dust]

    ##=> get fluxes
    if len(idxs) > 0:

        dust_index = 0.0
        redshift_flux, redshift_flux_dust, mean_AV, mean_tauV = \
            calc_csp_fluxes_modified_Cal20_with_unresbc_detailed(sp=sp_instance, z=snap_z, filters=filters,
                                                                 pix_area_kpc2=pix_area_kpc2,
                                                                 stars_age=stars_age[idxs],
                                                                 stars_zsol=stars_zsol[idxs],
                                                                 stars_mass=stars_mass[idxs],
                                                                 stars_coords_z=coords_z[idxs],
                                                                 gas_mass_H=gas_mass_H[idxg_dust],
                                                                 gas_coords_z=gas_coords_z[idxg_dust],
                                                                 gas_mass=gas_mass[idxg_dust],
                                                                 gas_zsol=gas_zsol[idxg_dust],
                                                                 imf_type=imf_type,
                                                                 mean_dust_AV_unres=mean_AV_unres,
                                                                 dust_index=dust_index,
                                                                 dust_index_bc=dust_index_bc,
                                                                 cosmo=cosmo, 
                                                                 filter_transmission=filter_transmission)

        if len(redshift_flux) > 0:
            pixel_results['map_flux'] = redshift_flux
            pixel_results['map_flux_dust'] = redshift_flux_dust

            pixel_results['map_dust_mean_tauV'] = mean_tauV
            pixel_results['map_dust_mean_AV'] = mean_AV

    return ii, jj, pixel_results


def generate_images_parallel(hdf5_file, snap_number, subhalo_id, filters, filter_transmission={}, filter_wave_eff={}, 
                            projection='yxz', z=None, dim_kpc=None, pix_arcsec=0.02, flux_unit='MJy/sr', cosmo=None, h=0.704, 
                            imf_type=1, add_neb_emission=True, name_out_img=None, n_jobs=-1):
    """
    Generates astrophysical images from HDF5 simulation data with parallelized pixel calculations.

    Parameters:
        hdf5_file (str): Path to the HDF5 simulation file.
        snap_number (int): Snapshot number.
        subhalo_id (int): Subhalo ID.
        filters (list): List of photometric filters.
        filter_transmission: [filter string]['wave' or 'trans']
        filter_wave_eff: [filter string]
        projection (str, optional): Projection axis ('yxz', 'xyz', 'zyx', 'yzx'). Defaults to 'yxz'.
        z (float, optional): Redshift. If None, derived from snap_number. Defaults to None.
        dim_kpc (float, optional): Dimension of the image in kpc. If None, assigned automatically. Defaults to None.
        pix_arcsec (float, optional): Pixel size in arcseconds. Defaults to 0.02.
        flux_unit (string, optional): Desired flux unit for the generated images. Options are: 'MJy/sr', 'nJy', 'AB magnitude', or 'erg/s/cm2/A'. Default to 'MJy/sr'.
        cosmo (astropy.cosmology, optional): Astropy Cosmology object. If None, Planck15 is used. Defaults to None.
        h (float, optional): Hubble constant in units of 100 km/s/Mpc. Defaults to 0.704.
        imf_type (int, optional): IMF type for FSPS. Defaults to 1.
        add_neb_emission (bool, optional): Whether to add nebular emission in FSPS. Defaults to True.
        name_out_img (str, optional): Output file name for images. Defaults to None.
        n_jobs (int, optional): Number of CPU cores to use for parallel processing. 
                                -1 means use all available cores. Defaults to -1.
    """

    sp_instance = fsps.StellarPopulation(zcontinuous=1, imf_type=imf_type, add_neb_emission=add_neb_emission)

    if cosmo is None:
        from astropy.cosmology import Planck15 as cosmo


    # --- Data Loading and Initial Calculations (Sequential) ---
    snap_z = get_snap_z(snap_number)
    if z is not None:
        snap_z = z

    snap_a = 1.0/(1.0 + snap_z)
    print ('Redshift: %lf' % snap_z)

    pix_kpc = angular_to_physical(snap_z, pix_arcsec)

    pix_area_kpc2 = pix_kpc*pix_kpc
    print ('pixel size: %lf arcsec / %lf kpc' % (pix_arcsec,pix_kpc))

    f = h5py.File(hdf5_file,'r')

    stars_form_a = f['PartType4']['GFM_StellarFormationTime'][:]
    stars_form_z = (1.0/stars_form_a) - 1.0

    stars_init_mass = f['PartType4']['GFM_InitialMass'][:]*1e+10/h
    stars_mass = f['PartType4']['Masses'][:]*1e+10/h
    stars_zsol = f['PartType4']['GFM_Metallicity'][:]/0.0127         # in solar metallicity

    coords = f['PartType4']['Coordinates'][:]
    coords_x = coords[:,0]*snap_a/h             # x in kpc
    coords_y = coords[:,1]*snap_a/h             # y in kpc
    coords_z = coords[:,2]*snap_a/h             # z in kpc

    snap_univ_age = cosmo.age(snap_z).value
    stars_form_age_univ = interp_age_univ_from_z(stars_form_z)
    stars_age = snap_univ_age - stars_form_age_univ                 # age in Gyr
    # select only stellar particles
    idx = np.where((stars_form_a>0) & (stars_age>=0))[0]
    stars_form_a = stars_form_a[idx]
    stars_form_z = stars_form_z[idx]
    stars_init_mass = stars_init_mass[idx]
    stars_mass = stars_mass[idx]
    stars_zsol = stars_zsol[idx]
    coords_x0 = coords_x[idx]
    coords_y0 = coords_y[idx]
    coords_z0 = coords_z[idx]
    stars_age = stars_age[idx]

    # read gas particles data
    gas_mass = f['PartType0']['Masses'][:]*1e+10/h
    gas_zsol = f['PartType0']['GFM_Metallicity'][:]/0.0127  # in solar metallicity
    gas_coords = f['PartType0']['Coordinates'][:]
    gas_coords_x0 = gas_coords[:,0]*snap_a/h                 # in kpc
    gas_coords_y0 = gas_coords[:,1]*snap_a/h
    gas_coords_z0 = gas_coords[:,2]*snap_a/h
    gas_sfr_inst = f['PartType0']['StarFormationRate'][:]   # in Msun/yr
    XH = 0.76                                    # the hydrogen mass fraction  
    gas_mass_H = gas_mass*XH                     # mass of hydrogen 
    # calculate gas temperature
    u = f['PartType0']['InternalEnergy'][:]      #  the Internal Energy
    Xe = f['PartType0']['ElectronAbundance'][:]  # xe (=ne/nH)  the electron abundance                
    gamma = 5.0/3.0          # the adiabatic index
    KB = 1.3807e-16          # the Boltzmann constant in CGS units  [cm^2 g s^-2 K^-1]
    mp = 1.6726e-24          # the proton mass  [g]
    little_h = 0.704         # NOTE: 0.6775 for all TNG simulations
    mu = (4*mp)/(1+3*XH+4*XH*Xe)
    gas_log_temp = np.log10((gamma-1.0)*(u/KB)*mu*1e+10)  # log temperature in Kelvin

    f.close()

    if dim_kpc is None:
        dim_kpc = assign_cutout_size(snap_z, np.log10(np.nansum(stars_mass)))

    # note: new z coordinates are modified such that values increases as distance increases from the observer
    if projection == 'yxz':
        if coords_z0.size > 0 and gas_coords_z0.size > 0: # Check if arrays are not empty
            maxz = max(coords_z0.max(), gas_coords_z0.max())
        elif coords_z0.size > 0:
            maxz = coords_z0.max()
        elif gas_coords_z0.size > 0:
            maxz = gas_coords_z0.max()
        else:
            maxz = 0.0 # Default if both are empty

        coords_x, coords_y, coords_z = coords_x0, coords_y0, np.absolute(coords_z0-maxz)
        gas_coords_x, gas_coords_y, gas_coords_z = gas_coords_x0, gas_coords_y0, np.absolute(gas_coords_z0-maxz)

    elif projection == 'xyz':
        coords_x, coords_y, coords_z = coords_y0, coords_x0, coords_z0
        gas_coords_x, gas_coords_y, gas_coords_z = gas_coords_y0, gas_coords_x0, gas_coords_z0

    elif projection == 'zyx':
        if coords_x0.size > 0 and gas_coords_x0.size > 0:
            maxz = max(coords_x0.max(), gas_coords_x0.max())
        elif coords_x0.size > 0:
            maxz = coords_x0.max()
        elif gas_coords_x0.size > 0:
            maxz = gas_coords_x0.max()
        else:
            maxz = 0.0

        coords_x, coords_y, coords_z = coords_y0, coords_z0, np.absolute(coords_x0-maxz)
        gas_coords_x, gas_coords_y, gas_coords_z = gas_coords_y0, gas_coords_z0, np.absolute(gas_coords_x0-maxz)

    elif projection == 'yzx':
        coords_x, coords_y, coords_z = coords_z0, coords_y0, coords_x0 
        gas_coords_x, gas_coords_y, gas_coords_z = gas_coords_z0, gas_coords_y0, gas_coords_x0

    else:
        print ('projection is not recognized!')
        sys.exit()

    #=> estimate central coordinate and determine size
    cent_x, cent_y = np.nanmedian(coords_x), np.nanmedian(coords_y)
    xmin, xmax, ymin, ymax = cent_x - 0.6*dim_kpc, cent_x + 0.6*dim_kpc, cent_y - 0.6*dim_kpc, cent_y + 0.6*dim_kpc
    nbins_x_initial, nbins_y_initial = int(np.rint((xmax-xmin)/pix_kpc)), int(np.rint((ymax-ymin)/pix_kpc))

    # Use a specific figure to avoid potential conflicts with parallel workers
    fig_initial, ax_initial = plt.subplots()
    map2d_tot_mass, xedges_initial, yedges_initial, image = ax_initial.hist2d(coords_x, coords_y, range=[[xmin,xmax], [ymin,ymax]],
                                                        weights=stars_mass, bins=[nbins_x_initial,nbins_y_initial])
    plt.close(fig_initial) # Close the figure to free up memory

    pix_rows, pix_cols = np.where(map2d_tot_mass > 0.0)
    arr_mass = map2d_tot_mass[pix_rows,pix_cols]
    
    if len(arr_mass) == 0:
        print("Warning: No stellar particles found in the initial map. Cannot determine central coordinate accurately.")
        # Using the center of the initial search box as a fallback
        cent_x, cent_y = xmin + 0.5 * (xmax - xmin), ymin + 0.5 * (ymax - ymin)
    else:
        idx0, val = max(enumerate(arr_mass), key=itemgetter(1))
        # central coordinate:
        cent_x, cent_y = (0.5*(xedges_initial[pix_cols[idx0]] + xedges_initial[pix_cols[idx0]+1]),
                          0.5*(yedges_initial[pix_rows[idx0]] + yedges_initial[pix_rows[idx0]+1]))
    
    print ('Central coordinate: x=%lf y=%lf' % (cent_x,cent_y))
    print ('Cutout size: %d x %d kpc' % (dim_kpc,dim_kpc))

    # calculate maps with the new defined cutout size
    xmin, xmax, ymin, ymax = cent_x-0.5*dim_kpc, cent_x+0.5*dim_kpc, cent_y-0.5*dim_kpc, cent_y+0.5*dim_kpc
    nbins_x, nbins_y = int(np.rint((xmax-xmin)/pix_kpc)), int(np.rint((ymax-ymin)/pix_kpc))

    fig_final, ax_final = plt.subplots()
    map2d_tot_mass, xedges, yedges, image = ax_final.hist2d(coords_x, coords_y, range=[[xmin,xmax], [ymin,ymax]],
                                                        weights=stars_mass, bins=[nbins_x,nbins_y])
    plt.close(fig_final) # Close the figure

    # calculate average AV of the diffuse ISM over the whole galaxy's region
    idxg1_global = np.where((gas_coords_x>=xmin) & (gas_coords_x<xmax) & (gas_coords_y>=ymin) & (gas_coords_y<ymax))[0]
    idxg2_global = np.where((gas_sfr_inst[idxg1_global]>0.0) | (gas_log_temp[idxg1_global]<4.0))[0]
    idxg_global = idxg1_global[idxg2_global]

    # calculate Hydrogen column density
    if np.nansum(gas_mass[idxg_global]) > 0:
        temp_mw_gas_zsol = np.nansum(gas_mass[idxg_global]*gas_zsol[idxg_global])/np.nansum(gas_mass[idxg_global])
    else:
        temp_mw_gas_zsol = 0.0 # Default if no gas mass
        
    nH = np.nansum(gas_mass_H[idxg_global])*1.247914e+14/dim_kpc/dim_kpc      # number of hydrogen atom per cm^2
    mean_tauV_res = tau_dust_given_z(snap_z)*temp_mw_gas_zsol*nH/2.1e+21 
    mean_AV_unres = -2.5*np.log10(np.exp(-2.0*mean_tauV_res))          # assumed twice of average tauV resolved (Vogelsberger+20)
    if np.isnan(mean_tauV_res)==True or np.isinf(mean_tauV_res)==True:
        mean_tauV_res, mean_AV_unres = 0.0, 0.0
    print ('mean_tauV_res=%lf mean_AV_unres=%lf' % (mean_tauV_res,mean_AV_unres))

    nbands = len(filters)

    # Initialize all result maps to zeros
    map_mw_age = np.zeros((nbins_y,nbins_x))
    map_stars_mw_zsol = np.zeros((nbins_y,nbins_x))
    map_stars_mass = np.zeros((nbins_y,nbins_x))
    map_sfr_100 = np.zeros((nbins_y,nbins_x))
    map_sfr_10 = np.zeros((nbins_y,nbins_x))
    map_sfr_30 = np.zeros((nbins_y,nbins_x))

    map_gas_mw_zsol = np.zeros((nbins_y,nbins_x))
    map_gas_mass = np.zeros((nbins_y,nbins_x))
    map_sfr_inst = np.zeros((nbins_y,nbins_x))

    map_dust_mean_tauV = np.zeros((nbins_y,nbins_x))
    map_dust_mean_AV = np.zeros((nbins_y,nbins_x))

    map_flux = np.zeros((nbins_y,nbins_x,nbands))
    map_flux_dust = np.zeros((nbins_y,nbins_x,nbands))

    dust_index_bc = -0.7 

    # --- Parallel Calculation Section ---
    tasks = []
    for ii in range(nbins_y): # Iterate over rows (y-axis)
        for jj in range(nbins_x): # Iterate over columns (x-axis)
            tasks.append((ii, jj, xedges, yedges, coords_x, coords_y, stars_mass,
                          stars_age, stars_zsol, stars_init_mass, coords_z,
                          gas_coords_x, gas_coords_y, gas_mass, gas_sfr_inst,
                          gas_zsol, gas_log_temp, gas_mass_H, gas_coords_z,
                          pix_area_kpc2, mean_AV_unres, filters, imf_type,
                          snap_z, dust_index_bc, add_neb_emission, cosmo, filter_transmission))

    # Determine the number of CPU cores to use
    num_cores = n_jobs
    if num_cores == -1:
        num_cores = multiprocessing.cpu_count() # Use all available cores

    #print(f"\nStarting parallel pixel processing on {num_cores} cores...")

    with tqdm_joblib(total=len(tasks), desc="Processing pixels") as progress_bar:
        results = Parallel(n_jobs=num_cores, verbose=0, initializer=init_worker)(
            delayed(_process_pixel_data)(*task_args) for task_args in tasks
        )
    #print("\nFinished parallel pixel processing.")

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

    #print("All calculations complete. Maps populated.")

    # convert flux units
    for i_band in range(nbands):
        map_flux[:,:,i_band] = convert_flux_map(map_flux[:,:,i_band], filter_wave_eff[filters[i_band]], to_unit=flux_unit, pixel_scale_arcsec=pix_arcsec)
        map_flux_dust[:,:,i_band] = convert_flux_map(map_flux_dust[:,:,i_band], filter_wave_eff[filters[i_band]], to_unit=flux_unit, pixel_scale_arcsec=pix_arcsec)


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
                prihdr['CRPIX1'] = nbins_x / 2.0 + 0.5 # Reference pixel X (center)
                prihdr['CRPIX2'] = nbins_y / 2.0 + 0.5 # Reference pixel Y (center)
                prihdr['CDELT1'] = pix_kpc # Pixel scale X (kpc/pixel)
                prihdr['CDELT2'] = pix_kpc # Pixel scale Y (kpc/pixel)
                prihdr['CUNIT1'] = 'kpc'
                prihdr['CUNIT2'] = 'kpc'
                prihdr['CRVAL1'] = cent_x # Value at reference pixel X (kpc)
                prihdr['CRVAL2'] = cent_y # Value at reference pixel Y (kpc)
                prihdr['CTYPE1'] = 'POS_X' # Example coordinate type
                prihdr['CTYPE2'] = 'POS_Y' # Example coordinate type
                prihdr['ORIGIN'] = 'generate_images_script'
                prihdr['REDSHIFT'] = snap_z
                prihdr['PROJ'] = projection
                prihdr['DIM_KPC'] = dim_kpc
                prihdr['PIX_KPC'] = pix_kpc
                prihdr['PIXSIZE'] = pix_arcsec
                prihdr['BUNIT'] = flux_unit

                primary_hdu = fits.PrimaryHDU(data=primary_data, header=prihdr)
                hdul.append(primary_hdu)

                # Add extensions for other bands (no dust)
                for i_band in range(nbands):
                    ext_hdr = fits.Header()
                    #ext_hdr['EXTNAME'] = f'FLUX_BAND_{i_band:02d}'
                    ext_hdr['EXTNAME'] = 'NODUST_'+filters[i_band].upper()
                    #ext_hdr['FILTER'] = filters[i_band] if i_band < len(filters) else 'N/A'
                    ext_hdr['FILTER'] = filters[i_band]
                    ext_hdr['COMMENT'] = f'Flux for filter: {filters[i_band]}'
                    ext_hdu = fits.ImageHDU(data=map_flux[:, :, i_band], header=ext_hdr)
                    hdul.append(ext_hdu)

                # Add extensions for other bands (with dust)
                for i_band in range(nbands):
                    ext_hdr = fits.Header()
                    #ext_hdr['EXTNAME'] = f'FLUX_DUST_BAND_{i_band:02d}'
                    ext_hdr['EXTNAME'] = 'DUST_'+filters[i_band].upper()
                    #ext_hdr['FILTER'] = filters[i_band] if i_band < len(filters) else 'N/A'
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
            print(f"Results saved to FITS file: {name_out_img}")

        except Exception as e:
            print(f"Error saving FITS file {name_out_img}: {e}")

