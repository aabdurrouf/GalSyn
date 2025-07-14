import sys
import numpy as np
from operator import itemgetter

def fsps_setup():
    from .config import IMF_TYPE, ADD_NEB_EMISSION, ADD_IGM_ABSORPTION, IGM_TYPE, DUST_INDEX_BC, GAS_LOGU, DUST_INDEX, T_ESC
    return IMF_TYPE, ADD_NEB_EMISSION, ADD_IGM_ABSORPTION, IGM_TYPE, DUST_INDEX_BC, GAS_LOGU, DUST_INDEX, T_ESC

def define_cosmo():
    from .config import COSMO, COSMO_LITTLE_H

    cosmo_name = COSMO.lower()  # Make case-insensitive

    if cosmo_name == "planck18":
        from astropy.cosmology import Planck18 as cosmo
    elif cosmo_name == "planck15":
        from astropy.cosmology import Planck15 as cosmo
    elif cosmo_name == "planck13":
        from astropy.cosmology import Planck13 as cosmo
    elif cosmo_name == "wmap5":
        from astropy.cosmology import WMAP5 as cosmo
    elif cosmo_name == "wmap7":
        from astropy.cosmology import WMAP7 as cosmo
    elif cosmo_name == "wmap9":
        from astropy.cosmology import WMAP9 as cosmo
    else:
        print("Selected cosmology is not recognized!")
        sys.exit()

    return cosmo, COSMO_LITTLE_H

global cosmo
cosmo, cosmo_h = define_cosmo()

def interp_age_univ_from_z(z):
    
    from scipy.interpolate import interp1d
    
    arr_z = np.linspace(0.0, 12.0, 100)
    age_univ = cosmo.age(arr_z).value

    f = interp1d(arr_z, age_univ, fill_value="extrapolate")
    return f(z)

def cosmo_redshifting(wave_rest, L_lambda_rest, z):
    """
    Performs cosmological redshifting of a spectrum.

    This function converts an emitted spectrum, provided in units of
    Solar luminosity per Angstrom, into an observed flux density in units
    of erg/s/cm^2/Angstrom, given a cosmological redshift.

    Parameters
    ----------
    wave_rest : `numpy.ndarray`
        Wavelengths of the emitted spectrum in Angstrom.
    L_lambda_rest : `numpy.ndarray` 
        Luminosity per unit wavelength of the emitted spectrum in Solar Luminosity per Angstrom ($L_{\odot}/\AA$).
    z : float
        The cosmological redshift (z) at which the spectrum is observed.

    Returns
    -------
    tuple of `astropy.units.Quantity`
        A tuple containing two astropy Quantity objects:
        - observed_wavelengths : Wavelengths in the observed frame, in Angstrom.
        - observed_flux_density : Flux density in the observed frame, in erg/s/cm^2/Angstrom.
    """

    from astropy.constants import L_sun

    # Observed wavelength (redshifted)
    wave_obs = wave_rest * (1 + z)

    # Convert luminosity to erg/s/Angstrom
    L_lambda_erg = L_lambda_rest * L_sun.to('erg/s').value  # Now in erg/s/Angstrom

    # Get luminosity distance in cm
    D_L_cm = cosmo.luminosity_distance(z).to('cm').value

    # Compute observed flux density using:
    # F_lambda = (1 / (4 * pi * D_L^2)) * (1 / (1 + z)) * L_lambda_rest
    F_lambda_obs = (L_lambda_erg / (4 * np.pi * D_L_cm**2)) / (1 + z)  # in erg/s/cm^2/Angstrom

    return wave_obs, F_lambda_obs
    

def filtering(wave_spec, flux_spec, wave_filter, trans_filter):
    """
    Compute photometric flux through a filter using spectrum in F_lambda units.

    Parameters
    ----------
    wave_spec : ndarray
        Wavelength array of the spectrum [Å].
    flux_spec : ndarray
        Flux density array of the spectrum [erg/s/cm²/Å].
    wave_filter : ndarray
        Wavelength array of the filter transmission curve [Å].
    trans_filter : ndarray
        Transmission values of the filter (unitless).

    Returns
    -------
    flux_phot : float
        Photometric flux in units of erg/s/cm²/Å.
    """

    from scipy.interpolate import interp1d

    # Interpolate spectrum and filter to common wavelength grid
    wave_min = max(np.min(wave_spec), np.min(wave_filter))
    wave_max = min(np.max(wave_spec), np.max(wave_filter))

    wave_common = np.linspace(wave_min, wave_max, 10000)

    flux_interp = interp1d(wave_spec, flux_spec, kind='linear', bounds_error=False, fill_value=0.0)
    trans_interp = interp1d(wave_filter, trans_filter, kind='linear', bounds_error=False, fill_value=0.0)

    F_lambda = flux_interp(wave_common)
    T_lambda = trans_interp(wave_common)

    numerator = np.trapz(F_lambda * T_lambda * wave_common, wave_common)
    denominator = np.trapz(T_lambda * wave_common, wave_common)

    flux_phot = numerator / denominator

    return flux_phot

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
    from .config import NORM_DUST_Z, NORM_DUST_TAU

    f = interp1d(NORM_DUST_Z, NORM_DUST_TAU, fill_value="extrapolate")
    return f(z)


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


# wave: wavelength in Angstroms
def igm_att_madau(wave,z):
    """
    Compute IGM attenuation (transmission) due to Lyman series and Lyman continuum
    following Madau et al. 1995.

    Parameters:
    -----------
    wave : array_like
        Observed-frame wavelength array in Angstroms.
    z : float
        Source redshift.

    Returns:
    --------
    transmission : ndarray
        Transmission factor at each wavelength (same shape as input).
    """

    wave = np.asarray(wave)
    nwaves = len(wave)

    lylambda = np.zeros(32)
    madauA = np.zeros(32)

    nseries = 4
    for ii in range(2,int(nseries)+2):
        lylambda[ii-2] = 912.0*np.power(ii,2.0)/(np.power(ii,2.0)-1.0)

    madauA[0] = 0.0036
    madauA[1] = 0.0017
    madauA[2] = 0.0012
    madauA[3] = 0.00093

    teffline = np.zeros(nwaves)
    for ii in range(0,nseries):
        lmin = lylambda[ii]
        lmax = lylambda[ii]*(1.0+z)

        idx0 = np.where((wave>=lmin) & (wave<=lmax))
        teffline[idx0[0]] += madauA[ii]*np.exp(3.46*np.log(wave[idx0[0]]/lylambda[ii]))

    xc = wave/912.0
    xem = 1.0+z

    idx = np.where(xc<1.0)
    xc[idx[0]] = 1.0

    idx = np.where(xc>xem)
    xc[idx[0]] = xem

    teffcont = 0.25*xc*xc*xc*(np.exp(0.46*np.log(xem)) - np.exp(0.46*np.log(xc)))
    teffcont = teffcont + 9.4*np.exp(1.5*np.log(xc))*(np.exp(0.18*np.log(xem)) - np.exp(0.18*np.log(xc)))
    teffcont = teffcont - 0.7*xc*xc*xc*(np.exp(-1.32*np.log(xc)) - np.exp(-1.32*np.log(xem)))
    teffcont = teffcont - 0.023*(np.exp(1.68*np.log(xem)) - np.exp(1.68*np.log(xc)))
    tefftot = teffline + teffcont

    return np.exp(-1.0*tefftot)

#==> IGM absorption based on Inoue+2014:

def tau_LAF_LS(wave,z):
	# based on table 2 of inoue et al. 2014
	nj = 39
	lj  = np.asarray([1215.67, 1025.72, 972.537, 949.743, 937.803, 930.748, 926.226, 923.150, 920.963, 919.352, 918.129, 917.181, 916.429, 915.824, 915.329, 914.919, 914.576, 914.286, 914.039, 913.826, 913.641, 913.480, 913.339, 913.215, 913.104, 913.006, 912.918, 912.839, 912.768, 912.703, 912.645, 912.592, 912.543, 912.499, 912.458, 912.420, 912.385, 912.353, 912.324])
	Aj1 = np.asarray([1.690e-2,4.692e-3,2.239e-3,1.319e-3,8.707e-4,6.178e-4,4.609e-4,3.569e-4,2.843e-4,2.318e-4,1.923e-4,1.622e-4,1.385e-4,1.196e-4,1.043e-4,9.174e-5,8.128e-5,7.251e-5,6.505e-5,5.868e-5,5.319e-5,4.843e-5,4.427e-5,4.063e-5,3.738e-5,3.454e-5,3.199e-5,2.971e-5,2.766e-5,2.582e-5,2.415e-5,2.263e-5,2.126e-5,2.000e-5,1.885e-5,1.779e-5,1.682e-5,1.593e-5,1.510e-5])
	Aj2 = np.asarray([2.354e-3,6.536e-4,3.119e-4,1.837e-4,1.213e-4,8.606e-5,6.421e-5,4.971e-5,3.960e-5,3.229e-5,2.679e-5,2.259e-5,1.929e-5,1.666e-5,1.453e-5,1.278e-5,1.132e-5,1.010e-5,9.062e-6,8.174e-6,7.409e-6,6.746e-6,6.167e-6,5.660e-6,5.207e-6,4.811e-6,4.456e-6,4.139e-6,3.853e-6,3.596e-6,3.364e-6,3.153e-6,2.961e-6,2.785e-6,2.625e-6,2.479e-6,2.343e-6,2.219e-6,2.103e-6])
	Aj3 = np.asarray([1.026e-4,2.849e-5,1.360e-5,8.010e-6,5.287e-6,3.752e-6,2.799e-6,2.167e-6,1.726e-6,1.407e-6,1.168e-6,9.847e-7,8.410e-7,7.263e-7,6.334e-7,5.571e-7,4.936e-7,4.403e-7,3.950e-7,3.563e-7,3.230e-7,2.941e-7,2.689e-7,2.467e-7,2.270e-7,2.097e-7,1.943e-7,1.804e-7,1.680e-7,1.568e-7,1.466e-7,1.375e-7,1.291e-7,1.214e-7,1.145e-7,1.080e-7,1.022e-7,9.673e-8,9.169e-8])

	tau = np.zeros(len(wave))
	for jj in range(0,int(nj)):
		# eqn 21 of inoue et al. 2014
		idx0 = np.where((wave>lj[jj]) & (wave<lj[jj]*(1.0+z)) & (wave<2.2*lj[jj]))
		tau[idx0[0]] += Aj1[jj]*np.power(wave[idx0[0]]/lj[jj],1.2)

		idx1 = np.where((wave>lj[jj]) & (wave<lj[jj]*(1.0+z)) & (wave>=2.2*lj[jj]) & (wave<5.7*lj[jj]))
		tau[idx1[0]] += Aj2[jj]*np.power(wave[idx1[0]]/lj[jj],3.7)

		idx2 = np.where((wave>lj[jj]) & (wave<lj[jj]*(1.0+z)) & (wave>=5.7*lj[jj]))
		tau[idx2[0]] += Aj3[jj]*np.power(wave[idx2[0]]/lj[jj],5.5)

	return tau


def tau_DLA_LS(wave,z):
	# table 2 of inoue et al. 2014
	nj = 39
	lj = np.asarray([1215.67, 1025.72, 972.537, 949.743, 937.803, 930.748, 926.226, 923.150, 920.963, 919.352, 918.129, 917.181, 916.429, 915.824, 915.329, 914.919, 914.576, 914.286, 914.039, 913.826, 913.641, 913.480, 913.339, 913.215, 913.104, 913.006, 912.918, 912.839, 912.768, 912.703, 912.645, 912.592, 912.543, 912.499, 912.458, 912.420, 912.385, 912.353, 912.324])
	Aj1 = np.asarray([1.617e-4,1.545e-4,1.498e-4,1.460e-4,1.429e-4,1.402e-4,1.377e-4,1.355e-4,1.335e-4,1.316e-4,1.298e-4,1.281e-4,1.265e-4,1.250e-4,1.236e-4,1.222e-4,1.209e-4,1.197e-4,1.185e-4,1.173e-4,1.162e-4,1.151e-4,1.140e-4,1.130e-4,1.120e-4,1.110e-4,1.101e-4,1.091e-4,1.082e-4,1.073e-4,1.065e-4,1.056e-4,1.048e-4,1.040e-4,1.032e-4,1.024e-4,1.017e-4,1.009e-4,1.002e-4])
	Aj2 = np.asarray([5.390e-5,5.151e-5,4.992e-5,4.868e-5,4.763e-5,4.672e-5,4.590e-5,4.516e-5,4.448e-5,4.385e-5,4.326e-5,4.271e-5,4.218e-5,4.168e-5,4.120e-5,4.075e-5,4.031e-5,3.989e-5,3.949e-5,3.910e-5,3.872e-5,3.836e-5,3.800e-5,3.766e-5,3.732e-5,3.700e-5,3.668e-5,3.637e-5,3.607e-5,3.578e-5,3.549e-5,3.521e-5,3.493e-5,3.466e-5,3.440e-5,3.414e-5,3.389e-5,3.364e-5,3.339e-5])

	tau = np.zeros(len(wave))
	for jj in range(0,int(nj)):
		# eqn 22 of inoue et al. 2014
		idx0 = np.where((wave>lj[jj]) & (wave<lj[jj]*(1.0+z)) & (wave<3.0*lj[jj]))
		tau[idx0[0]] += Aj1[jj]*np.power(wave[idx0[0]]/lj[jj],2.0)

		idx0 = np.where((wave>lj[jj]) & (wave<lj[jj]*(1.0+z)) & (wave>=3.0*lj[jj]))
		tau[idx0[0]] += Aj2[jj]*np.power(wave[idx0[0]]/lj[jj],3.0)

	return tau


def tau_DLA_LC(wave,z):
	lL = 911.8   # Lyman limit

	# eqn 28 and 29 of Inoue et al. 2014
	tau = np.zeros(len(wave))
	if z<2.0:
		idx0 = np.where(wave<lL*(1+z))
		tau[idx0[0]] = 0.211*np.power(1.0+z,2) - 7.66e-2*np.power(1.0+z,2.3)*np.power(wave[idx0[0]]/lL,-0.3) - 0.135*np.power(wave[idx0[0]]/lL,2.0)

		idx1 = np.where(wave>=lL*(1+z))
		tau[idx1[0]] = 0.0

	else:
		idx0 = np.where(wave<3.0*lL)
		tau[idx0[0]] = 0.634 + 4.70e-2*np.power(1.0+z,3.0) - 1.78e-2*np.power(1.0+z,3.3)*np.power(wave[idx0[0]]/lL,-0.3) - 0.135*np.power(wave[idx0[0]]/lL,2.0) - 0.291*np.power(wave[idx0[0]]/lL,-0.3)

		idx1 = np.where((wave>=3.0*lL) & (wave<lL*(1.0+z)))
		tau[idx1[0]] = 4.70e-2*np.power(1.0+z,3.0) - 1.78e-2*np.power(1.0+z,3.3)*np.power(wave[idx1[0]]/lL,-0.3) - 2.92e-2*np.power(wave[idx1[0]]/lL,3)

		idx2 = np.where(wave>=lL*(1.0+z))
		tau[idx2[0]] = 0.0

	return tau


def tau_LAF_LC(wave,z):
	# eqn 25, 26 and 29 of Inoue et al. 2014:
	lL = 911.8   # Lyman limit

	tau = np.zeros(len(wave))

	if z<1.2:
		idx0 = np.where(wave<lL*(1+z))
		tau[idx0[0]] = 0.325*(np.power(wave[idx0[0]]/lL,1.2)-np.power(1.0+z,-0.9)*np.power(wave[idx0[0]]/lL,2.1))

		idx1 = np.where(wave>=lL*(1+z))
		tau[idx1[0]] = 0.0

	elif z<4.7:
		idx0 = np.where(wave<2.2*lL)
		tau[idx0[0]] = 2.55e-2*np.power(1.0+z,1.6)*np.power(wave[idx0[0]]/lL,2.1) + 0.325*np.power(wave[idx0[0]]/lL,1.2) - 0.250*np.power(wave[idx0[0]]/lL,2.1)

		idx1 = np.where((wave>=2.2*lL) & (wave<lL*(1+z)))
		tau[idx1[0]] = 2.55e-2*(np.power(1.0+z,1.6)*np.power(wave[idx1[0]]/lL,2.1) - np.power(wave[idx1[0]]/lL,3.7))

		idx2 = np.where(wave>=lL*(1+z))
		tau[idx2[0]] = 0.0
			
	else:
		idx0 = np.where(wave<2.2*lL)
		tau[idx0[0]] = 5.22e-4*np.power(1.0+z,3.4)*np.power(wave[idx0[0]]/lL,2.1) + 0.325*np.power(wave[idx0[0]]/lL,1.2) - 3.14e-2*np.power(wave[idx0[0]]/lL,2.1)

		idx1 = np.where((wave>=2.2*lL) & (wave<5.7*lL))
		tau[idx1[0]] = 5.22e-4*np.power(1.0+z,3.4)*np.power(wave[idx1[0]]/lL,2.1) + 0.218*np.power(wave[idx1[0]]/lL,2.1) - 2.55e-2*np.power(wave[idx1[0]]/lL,3.7)

		idx2 = np.where((wave>=5.7*lL) & (wave<lL*(1+z)))
		tau[idx2[0]] = 5.22e-4*(np.power(1.0+z,3.4)*np.power(wave[idx2[0]]/lL,2.1) - np.power(wave[idx2[0]]/lL,5.5))

		idx3 = np.where(wave>=lL*(1+z))
		tau[idx3[0]] = 0.0

	return tau


def igm_att_inoue(wave,z):
	# eq. 15 of Inoue + 2014:
	tau = tau_LAF_LS(wave,z) + tau_DLA_LS(wave,z) + tau_LAF_LC(wave,z) + tau_DLA_LC(wave,z)

	return np.exp(-1.0*tau)


def get_2d_density_projection_no_los_binning(star_coords, particle_masses, pixel_size, output_dimension, 
                                             proj_angle_deg=0, projection_mode=None, gas_coords=None, gas_masses=None):
    """
    Calculates the 2D mass density projection of star particles given 3D coordinates,
    their masses, a fixed pixel size, and a desired output dimension (cutout size).
    The output grid is geometrically centered around the most massive pixel of the
    full projected star data. It also calculates the projected distance along the line of sight
    for each star particle within its 2D pixel, and outputs the geometric central pixel coordinate of the cutout.

    If `gas_coords` and `gas_masses` are provided, it also calculates the membership of gas particles
    and their mass density within the *same* projected pixel gridding defined by the star particles.

    The line-of-sight distance for each particle (star or gas) is normalized such that the closest
    particle (considering both stars and gas, if gas is provided) has a line-of-sight
    distance of 0, and distances increase farther away from this closest point.

    If `projection_mode` is specified, it overrides `proj_angle_deg`.
    The `proj_angle_deg` is measured counter-clockwise from the positive Y-axis.
    The line-of-sight distance corresponds to the new Z-coordinate after this transformation.

    Args:
        star_coords (np.ndarray): A NumPy array of shape (N, 3) representing the
                                  (x, y, z) coordinates of N star particles.
        particle_masses (np.ndarray): A NumPy array of shape (N,) representing the
                                      masses of N star particles.
        pixel_size (float): The size of each pixel in the 2D grid. Must be positive.
        output_dimension (tuple): A tuple (width, height) specifying the desired total
                                  physical width and height of the 2D output map.
                                  Each dimension must be positive.
        proj_angle_deg (float, optional): The projection angle in degrees, measured from the
                                          positive Z-axis (rotation around Y-axis).
                                          Defaults to 0. Ignored if `projection_mode` is provided.
        projection_mode (str, optional): A string specifying a predefined orthogonal projection.
                                         Can be 'XYZ', 'YXZ', 'ZYX', or 'YZX'.
                                         If provided, it overrides `proj_angle_deg`.
                                         Defaults to None.
        gas_coords (np.ndarray, optional): A NumPy array of shape (M, 3) representing the
                                           (x, y, z) coordinates of M gas particles.
                                           Defaults to None.
        gas_masses (np.ndarray, optional): A NumPy array of shape (M,) representing the
                                           masses of M gas particles. Must be provided if
                                           `gas_coords` is provided. Defaults to None.

    Returns:
        tuple: A tuple containing:
            - star_particle_membership (list of lists of lists): A 2D list (grid) where
              each element `star_particle_membership[y_idx][x_idx]` is a list of
              tuples `(original_particle_index, line_of_sight_distance)`
              for star particles that fall into that pixel. The `line_of_sight_distance`
              is now relative to the closest particle (0 being the closest).
            - star_mass_density_map (np.ndarray): A 2D NumPy array representing the mass
              density map (sum of masses per pixel) for star particles.
              Its shape is (num_pixels_y, num_pixels_x).
            - central_pixel_coords (tuple): A tuple (x_pixel_coord, y_pixel_coord)
              representing the geometric central pixel coordinate *within the cutout*.
            - grid_info (dict): A dictionary containing information about the grid:
              'min_x_proj': The minimum x-coordinate of the projected cutout.
              'min_y_proj': The minimum y-coordinate of the projected cutout.
              'num_pixels_x': The number of pixels along the x-dimension.
              'num_pixels_y': The number of pixels along the y-dimension.
              'effective_pixel_size_x': The actual pixel size used for x-dimension (same as pixel_size).
              'effective_pixel_size_y': The actual pixel size used for y-dimension (same as pixel_size).
            - gas_particle_membership (list of lists of lists or None): Same structure as
              `star_particle_membership` but for gas particles. None if `gas_coords` not provided.
            - gas_mass_density_map (np.ndarray or None): A 2D NumPy array representing the mass
              density map for gas particles. Its shape is (num_pixels_y, num_pixels_x).
              None if `gas_coords` not provided.

    Raises:
        ValueError: If star_coords/particle_masses are invalid, pixel_size is not positive,
                    output_dimension is invalid, an invalid `projection_mode` is provided,
                    or `gas_coords` is provided without `gas_masses` (or vice-versa).
    """
    # --- Input Validation ---
    if not isinstance(star_coords, np.ndarray) or star_coords.ndim != 2 or star_coords.shape[1] != 3:
        raise ValueError("star_coords must be a NumPy array of shape (N, 3).")
    if not isinstance(particle_masses, np.ndarray) or particle_masses.ndim != 1 or particle_masses.shape[0] != star_coords.shape[0]:
        raise ValueError("particle_masses must be a 1D NumPy array of shape (N,) matching star_coords.")
    if pixel_size <= 0:
        raise ValueError("pixel_size must be positive.")
    if not (isinstance(output_dimension, tuple) and len(output_dimension) == 2 and
            output_dimension[0] > 0 and output_dimension[1] > 0):
        raise ValueError("'output_dimension' must be a tuple (width, height) of two positive numbers.")

    if (gas_coords is not None and gas_masses is None) or \
       (gas_coords is None and gas_masses is not None):
        raise ValueError("Both 'gas_coords' and 'gas_masses' must be provided together, or neither.")
    if gas_coords is not None:
        if not (isinstance(gas_coords, np.ndarray) or gas_coords.ndim != 2 or gas_coords.shape[1] != 3):
            raise ValueError("gas_coords must be a NumPy array of shape (M, 3).")
        if not (isinstance(gas_masses, np.ndarray) or gas_masses.ndim != 1 or gas_masses.shape[0] != gas_coords.shape[0]):
            raise ValueError("gas_masses must be a 1D NumPy array of shape (M,) matching gas_coords.")


    # --- Handle Empty Particle Set (Stars) ---
    if star_coords.shape[0] == 0:
        print("Warning: No star particles provided. Returning empty results for no LOS binning.")
        return [], np.array([[]]), (0, 0), {'min_x_proj': 0, 'min_y_proj': 0, 'num_pixels_x': 0, 'num_pixels_y': 0,
                                            'effective_pixel_size_x': pixel_size, 'effective_pixel_size_y': pixel_size}, [], np.array([[]])

    # --- Determine transformation based on projection_mode or proj_angle_deg ---
    if projection_mode:
        mode_map = {
            'XYZ': (0, 1, 2), # New X=Orig X, New Y=Orig Y, LOS=Orig Z
            'YXZ': (1, 0, 2), # New X=Orig Y, New Y=Orig X, LOS=Orig Z
            'ZYX': (2, 1, 0), # New X=Orig Z, New Y=Orig Y, LOS=Orig X
            'YZX': (1, 2, 0)  # New X=Orig Y, New Y=Orig Z, LOS=Orig X
        }
        if projection_mode not in mode_map:
            raise ValueError(f"Invalid projection_mode: '{projection_mode}'. "
                             "Accepted modes are 'XYZ', 'YXZ', 'ZYX', 'YZX'.")
        x_idx, y_idx, z_los_idx = mode_map[projection_mode]
        rotated_star_coords = star_coords[:, [x_idx, y_idx, z_los_idx]]
        if gas_coords is not None:
            rotated_gas_coords = gas_coords[:, [x_idx, y_idx, z_los_idx]]
    else:
        proj_angle_rad = np.deg2rad(proj_angle_deg)
        cos_theta = np.cos(proj_angle_rad)
        sin_theta = np.sin(proj_angle_rad)
        rotation_matrix = np.array([
            [cos_theta, 0, sin_theta],
            [0, 1, 0],
            [-sin_theta, 0, cos_theta]
        ])
        rotated_star_coords = np.dot(star_coords, rotation_matrix.T)
        if gas_coords is not None:
            rotated_gas_coords = np.dot(gas_coords, rotation_matrix.T)

    # --- Extract projected 2D coordinates and raw line-of-sight distances for STARS ---
    projected_star_2d_coords = rotated_star_coords[:, :2]
    star_line_of_sight_distances_raw = rotated_star_coords[:, 2]

    # --- Determine the global minimum LOS distance for normalization (from all particles) ---
    all_los_distances = star_line_of_sight_distances_raw
    if gas_coords is not None:
        gas_line_of_sight_distances_raw = rotated_gas_coords[:, 2]
        all_los_distances = np.concatenate((all_los_distances, gas_line_of_sight_distances_raw))

    # Handle case where all_los_distances might be empty (e.g., if both star_coords and gas_coords are empty after initial check)
    min_global_los = np.min(all_los_distances) if all_los_distances.size > 0 else 0.0

    # --- Normalize line-of-sight distances for STARS ---
    star_line_of_sight_distances_normalized = star_line_of_sight_distances_raw - min_global_los

    # --- Calculate the extent of the *entire* projected STAR dataset for global map ---
    min_x_full, min_y_full = np.min(projected_star_2d_coords, axis=0)
    max_x_full, max_y_full = np.max(projected_star_2d_coords, axis=0)

    # Use the input pixel_size for consistency, but ensure it covers the full range
    effective_pixel_size_global = pixel_size
    epsilon_global = 1e-9 * effective_pixel_size_global

    num_pixels_x_global = int(np.ceil((max_x_full - min_x_full + epsilon_global) / effective_pixel_size_global))
    num_pixels_y_global = int(np.ceil((max_y_full - min_y_full + epsilon_global) / effective_pixel_size_global))

    if num_pixels_x_global == 0: num_pixels_x_global = 1
    if num_pixels_y_global == 0: num_pixels_y_global = 1

    # --- Create a global mass density map (STARS) to find the most massive pixel ---
    global_star_mass_density_map = np.zeros((num_pixels_y_global, num_pixels_x_global), dtype=float)

    for i in range(star_coords.shape[0]):
        x_coord_proj = projected_star_2d_coords[i, 0]
        y_coord_proj = projected_star_2d_coords[i, 1]

        x_idx_global = int(np.floor((x_coord_proj - min_x_full) / effective_pixel_size_global))
        y_idx_global = int(np.floor((y_coord_proj - min_y_full) / effective_pixel_size_global))

        # Clip indices to ensure they are within the valid range for the global map
        x_idx_global = np.clip(x_idx_global, 0, num_pixels_x_global - 1)
        y_idx_global = np.clip(y_idx_global, 0, num_pixels_y_global - 1)

        global_star_mass_density_map[y_idx_global][x_idx_global] += particle_masses[i]

    # --- Find the most massive pixel in the global STAR map ---
    most_massive_pixel_x_idx_global = 0
    most_massive_pixel_y_idx_global = 0

    if np.sum(global_star_mass_density_map) > 0: # Only search if there's mass
        # Find the index of the maximum value in the flattened array
        flat_idx = np.argmax(global_star_mass_density_map)
        most_massive_pixel_y_idx_global, most_massive_pixel_x_idx_global = np.unravel_index(flat_idx, global_star_mass_density_map.shape)

    # Convert most massive pixel index to its physical center coordinate in the global space
    most_massive_pixel_x_coord_global = min_x_full + (most_massive_pixel_x_idx_global + 0.5) * effective_pixel_size_global
    most_massive_pixel_y_coord_global = min_y_full + (most_massive_pixel_y_idx_global + 0.5) * effective_pixel_size_global


    # --- Define the cutout grid's extent and dimensions based on most massive STAR pixel ---
    num_pixels_x_cutout = int(np.ceil(output_dimension[0] / pixel_size))
    num_pixels_y_cutout = int(np.ceil(output_dimension[1] / pixel_size))

    if num_pixels_x_cutout == 0: num_pixels_x_cutout = 1
    if num_pixels_y_cutout == 0: num_pixels_y_cutout = 1

    # Calculate the minimum projected coordinates for the cutout,
    # ensuring its geometric center aligns with the most massive pixel's center.
    min_x_cutout = most_massive_pixel_x_coord_global - (num_pixels_x_cutout * pixel_size / 2.0)
    min_y_cutout = most_massive_pixel_y_coord_global - (num_pixels_y_cutout * pixel_size / 2.0)

    # --- Initialize output arrays for the cutout region (STARS) ---
    star_particle_membership = [[[] for _ in range(num_pixels_x_cutout)] for _ in range(num_pixels_y_cutout)]
    star_mass_density_map = np.zeros((num_pixels_y_cutout, num_pixels_x_cutout), dtype=float)

    # --- Assign STAR particles to the cutout grid ---
    for i in range(star_coords.shape[0]):
        x_coord_proj = projected_star_2d_coords[i, 0]
        y_coord_proj = projected_star_2d_coords[i, 1]

        x_idx_cutout = int(np.floor((x_coord_proj - min_x_cutout) / pixel_size))
        y_idx_cutout = int(np.floor((y_coord_proj - min_y_cutout) / pixel_size))

        # Only include particles that fall within the cutout dimensions
        if 0 <= x_idx_cutout < num_pixels_x_cutout and 0 <= y_idx_cutout < num_pixels_y_cutout:
            star_particle_membership[y_idx_cutout][x_idx_cutout].append((i, star_line_of_sight_distances_normalized[i])) # Use normalized distance
            star_mass_density_map[y_idx_cutout][x_idx_cutout] += particle_masses[i]

    # --- Set central pixel coordinate to the geometric center of the cutout ---
    central_pixel_x = num_pixels_x_cutout // 2
    central_pixel_y = num_pixels_y_cutout // 2
    central_pixel_coords = (central_pixel_x, central_pixel_y)

    # --- Process GAS particles if provided ---
    gas_particle_membership = None
    gas_mass_density_map = None
    if gas_coords is not None:
        gas_particle_membership = [[[] for _ in range(num_pixels_x_cutout)] for _ in range(num_pixels_y_cutout)]
        gas_mass_density_map = np.zeros((num_pixels_y_cutout, num_pixels_x_cutout), dtype=float)
        projected_gas_2d_coords = rotated_gas_coords[:, :2]
        # gas_line_of_sight_distances_raw was already extracted and used for min_global_los
        gas_line_of_sight_distances_normalized = gas_line_of_sight_distances_raw - min_global_los # Normalize gas LOS

        for i in range(gas_coords.shape[0]):
            x_coord_proj = projected_gas_2d_coords[i, 0]
            y_coord_proj = projected_gas_2d_coords[i, 1]

            x_idx_cutout = int(np.floor((x_coord_proj - min_x_cutout) / pixel_size))
            y_idx_cutout = int(np.floor((y_coord_proj - min_y_cutout) / pixel_size))

            if 0 <= x_idx_cutout < num_pixels_x_cutout and 0 <= y_idx_cutout < num_pixels_y_cutout:
                gas_particle_membership[y_idx_cutout][x_idx_cutout].append((i, gas_line_of_sight_distances_normalized[i])) # Use normalized distance
                gas_mass_density_map[y_idx_cutout][x_idx_cutout] += gas_masses[i]

    # --- Prepare grid information for output ---
    grid_info = {
        'min_x_proj': min_x_cutout,
        'min_y_proj': min_y_cutout,
        'num_pixels_x': num_pixels_x_cutout,
        'num_pixels_y': num_pixels_y_cutout,
        'effective_pixel_size_x': pixel_size,
        'effective_pixel_size_y': pixel_size
    }

    return star_particle_membership, star_mass_density_map, central_pixel_coords, grid_info, gas_particle_membership, gas_mass_density_map



def get_2d_density_projection_with_los_binning(star_coords, particle_masses, pixel_size, output_dimension, los_bin_size,
                                               los_dimension, proj_angle_deg=0, projection_mode=None, gas_coords=None, gas_masses=None):
    """
    Calculates the 3D mass density projection of star particles given 3D coordinates,
    their masses, fixed pixel/bin sizes, and desired output dimensions for all axes.
    The output grid is geometrically centered around the most massive pixel of the
    full projected 2D star map (after LOS summation). It also calculates the projected distance
    along the line of sight for each star particle within its 3D bin, and outputs the
    geometric central pixel coordinate of the *summed* 2D star mass map.

    If `gas_coords` and `gas_masses` are provided, it also calculates the membership of gas particles
    and their mass density within the *same* projected 3D pixel gridding defined by the star particles.

    If `projection_mode` is specified, it overrides `proj_angle_deg`.
    The `proj_angle_deg` is measured counter-clockwise from the positive Y-axis.
    The line-of-sight distance corresponds to the new Z-coordinate after this transformation.

    The line-of-sight distance for each particle is normalized such that the closest
    particle (considering both stars and gas, if gas is provided) has a line-of-sight
    distance of 0, and distances increase farther away from this closest point.

    Args:
        star_coords (np.ndarray): A NumPy array of shape (N, 3) representing the
                                  (x, y, z) coordinates of N star particles.
        particle_masses (np.ndarray): A NumPy array of shape (N,) representing the
                                      masses of N star particles.
        pixel_size (float): The size of each pixel in the 2D grid (x and y dimensions).
                            Must be positive.
        output_dimension (tuple): A tuple (width, height) specifying the desired total
                                  physical width and height of the 2D output map.
                                  Each dimension must be positive.
        los_bin_size (float): The size of each bin along the line-of-sight (z) dimension.
                              Must be positive.
        los_dimension (float): The desired total physical depth along the line-of-sight.
                               Must be positive.
        proj_angle_deg (float, optional): The projection angle in degrees, measured from the
                                          positive Z-axis (rotation around Y-axis).
                                          Defaults to 0. Ignored if `projection_mode` is provided.
        projection_mode (str, optional): A string specifying a predefined orthogonal projection.
                                         Can be 'XYZ', 'YXZ', 'ZYX', or 'YZX'.
                                         If provided, it overrides `proj_angle_deg`.
                                         Defaults to None.
        gas_coords (np.ndarray, optional): A NumPy array of shape (M, 3) representing the
                                           (x, y, z) coordinates of M gas particles.
                                           Defaults to None.
        gas_masses (np.ndarray, optional): A NumPy array of shape (M,) representing the
                                           masses of M gas particles. Must be provided if
                                           `gas_coords` is provided. Defaults to None.

    Returns:
        tuple: A tuple containing:
            - star_particle_membership (list of lists of lists of lists): A 3D list (grid) where
              each element `star_particle_membership[y_idx][x_idx][z_idx]` is a list of
              `original_particle_index` for star particles that fall into that specific 3D bin.
            - star_mass_density_map (np.ndarray): A 3D NumPy array representing the mass
              density map (sum of masses per 3D bin) for star particles.
              Its shape is (num_pixels_y, num_pixels_x, num_pixels_z).
            - central_pixel_coords (tuple): A tuple (x_pixel_coord, y_pixel_coord)
              representing the geometric central pixel coordinate of the *summed* 2D map.
            - grid_info (dict): A dictionary containing information about the grid:
              'min_x_proj': The minimum x-coordinate of the projected cutout.
              'min_y_proj': The minimum y-coordinate of the projected cutout.
              'min_z_proj': The minimum z-coordinate (line-of-sight) of the projected cutout.
              'num_pixels_x': The number of pixels along the x-dimension.
              'num_pixels_y': The number of pixels along the y-dimension.
              'num_pixels_z': The number of pixels along the z-dimension (line-of-sight).
              'effective_pixel_size_x': The actual pixel size used for x-dimension (same as pixel_size).
              'effective_pixel_size_y': The actual pixel size used for y-dimension (same as pixel_size).
              'effective_los_bin_size': The actual bin size used for z-dimension (same as los_bin_size).
            - gas_particle_membership (list of lists of lists of lists or None): Same structure as
              `star_particle_membership` but for gas particles, containing `original_particle_index`.
              None if `gas_coords` not provided.
            - gas_mass_density_map (np.ndarray or None): A 3D NumPy array representing the mass
              density map for gas particles. Its shape is (num_pixels_y, num_pixels_x, num_pixels_z).
              None if `gas_coords` not provided.
            - gas_particles_in_front_of_grid (list of lists of lists of lists or None): A 3D list (grid) where
              each element `gas_particles_in_front_of_grid[y_idx][x_idx][z_idx]` is a list of
              `original_particle_index` for all gas particles in bins *in front* of the current
              `z_idx` along the line of sight for that (y_idx, x_idx) column. None if `gas_coords` not provided.
            - star_particles_in_front_of_grid (list of lists of lists of lists or None): A 3D list (grid) where
              each element `star_particles_in_front_of_grid[y_idx][x_idx][z_idx]` is a list of
              `original_particle_index` for all star particles in bins *in front* of the current
              `z_idx` along the line of sight for that (y_idx, x_idx) column. None if no star particles.

    Raises:
        ValueError: If star_coords/particle_masses are invalid, pixel_size/los_bin_size are not positive,
                    output_dimension/los_dimension are invalid, an invalid `projection_mode` is provided,
                    or `gas_coords` is provided without `gas_masses` (or vice-versa).
    """
    # --- Input Validation ---
    if not isinstance(star_coords, np.ndarray) or star_coords.ndim != 2 or star_coords.shape[1] != 3:
        raise ValueError("star_coords must be a NumPy array of shape (N, 3).")
    if not isinstance(particle_masses, np.ndarray) or particle_masses.ndim != 1 or particle_masses.shape[0] != star_coords.shape[0]:
        raise ValueError("particle_masses must be a 1D NumPy array of shape (N,) matching star_coords.")
    if pixel_size <= 0:
        raise ValueError("pixel_size must be positive.")
    if los_bin_size <= 0:
        raise ValueError("los_bin_size must be positive.")
    if not (isinstance(output_dimension, tuple) and len(output_dimension) == 2 and
            output_dimension[0] > 0 and output_dimension[1] > 0):
        raise ValueError("'output_dimension' must be a tuple (width, height) of two positive numbers.")
    if not (isinstance(los_dimension, (int, float)) and los_dimension > 0):
        raise ValueError("'los_dimension' must be a positive number.")

    if (gas_coords is not None and gas_masses is None) or \
       (gas_coords is None and gas_masses is not None):
        raise ValueError("Both 'gas_coords' and 'gas_masses' must be provided together, or neither.")
    if gas_coords is not None:
        if not (isinstance(gas_coords, np.ndarray) or gas_coords.ndim != 2 or gas_coords.shape[1] != 3):
            raise ValueError("gas_coords must be a NumPy array of shape (M, 3).")
        if not (isinstance(gas_masses, np.ndarray) or gas_masses.ndim != 1 or gas_masses.shape[0] != gas_coords.shape[0]):
            raise ValueError("gas_masses must be a 1D NumPy array of shape (M,) matching gas_coords.")

    # --- Handle Empty Particle Set (Stars) ---
    if star_coords.shape[0] == 0:
        print("Warning: No star particles provided. Returning empty results for LOS binning.")
        return [], np.array([[]]), (0, 0), {'min_x_proj': 0, 'min_y_proj': 0, 'min_z_proj': 0,
                                            'num_pixels_x': 0, 'num_pixels_y': 0, 'num_pixels_z': 0,
                                            'effective_pixel_size_x': pixel_size, 'effective_pixel_size_y': pixel_size, 'effective_los_bin_size': los_bin_size}, [], np.array([[]]), None, None

    # --- Determine transformation based on projection_mode or proj_angle_deg ---
    if projection_mode:
        mode_map = {
            'XYZ': (0, 1, 2), # New X=Orig X, New Y=Orig Y, LOS=Orig Z
            'YXZ': (1, 0, 2), # New X=Orig Y, New Y=Orig X, LOS=Orig Z
            'ZYX': (2, 1, 0), # New X=Orig Z, New Y=Orig Y, LOS=Orig X
            'YZX': (1, 2, 0)  # New X=Orig Y, New Y=Orig Z, LOS=Orig X
        }
        if projection_mode not in mode_map:
            raise ValueError(f"Invalid projection_mode: '{projection_mode}'. "
                             "Accepted modes are 'XYZ', 'YXZ', 'ZYX', 'YZX'.")
        x_idx, y_idx, z_los_idx = mode_map[projection_mode]
        rotated_star_coords = star_coords[:, [x_idx, y_idx, z_los_idx]]
        if gas_coords is not None:
            rotated_gas_coords = gas_coords[:, [x_idx, y_idx, z_los_idx]]
    else:
        proj_angle_rad = np.deg2rad(proj_angle_deg)
        cos_theta = np.cos(proj_angle_rad)
        sin_theta = np.sin(proj_angle_rad)
        rotation_matrix = np.array([
            [cos_theta, 0, sin_theta],
            [0, 1, 0],
            [-sin_theta, 0, cos_theta]
        ])
        rotated_star_coords = np.dot(star_coords, rotation_matrix.T)
        if gas_coords is not None:
            rotated_gas_coords = np.dot(gas_coords, rotation_matrix.T)

    # --- Extract projected 2D coordinates and line-of-sight distances for STARS ---
    projected_star_2d_coords = rotated_star_coords[:, :2]
    star_line_of_sight_distances = rotated_star_coords[:, 2]

    # --- Determine the global minimum LOS distance for normalization ---
    all_los_distances = star_line_of_sight_distances
    if gas_coords is not None:
        gas_line_of_sight_distances_raw = rotated_gas_coords[:, 2]
        all_los_distances = np.concatenate((all_los_distances, gas_line_of_sight_distances_raw))

    min_global_los = np.min(all_los_distances) if all_los_distances.size > 0 else 0.0

    # --- Normalize line-of-sight distances ---
    star_line_of_sight_distances_normalized = star_line_of_sight_distances - min_global_los
    if gas_coords is not None:
        gas_line_of_sight_distances_normalized = gas_line_of_sight_distances_raw - min_global_los

    # --- Calculate the extent of the *entire* projected STAR dataset for global map ---
    # Use normalized LOS distances for min_z_full and max_z_full
    min_x_full, min_y_full = np.min(projected_star_2d_coords, axis=0)
    max_x_full, max_y_full = np.max(projected_star_2d_coords, axis=0)
    min_z_full, max_z_full = np.min(star_line_of_sight_distances_normalized), np.max(star_line_of_sight_distances_normalized)


    # Define a temporary pixel size for the global map if it's too fine/coarse
    effective_pixel_size_global = pixel_size
    effective_los_bin_size_global = los_bin_size
    epsilon_global_xy = 1e-9 * effective_pixel_size_global
    epsilon_global_z = 1e-9 * effective_los_bin_size_global

    num_pixels_x_global = int(np.ceil((max_x_full - min_x_full + epsilon_global_xy) / effective_pixel_size_global))
    num_pixels_y_global = int(np.ceil((max_y_full - min_y_full + epsilon_global_xy) / effective_pixel_size_global))
    num_pixels_z_global = int(np.ceil((max_z_full - min_z_full + epsilon_global_z) / effective_los_bin_size_global))

    if num_pixels_x_global == 0: num_pixels_x_global = 1
    if num_pixels_y_global == 0: num_pixels_y_global = 1
    if num_pixels_z_global == 0: num_pixels_z_global = 1

    # --- Create a global 3D mass density map (STARS) ---
    global_star_mass_density_map_3d = np.zeros((num_pixels_y_global, num_pixels_x_global, num_pixels_z_global), dtype=float)

    for i in range(star_coords.shape[0]):
        x_coord_proj = projected_star_2d_coords[i, 0]
        y_coord_proj = projected_star_2d_coords[i, 1]
        z_coord_los = star_line_of_sight_distances_normalized[i] # Use normalized distance

        x_idx_global = int(np.floor((x_coord_proj - min_x_full) / effective_pixel_size_global))
        y_idx_global = int(np.floor((y_coord_proj - min_y_full) / effective_pixel_size_global))
        z_idx_global = int(np.floor((z_coord_los - min_z_full) / effective_los_bin_size_global))

        # Clip indices to ensure they are within the valid range for the global map
        x_idx_global = np.clip(x_idx_global, 0, num_pixels_x_global - 1)
        y_idx_global = np.clip(y_idx_global, 0, num_pixels_y_global - 1)
        z_idx_global = np.clip(z_idx_global, 0, num_pixels_z_global - 1)

        global_star_mass_density_map_3d[y_idx_global][x_idx_global][z_idx_global] += particle_masses[i]

    # --- Find the most massive pixel in the 2D summed global STAR map ---
    global_star_mass_density_map_2d_summed = np.sum(global_star_mass_density_map_3d, axis=2)

    most_massive_pixel_x_idx_global = 0
    most_massive_pixel_y_idx_global = 0

    if np.sum(global_star_mass_density_map_2d_summed) > 0: # Only search if there's mass
        flat_idx = np.argmax(global_star_mass_density_map_2d_summed)
        most_massive_pixel_y_idx_global, most_massive_pixel_x_idx_global = np.unravel_index(flat_idx, global_star_mass_density_map_2d_summed.shape)

    # Convert most massive pixel index to its physical center coordinate in the global 2D space
    most_massive_pixel_x_coord_global = min_x_full + (most_massive_pixel_x_idx_global + 0.5) * effective_pixel_size_global
    most_massive_pixel_y_coord_global = min_y_full + (most_massive_pixel_y_idx_global + 0.5) * effective_pixel_size_global

    # For LOS dimension, we'll center around the median of the full LOS distances of STARS
    # Use normalized median for centering
    median_z_full = np.median(star_line_of_sight_distances_normalized)

    # --- Define the cutout grid's extent and dimensions based on most massive STAR pixel and LOS median ---
    num_pixels_x_cutout = int(np.ceil(output_dimension[0] / pixel_size))
    num_pixels_y_cutout = int(np.ceil(output_dimension[1] / pixel_size))
    num_pixels_z_cutout = int(np.ceil(los_dimension / los_bin_size))

    if num_pixels_x_cutout == 0: num_pixels_x_cutout = 1
    if num_pixels_y_cutout == 0: num_pixels_y_cutout = 1
    if num_pixels_z_cutout == 0: num_pixels_z_cutout = 1

    # Calculate the minimum projected coordinates for the cutout,
    # ensuring its geometric center aligns with the most massive pixel (XY) and LOS median (Z).
    min_x_cutout = most_massive_pixel_x_coord_global - (num_pixels_x_cutout * pixel_size / 2.0)
    min_y_cutout = most_massive_pixel_y_coord_global - (num_pixels_y_cutout * pixel_size / 2.0)
    # min_z_cutout should now be relative to the normalized LOS distances
    min_z_cutout = median_z_full - (num_pixels_z_cutout * los_bin_size / 2.0)


    # --- Initialize output arrays for the cutout region (STARS) ---
    star_particle_membership = [[[[] for _ in range(num_pixels_z_cutout)]
                            for _ in range(num_pixels_x_cutout)]
                           for _ in range(num_pixels_y_cutout)]
    star_mass_density_map = np.zeros((num_pixels_y_cutout, num_pixels_x_cutout, num_pixels_z_cutout), dtype=float)

    # --- Assign STAR particles to the cutout grid ---
    for i in range(star_coords.shape[0]):
        x_coord_proj = projected_star_2d_coords[i, 0]
        y_coord_proj = projected_star_2d_coords[i, 1]
        z_coord_los = star_line_of_sight_distances_normalized[i] # Use normalized distance

        x_idx_cutout = int(np.floor((x_coord_proj - min_x_cutout) / pixel_size))
        y_idx_cutout = int(np.floor((y_coord_proj - min_y_cutout) / pixel_size))
        z_idx_cutout = int(np.floor((z_coord_los - min_z_cutout) / los_bin_size))

        # Only include particles that fall within the cutout dimensions
        if 0 <= x_idx_cutout < num_pixels_x_cutout and 0 <= y_idx_cutout < num_pixels_y_cutout and 0 <= z_idx_cutout < num_pixels_z_cutout:
            star_particle_membership[y_idx_cutout][x_idx_cutout][z_idx_cutout].append(i) # Removed line_of_sight_distance
            star_mass_density_map[y_idx_cutout][x_idx_cutout][z_idx_cutout] += particle_masses[i]

    # --- Calculate star particles in front of each grid cell ---
    star_particles_in_front_of_grid = [[[[] for _ in range(num_pixels_z_cutout)]
                                       for _ in range(num_pixels_x_cutout)]
                                      for _ in range(num_pixels_y_cutout)]

    for y_idx in range(num_pixels_y_cutout):
        for x_idx in range(num_pixels_x_cutout):
            particles_in_previous_bins = []
            for z_idx in range(num_pixels_z_cutout):
                star_particles_in_front_of_grid[y_idx][x_idx][z_idx] = list(particles_in_previous_bins)

                for original_idx in star_particle_membership[y_idx][x_idx][z_idx]: # Iterate directly over original_idx
                    particles_in_previous_bins.append(original_idx)
                particles_in_previous_bins = sorted(list(set(particles_in_previous_bins)))

    # --- Set central pixel coordinate to the geometric center of the cutout ---
    central_pixel_x = num_pixels_x_cutout // 2
    central_pixel_y = num_pixels_y_cutout // 2
    central_pixel_coords = (central_pixel_x, central_pixel_y)

    # --- Process GAS particles if provided ---
    gas_particle_membership = None
    gas_mass_density_map = None
    gas_particles_in_front_of_grid = None

    if gas_coords is not None:
        gas_particle_membership = [[[[] for _ in range(num_pixels_z_cutout)]
                                    for _ in range(num_pixels_x_cutout)]
                                   for _ in range(num_pixels_y_cutout)]
        gas_mass_density_map = np.zeros((num_pixels_y_cutout, num_pixels_x_cutout, num_pixels_z_cutout), dtype=float)
        projected_gas_2d_coords = rotated_gas_coords[:, :2]
        # gas_line_of_sight_distances_normalized is already calculated above

        for i in range(gas_coords.shape[0]):
            x_coord_proj = projected_gas_2d_coords[i, 0]
            y_coord_proj = projected_gas_2d_coords[i, 1]
            z_coord_los = gas_line_of_sight_distances_normalized[i] # Use normalized distance

            x_idx_cutout = int(np.floor((x_coord_proj - min_x_cutout) / pixel_size))
            y_idx_cutout = int(np.floor((y_coord_proj - min_y_cutout) / pixel_size))
            z_idx_cutout = int(np.floor((z_coord_los - min_z_cutout) / los_bin_size))

            if 0 <= x_idx_cutout < num_pixels_x_cutout and 0 <= y_idx_cutout < num_pixels_y_cutout and 0 <= z_idx_cutout < num_pixels_z_cutout:
                gas_particle_membership[y_idx_cutout][x_idx_cutout][z_idx_cutout].append(i) # Removed line_of_sight_distance
                gas_mass_density_map[y_idx_cutout][x_idx_cutout][z_idx_cutout] += gas_masses[i]

        # --- Calculate gas particles in front of each grid cell ---
        gas_particles_in_front_of_grid = [[[[] for _ in range(num_pixels_z_cutout)]
                                           for _ in range(num_pixels_x_cutout)]
                                          for _ in range(num_pixels_y_cutout)]

        for y_idx in range(num_pixels_y_cutout):
            for x_idx in range(num_pixels_x_cutout):
                particles_in_previous_bins = []
                for z_idx in range(num_pixels_z_cutout):
                    gas_particles_in_front_of_grid[y_idx][x_idx][z_idx] = list(particles_in_previous_bins)

                    for original_idx in gas_particle_membership[y_idx][x_idx][z_idx]: # Iterate directly over original_idx
                        particles_in_previous_bins.append(original_idx)
                    particles_in_previous_bins = sorted(list(set(particles_in_previous_bins)))


    # --- Prepare grid information for output ---
    grid_info = {
        'min_x_proj': min_x_cutout,
        'min_y_proj': min_y_cutout,
        'min_z_proj': min_z_cutout, # This will now be relative to the closest particle
        'num_pixels_x': num_pixels_x_cutout,
        'num_pixels_y': num_pixels_y_cutout,
        'num_pixels_z': num_pixels_z_cutout,
        'effective_pixel_size_x': pixel_size,
        'effective_pixel_size_y': pixel_size,
        'effective_los_bin_size': los_bin_size
    }

    return star_particle_membership, star_mass_density_map, central_pixel_coords, grid_info, gas_particle_membership, gas_mass_density_map, gas_particles_in_front_of_grid, star_particles_in_front_of_grid



def get_2d_density_projection_with_los_binning_old(star_coords, particle_masses, pixel_size, output_dimension, los_bin_size,
                                               los_dimension, proj_angle_deg=0, projection_mode=None, gas_coords=None, gas_masses=None):
    """
    Calculates the 3D mass density projection of star particles given 3D coordinates,
    their masses, fixed pixel/bin sizes, and desired output dimensions for all axes.
    The output grid is geometrically centered around the most massive pixel of the
    full projected 2D star map (after LOS summation). It also calculates the projected distance
    along the line of sight for each star particle within its 3D bin, and outputs the
    geometric central pixel coordinate of the *summed* 2D star mass map.

    If `gas_coords` and `gas_masses` are provided, it also calculates the membership of gas particles
    and their mass density within the *same* projected 3D pixel gridding defined by the star particles.

    If `projection_mode` is specified, it overrides `proj_angle_deg`.
    The `proj_angle_deg` is measured counter-clockwise from the positive Y-axis.
    The line-of-sight distance corresponds to the new Z-coordinate after this transformation.

    The line-of-sight distance for each particle is normalized such that the closest
    particle (considering both stars and gas, if gas is provided) has a line-of-sight
    distance of 0, and distances increase farther away from this closest point.

    Args:
        star_coords (np.ndarray): A NumPy array of shape (N, 3) representing the
                                  (x, y, z) coordinates of N star particles.
        particle_masses (np.ndarray): A NumPy array of shape (N,) representing the
                                      masses of N star particles.
        pixel_size (float): The size of each pixel in the 2D grid (x and y dimensions).
                            Must be positive.
        output_dimension (tuple): A tuple (width, height) specifying the desired total
                                  physical width and height of the 2D output map.
                                  Each dimension must be positive.
        los_bin_size (float): The size of each bin along the line-of-sight (z) dimension.
                              Must be positive.
        los_dimension (float): The desired total physical depth along the line-of-sight.
                               Must be positive.
        proj_angle_deg (float, optional): The projection angle in degrees, measured from the
                                          positive Z-axis (rotation around Y-axis).
                                          Defaults to 0. Ignored if `projection_mode` is provided.
        projection_mode (str, optional): A string specifying a predefined orthogonal projection.
                                         Can be 'XYZ', 'YXZ', 'ZYX', or 'YZX'.
                                         If provided, it overrides `proj_angle_deg`.
                                         Defaults to None.
        gas_coords (np.ndarray, optional): A NumPy array of shape (M, 3) representing the
                                           (x, y, z) coordinates of M gas particles.
                                           Defaults to None.
        gas_masses (np.ndarray, optional): A NumPy array of shape (M,) representing the
                                           masses of M gas particles. Must be provided if
                                           `gas_coords` is provided. Defaults to None.

    Returns:
        tuple: A tuple containing:
            - star_particle_membership (list of lists of lists of lists): A 3D list (grid) where
              each element `star_particle_membership[y_idx][x_idx][z_idx]` is a list of
              tuples `(original_particle_index, line_of_sight_distance)`
              for star particles that fall into that specific 3D bin. The `line_of_sight_distance`
              is now relative to the closest particle (0 being the closest).
            - star_mass_density_map (np.ndarray): A 3D NumPy array representing the mass
              density map (sum of masses per 3D bin) for star particles.
              Its shape is (num_pixels_y, num_pixels_x, num_pixels_z).
            - central_pixel_coords (tuple): A tuple (x_pixel_coord, y_pixel_coord)
              representing the geometric central pixel coordinate of the *summed* 2D map.
            - grid_info (dict): A dictionary containing information about the grid:
              'min_x_proj': The minimum x-coordinate of the projected cutout.
              'min_y_proj': The minimum y-coordinate of the projected cutout.
              'min_z_proj': The minimum z-coordinate (line-of-sight) of the projected cutout.
              'num_pixels_x': The number of pixels along the x-dimension.
              'num_pixels_y': The number of pixels along the y-dimension.
              'num_pixels_z': The number of pixels along the z-dimension (line-of-sight).
              'effective_pixel_size_x': The actual pixel size used for x-dimension (same as pixel_size).
              'effective_pixel_size_y': The actual pixel size used for y-dimension (same as pixel_size).
              'effective_los_bin_size': The actual bin size used for z-dimension (same as los_bin_size).
            - gas_particle_membership (list of lists of lists of lists or None): Same structure as
              `star_particle_membership` but for gas particles. None if `gas_coords` not provided.
            - gas_mass_density_map (np.ndarray or None): A 3D NumPy array representing the mass
              density map for gas particles. Its shape is (num_pixels_y, num_pixels_x, num_pixels_z).
              None if `gas_coords` not provided.
            - gas_particles_in_front_of_grid (list of lists of lists of lists or None): A 3D list (grid) where
              each element `gas_particles_in_front_of_grid[y_idx][x_idx][z_idx]` is a list of
              `original_particle_index` for all gas particles in bins *in front* of the current
              `z_idx` along the line of sight for that (y_idx, x_idx) column. None if `gas_coords` not provided.
            - star_particles_in_front_of_grid (list of lists of lists of lists or None): A 3D list (grid) where
              each element `star_particles_in_front_of_grid[y_idx][x_idx][z_idx]` is a list of
              `original_particle_index` for all star particles in bins *in front* of the current
              `z_idx` along the line of sight for that (y_idx, x_idx) column. None if no star particles.

    Raises:
        ValueError: If star_coords/particle_masses are invalid, pixel_size/los_bin_size are not positive,
                    output_dimension/los_dimension are invalid, an invalid `projection_mode` is provided,
                    or `gas_coords` is provided without `gas_masses` (or vice-versa).
    """
    # --- Input Validation ---
    if not isinstance(star_coords, np.ndarray) or star_coords.ndim != 2 or star_coords.shape[1] != 3:
        raise ValueError("star_coords must be a NumPy array of shape (N, 3).")
    if not isinstance(particle_masses, np.ndarray) or particle_masses.ndim != 1 or particle_masses.shape[0] != star_coords.shape[0]:
        raise ValueError("particle_masses must be a 1D NumPy array of shape (N,) matching star_coords.")
    if pixel_size <= 0:
        raise ValueError("pixel_size must be positive.")
    if los_bin_size <= 0:
        raise ValueError("los_bin_size must be positive.")
    if not (isinstance(output_dimension, tuple) and len(output_dimension) == 2 and
            output_dimension[0] > 0 and output_dimension[1] > 0):
        raise ValueError("'output_dimension' must be a tuple (width, height) of two positive numbers.")
    if not (isinstance(los_dimension, (int, float)) and los_dimension > 0):
        raise ValueError("'los_dimension' must be a positive number.")

    if (gas_coords is not None and gas_masses is None) or \
       (gas_coords is None and gas_masses is not None):
        raise ValueError("Both 'gas_coords' and 'gas_masses' must be provided together, or neither.")
    if gas_coords is not None:
        if not (isinstance(gas_coords, np.ndarray) or gas_coords.ndim != 2 or gas_coords.shape[1] != 3):
            raise ValueError("gas_coords must be a NumPy array of shape (M, 3).")
        if not (isinstance(gas_masses, np.ndarray) or gas_masses.ndim != 1 or gas_masses.shape[0] != gas_coords.shape[0]):
            raise ValueError("gas_masses must be a 1D NumPy array of shape (M,) matching gas_coords.")

    # --- Handle Empty Particle Set (Stars) ---
    if star_coords.shape[0] == 0:
        print("Warning: No star particles provided. Returning empty results for LOS binning.")
        return [], np.array([[]]), (0, 0), {'min_x_proj': 0, 'min_y_proj': 0, 'min_z_proj': 0,
                                            'num_pixels_x': 0, 'num_pixels_y': 0, 'num_pixels_z': 0,
                                            'effective_pixel_size_x': pixel_size, 'effective_pixel_size_y': pixel_size, 'effective_los_bin_size': los_bin_size}, [], np.array([[]]), None, None

    # --- Determine transformation based on projection_mode or proj_angle_deg ---
    if projection_mode:
        mode_map = {
            'XYZ': (0, 1, 2), # New X=Orig X, New Y=Orig Y, LOS=Orig Z
            'YXZ': (1, 0, 2), # New X=Orig Y, New Y=Orig X, LOS=Orig Z
            'ZYX': (2, 1, 0), # New X=Orig Z, New Y=Orig Y, LOS=Orig X
            'YZX': (1, 2, 0)  # New X=Orig Y, New Y=Orig Z, LOS=Orig X
        }
        if projection_mode not in mode_map:
            raise ValueError(f"Invalid projection_mode: '{projection_mode}'. "
                             "Accepted modes are 'XYZ', 'YXZ', 'ZYX', 'YZX'.")
        x_idx, y_idx, z_los_idx = mode_map[projection_mode]
        rotated_star_coords = star_coords[:, [x_idx, y_idx, z_los_idx]]
        if gas_coords is not None:
            rotated_gas_coords = gas_coords[:, [x_idx, y_idx, z_los_idx]]
    else:
        proj_angle_rad = np.deg2rad(proj_angle_deg)
        cos_theta = np.cos(proj_angle_rad)
        sin_theta = np.sin(proj_angle_rad)
        rotation_matrix = np.array([
            [cos_theta, 0, sin_theta],
            [0, 1, 0],
            [-sin_theta, 0, cos_theta]
        ])
        rotated_star_coords = np.dot(star_coords, rotation_matrix.T)
        if gas_coords is not None:
            rotated_gas_coords = np.dot(gas_coords, rotation_matrix.T)

    # --- Extract projected 2D coordinates and line-of-sight distances for STARS ---
    projected_star_2d_coords = rotated_star_coords[:, :2]
    star_line_of_sight_distances = rotated_star_coords[:, 2]

    # --- Determine the global minimum LOS distance for normalization ---
    all_los_distances = star_line_of_sight_distances
    if gas_coords is not None:
        gas_line_of_sight_distances_raw = rotated_gas_coords[:, 2]
        all_los_distances = np.concatenate((all_los_distances, gas_line_of_sight_distances_raw))

    min_global_los = np.min(all_los_distances) if all_los_distances.size > 0 else 0.0

    # --- Normalize line-of-sight distances ---
    star_line_of_sight_distances_normalized = star_line_of_sight_distances - min_global_los
    if gas_coords is not None:
        gas_line_of_sight_distances_normalized = gas_line_of_sight_distances_raw - min_global_los

    # --- Calculate the extent of the *entire* projected STAR dataset for global map ---
    # Use normalized LOS distances for min_z_full and max_z_full
    min_x_full, min_y_full = np.min(projected_star_2d_coords, axis=0)
    max_x_full, max_y_full = np.max(projected_star_2d_coords, axis=0)
    min_z_full, max_z_full = np.min(star_line_of_sight_distances_normalized), np.max(star_line_of_sight_distances_normalized)


    # Define a temporary pixel size for the global map if it's too fine/coarse
    effective_pixel_size_global = pixel_size
    effective_los_bin_size_global = los_bin_size
    epsilon_global_xy = 1e-9 * effective_pixel_size_global
    epsilon_global_z = 1e-9 * effective_los_bin_size_global

    num_pixels_x_global = int(np.ceil((max_x_full - min_x_full + epsilon_global_xy) / effective_pixel_size_global))
    num_pixels_y_global = int(np.ceil((max_y_full - min_y_full + epsilon_global_xy) / effective_pixel_size_global))
    num_pixels_z_global = int(np.ceil((max_z_full - min_z_full + epsilon_global_z) / effective_los_bin_size_global))

    if num_pixels_x_global == 0: num_pixels_x_global = 1
    if num_pixels_y_global == 0: num_pixels_y_global = 1
    if num_pixels_z_global == 0: num_pixels_z_global = 1

    # --- Create a global 3D mass density map (STARS) ---
    global_star_mass_density_map_3d = np.zeros((num_pixels_y_global, num_pixels_x_global, num_pixels_z_global), dtype=float)

    for i in range(star_coords.shape[0]):
        x_coord_proj = projected_star_2d_coords[i, 0]
        y_coord_proj = projected_star_2d_coords[i, 1]
        z_coord_los = star_line_of_sight_distances_normalized[i] # Use normalized distance

        x_idx_global = int(np.floor((x_coord_proj - min_x_full) / effective_pixel_size_global))
        y_idx_global = int(np.floor((y_coord_proj - min_y_full) / effective_pixel_size_global))
        z_idx_global = int(np.floor((z_coord_los - min_z_full) / effective_los_bin_size_global))

        # Clip indices to ensure they are within the valid range for the global map
        x_idx_global = np.clip(x_idx_global, 0, num_pixels_x_global - 1)
        y_idx_global = np.clip(y_idx_global, 0, num_pixels_y_global - 1)
        z_idx_global = np.clip(z_idx_global, 0, num_pixels_z_global - 1)

        global_star_mass_density_map_3d[y_idx_global][x_idx_global][z_idx_global] += particle_masses[i]

    # --- Find the most massive pixel in the 2D summed global STAR map ---
    global_star_mass_density_map_2d_summed = np.sum(global_star_mass_density_map_3d, axis=2)

    most_massive_pixel_x_idx_global = 0
    most_massive_pixel_y_idx_global = 0

    if np.sum(global_star_mass_density_map_2d_summed) > 0: # Only search if there's mass
        flat_idx = np.argmax(global_star_mass_density_map_2d_summed)
        most_massive_pixel_y_idx_global, most_massive_pixel_x_idx_global = np.unravel_index(flat_idx, global_star_mass_density_map_2d_summed.shape)

    # Convert most massive pixel index to its physical center coordinate in the global 2D space
    most_massive_pixel_x_coord_global = min_x_full + (most_massive_pixel_x_idx_global + 0.5) * effective_pixel_size_global
    most_massive_pixel_y_coord_global = min_y_full + (most_massive_pixel_y_idx_global + 0.5) * effective_pixel_size_global

    # For LOS dimension, we'll center around the median of the full LOS distances of STARS
    # Use normalized median for centering
    median_z_full = np.median(star_line_of_sight_distances_normalized)

    # --- Define the cutout grid's extent and dimensions based on most massive STAR pixel and LOS median ---
    num_pixels_x_cutout = int(np.ceil(output_dimension[0] / pixel_size))
    num_pixels_y_cutout = int(np.ceil(output_dimension[1] / pixel_size))
    num_pixels_z_cutout = int(np.ceil(los_dimension / los_bin_size))

    if num_pixels_x_cutout == 0: num_pixels_x_cutout = 1
    if num_pixels_y_cutout == 0: num_pixels_y_cutout = 1
    if num_pixels_z_cutout == 0: num_pixels_z_cutout = 1

    # Calculate the minimum projected coordinates for the cutout,
    # ensuring its geometric center aligns with the most massive pixel (XY) and LOS median (Z).
    min_x_cutout = most_massive_pixel_x_coord_global - (num_pixels_x_cutout * pixel_size / 2.0)
    min_y_cutout = most_massive_pixel_y_coord_global - (num_pixels_y_cutout * pixel_size / 2.0)
    # min_z_cutout should now be relative to the normalized LOS distances
    min_z_cutout = median_z_full - (num_pixels_z_cutout * los_bin_size / 2.0)


    # --- Initialize output arrays for the cutout region (STARS) ---
    star_particle_membership = [[[[] for _ in range(num_pixels_z_cutout)]
                            for _ in range(num_pixels_x_cutout)]
                           for _ in range(num_pixels_y_cutout)]
    star_mass_density_map = np.zeros((num_pixels_y_cutout, num_pixels_x_cutout, num_pixels_z_cutout), dtype=float)

    # --- Assign STAR particles to the cutout grid ---
    for i in range(star_coords.shape[0]):
        x_coord_proj = projected_star_2d_coords[i, 0]
        y_coord_proj = projected_star_2d_coords[i, 1]
        z_coord_los = star_line_of_sight_distances_normalized[i] # Use normalized distance

        x_idx_cutout = int(np.floor((x_coord_proj - min_x_cutout) / pixel_size))
        y_idx_cutout = int(np.floor((y_coord_proj - min_y_cutout) / pixel_size))
        z_idx_cutout = int(np.floor((z_coord_los - min_z_cutout) / los_bin_size))

        # Only include particles that fall within the cutout dimensions
        if 0 <= x_idx_cutout < num_pixels_x_cutout and 0 <= y_idx_cutout < num_pixels_y_cutout and 0 <= z_idx_cutout < num_pixels_z_cutout:
            star_particle_membership[y_idx_cutout][x_idx_cutout][z_idx_cutout].append((i, star_line_of_sight_distances_normalized[i]))
            star_mass_density_map[y_idx_cutout][x_idx_cutout][z_idx_cutout] += particle_masses[i]

    # --- Calculate star particles in front of each grid cell ---
    star_particles_in_front_of_grid = [[[[] for _ in range(num_pixels_z_cutout)]
                                       for _ in range(num_pixels_x_cutout)]
                                      for _ in range(num_pixels_y_cutout)]

    for y_idx in range(num_pixels_y_cutout):
        for x_idx in range(num_pixels_x_cutout):
            particles_in_previous_bins = []
            for z_idx in range(num_pixels_z_cutout):
                star_particles_in_front_of_grid[y_idx][x_idx][z_idx] = list(particles_in_previous_bins)

                for original_idx, _ in star_particle_membership[y_idx][x_idx][z_idx]:
                    particles_in_previous_bins.append(original_idx)
                particles_in_previous_bins = sorted(list(set(particles_in_previous_bins)))

    # --- Set central pixel coordinate to the geometric center of the cutout ---
    central_pixel_x = num_pixels_x_cutout // 2
    central_pixel_y = num_pixels_y_cutout // 2
    central_pixel_coords = (central_pixel_x, central_pixel_y)

    # --- Process GAS particles if provided ---
    gas_particle_membership = None
    gas_mass_density_map = None
    gas_particles_in_front_of_grid = None

    if gas_coords is not None:
        gas_particle_membership = [[[[] for _ in range(num_pixels_z_cutout)]
                                    for _ in range(num_pixels_x_cutout)]
                                   for _ in range(num_pixels_y_cutout)]
        gas_mass_density_map = np.zeros((num_pixels_y_cutout, num_pixels_x_cutout, num_pixels_z_cutout), dtype=float)
        projected_gas_2d_coords = rotated_gas_coords[:, :2]
        # gas_line_of_sight_distances_normalized is already calculated above

        for i in range(gas_coords.shape[0]):
            x_coord_proj = projected_gas_2d_coords[i, 0]
            y_coord_proj = projected_gas_2d_coords[i, 1]
            z_coord_los = gas_line_of_sight_distances_normalized[i] # Use normalized distance

            x_idx_cutout = int(np.floor((x_coord_proj - min_x_cutout) / pixel_size))
            y_idx_cutout = int(np.floor((y_coord_proj - min_y_cutout) / pixel_size))
            z_idx_cutout = int(np.floor((z_coord_los - min_z_cutout) / los_bin_size))

            if 0 <= x_idx_cutout < num_pixels_x_cutout and 0 <= y_idx_cutout < num_pixels_y_cutout and 0 <= z_idx_cutout < num_pixels_z_cutout:
                gas_particle_membership[y_idx_cutout][x_idx_cutout][z_idx_cutout].append((i, gas_line_of_sight_distances_normalized[i]))
                gas_mass_density_map[y_idx_cutout][x_idx_cutout][z_idx_cutout] += gas_masses[i]

        # --- Calculate gas particles in front of each grid cell ---
        gas_particles_in_front_of_grid = [[[[] for _ in range(num_pixels_z_cutout)]
                                           for _ in range(num_pixels_x_cutout)]
                                          for _ in range(num_pixels_y_cutout)]

        for y_idx in range(num_pixels_y_cutout):
            for x_idx in range(num_pixels_x_cutout):
                particles_in_previous_bins = []
                for z_idx in range(num_pixels_z_cutout):
                    gas_particles_in_front_of_grid[y_idx][x_idx][z_idx] = list(particles_in_previous_bins)

                    for original_idx, _ in gas_particle_membership[y_idx][x_idx][z_idx]:
                        particles_in_previous_bins.append(original_idx)
                    particles_in_previous_bins = sorted(list(set(particles_in_previous_bins)))


    # --- Prepare grid information for output ---
    grid_info = {
        'min_x_proj': min_x_cutout,
        'min_y_proj': min_y_cutout,
        'min_z_proj': min_z_cutout, # This will now be relative to the closest particle
        'num_pixels_x': num_pixels_x_cutout,
        'num_pixels_y': num_pixels_y_cutout,
        'num_pixels_z': num_pixels_z_cutout,
        'effective_pixel_size_x': pixel_size,
        'effective_pixel_size_y': pixel_size,
        'effective_los_bin_size': los_bin_size
    }

    return star_particle_membership, star_mass_density_map, central_pixel_coords, grid_info, gas_particle_membership, gas_mass_density_map, gas_particles_in_front_of_grid, star_particles_in_front_of_grid



def construct_SFH_TNG(stars_form_lbt, stars_init_mass, stars_metallicity, del_t=0.3, max_lbt=14.0):
    """
    Constructs the Star Formation History (SFH) from stellar formation times,
    initial masses, and metallicities using vectorized NumPy operations for efficiency.

    Calculates the mass-weighted average metallicity and mass-weighted average age
    within each lookback time bin.

    Args:
        stars_form_lbt (array-like): Array of stellar formation lookback times in Gyr.
        stars_init_mass (array-like): Array of initial stellar masses in solar mass,
                                       corresponding to stars_form_lbt.
        stars_metallicity (array-like): Array of stellar metallicities (e.g., [Fe/H] or Z),
                                        corresponding to stars_form_lbt.
        del_t (float, optional): Time bin width in Gyr. Defaults to 0.3 Gyr.
        max_lbt (float, optional): Maximum lookback time to consider for binning.
                                   Defaults to 18.0 Gyr.

    Returns:
        dict: A dictionary containing:
              - 'lbt' (np.ndarray): Midpoints of lookback time bins where stars formed.
              - 'sfr' (np.ndarray): Star formation rate in solar mass per year.
              - 'nstars' (np.ndarray): Number of stars formed in each bin.
              - 'mass' (np.ndarray): Total initial stellar mass formed within each bin.
              - 'cumul_mass' (np.ndarray): Stellar mass growth history (total initial mass not yet formed).
              - 'metallicity' (np.ndarray): Mass-weighted average metallicity in each bin.
              - 'mass_weighted_age' (np.ndarray): Mass-weighted average stellar population age (lookback time) in each bin.

    Raises:
        ValueError: If input arrays have incompatible shapes or invalid values.
    """
    stars_form_lbt = np.asarray(stars_form_lbt)
    stars_init_mass = np.asarray(stars_init_mass)
    stars_metallicity = np.asarray(stars_metallicity)

    # --- Input Validation ---
    if not (stars_form_lbt.shape == stars_init_mass.shape == stars_metallicity.shape and stars_form_lbt.ndim == 1):
        raise ValueError("stars_form_lbt, stars_init_mass, and stars_metallicity must be 1D arrays of the same shape.")
    if del_t <= 0:
        raise ValueError("del_t (time bin width) must be positive.")
    if max_lbt <= 0:
        raise ValueError("max_lbt (maximum lookback time) must be positive.")

    # Handle empty particle set
    if stars_form_lbt.shape[0] == 0:
        print("Warning: No star particles provided. Returning empty SFH.")
        return {
            'lbt': np.array([]),
            'sfr': np.array([]),
            'nstars': np.array([]),
            'mass': np.array([]),
            'cumul_mass': np.array([]),
            'metallicity': np.array([]),
            'mass_weighted_age': np.array([])
        }

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

    # Calculate sum of (mass * metallicity) in each bin for mass-weighted average metallicity
    mass_times_metallicity_in_bins, _ = np.histogram(stars_form_lbt, bins=bins, weights=stars_init_mass * stars_metallicity)

    # Calculate sum of (mass * lookback_time) in each bin for mass-weighted average age
    mass_times_lbt_in_bins, _ = np.histogram(stars_form_lbt, bins=bins, weights=stars_init_mass * stars_form_lbt)

    # Identify valid bins (where at least one star was formed, or mass is non-zero)
    # Using mass_in_bins > 0 is more robust for mass-weighted averages.
    valid_bins_mask = mass_in_bins > 0

    # Extract data for valid bins
    valid_masses = mass_in_bins[valid_bins_mask]
    valid_nstars = nstars_in_bins[valid_bins_mask]
    valid_mass_times_metallicity = mass_times_metallicity_in_bins[valid_bins_mask]
    valid_mass_times_lbt = mass_times_lbt_in_bins[valid_bins_mask] # Extract for valid bins

    # Calculate lookback time midpoints for the sfh dictionary
    # bins[:-1] gives the start times of the bins
    sfh_lbt_midpoints = bins[:-1][valid_bins_mask] + 0.5 * del_t

    # Calculate Star Formation Rate (SFR)
    # SFR is mass formed per unit time, converted to solar mass per year (1e9 for Gyr to year)
    sfh_sfr = valid_masses / del_t / 1e9

    # Total stellar mass in each bin
    sfh_total_stellar_mass_in_bin = valid_masses

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

    # Calculate Mass-Weighted Average Metallicity
    # Avoid division by zero for bins with no mass (though valid_bins_mask should handle this)
    sfh_metallicity = np.zeros_like(valid_masses)
    non_zero_mass_mask = valid_masses > 0
    sfh_metallicity[non_zero_mass_mask] = valid_mass_times_metallicity[non_zero_mass_mask] / valid_masses[non_zero_mass_mask]

    # Calculate Mass-Weighted Average Age
    sfh_mass_weighted_age = np.zeros_like(valid_masses)
    sfh_mass_weighted_age[non_zero_mass_mask] = valid_mass_times_lbt[non_zero_mass_mask] / valid_masses[non_zero_mass_mask]


    # Construct the sfh dictionary
    sfh = {
        'lbt': sfh_lbt_midpoints,
        'sfr': sfh_sfr,
        'nstars': valid_nstars,
        'mass': sfh_total_stellar_mass_in_bin,
        'cumul_mass': sfh_smgh,
        'metallicity': sfh_metallicity,
        'mass_weighted_age': sfh_mass_weighted_age
    }

    return sfh


def get_filter_transmission_pixedfit(filters):
    from piXedfit.utils.filtering import get_filter_curve, cwave_filters

    filter_transmission = {}
    filter_wave_eff = {}
    for ff in filters:
        filter_wave_eff[ff] = cwave_filters(ff)

        w, t = get_filter_curve(ff)
        filter_transmission[ff] = {}
        filter_transmission[ff]['wave'] = w
        filter_transmission[ff]['trans'] = t

    return filter_transmission, filter_wave_eff

