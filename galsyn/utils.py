import sys, os
import numpy as np
from operator import itemgetter

def define_cosmo(cosmo_str):
    """
    Selects and returns a specific cosmology model from the astropy.cosmology library.

    Parameters
    ----------
    cosmo_str : str
        The name of the cosmology model to be used.
        Valid options are: "planck18", "planck15", "planck13",
        "wmap5", "wmap7", or "wmap9". The check is case-insensitive.

    Returns
    -------
    cosmology object
        An astropy.cosmology object containing the parameters for the
        selected model.
    """

    cosmo_name = cosmo_str.lower()  # Make case-insensitive

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

    return cosmo

def interp_age_univ_from_z(z, cosmo):

    """
    Interpolates the age of the universe at a given redshift (z).

    This function first creates an array of redshifts and calculates the age of the universe
    for each redshift using a provided cosmology object. It then creates an interpolation
    function and uses it to find the age of the universe at the specific redshift(s)
    provided as input.

    Parameters
    ----------
    z : float or array_like
        The redshift(s) at which to calculate the age of the universe.

    cosmo : astropy.cosmology object
        The cosmology object to use for the age calculation. This object
        should be an instance from the `astropy.cosmology` module.

    Returns
    -------
    float or ndarray
        The age(s) of the universe at the given redshift(s), in Gyr.
    """
    
    from scipy.interpolate import interp1d
    
    arr_z = np.linspace(0.0, 40.0, 1000)
    # arr_z = np.linspace(0.0, 10.0, 100)
    age_univ = cosmo.age(arr_z).value

    f = interp1d(arr_z, age_univ)#, fill_value="extrapolate")
    return f(z)

def cosmo_redshifting(wave_rest, L_lambda_rest, z, cosmo):
    r"""
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
    # FIX: Use np.clip to handle extreme large values that could cause overflow
    # Use L_lambda_rest_safe to avoid modifying the input array
    L_lambda_rest_safe = np.asarray(L_lambda_rest)
    
    # Clip luminosity values to prevent overflow before multiplication.
    # The maximum value for L_lambda_rest that won't overflow when multiplied by 
    # L_sun (approx 3.8e33) is around 4.7e27. We use a slightly safer estimate.
    L_sun_erg = L_sun.to('erg/s').value
    max_safe_l_rest = 1e300 / L_sun_erg 
    L_lambda_rest_clipped = np.clip(L_lambda_rest_safe, 0.0, max_safe_l_rest)

    # Perform the conversion, suppressing overflow warnings during this line only
    with np.errstate(over='ignore'):
        L_lambda_erg = L_lambda_rest_clipped * L_sun_erg

    # After multiplication, treat any resulting 'inf' values (if clipping wasn't enough) as 0.0
    L_lambda_erg[np.isinf(L_lambda_erg)] = 0.0 


    # Get luminosity distance in cm
    D_L_cm = cosmo.luminosity_distance(z).to('cm').value

    # Compute observed flux density using:
    # F_lambda = (1 / (4 * pi * D_L^2)) * (1 / (1 + z)) * L_lambda_rest
    F_lambda_obs = (L_lambda_erg / (4 * np.pi * D_L_cm**2)) / (1 + z)  # in erg/s/cm^2/Angstrom
    
    # Also ensure any remaining NaN/inf/very large values are cleaned up, though the clipping should help
    F_lambda_obs[~np.isfinite(F_lambda_obs)] = 0.0

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

    # FIX: Add error handling for the multiplication in the integral to catch NaNs/Infs
    with np.errstate(invalid='ignore', divide='ignore', over='ignore'):
        integrand = F_lambda * T_lambda * wave_common
        # Replace non-finite values (like NaNs/Infs from invalid ops) with 0 before integration
        integrand[~np.isfinite(integrand)] = 0.0 

    # Check if trapezoid exists, otherwise fallback to trapz
    if hasattr(np, 'trapezoid'):
        nptrapz = np.trapezoid
    else:
        nptrapz = np.trapz
    
    numerator = nptrapz(integrand, wave_common)
    denominator = nptrapz(T_lambda * wave_common, wave_common)

    # Handle division by zero if the filter denominator is zero (i.e., filter has no transmission)
    if denominator == 0.0:
        return 0.0

    flux_phot = numerator / denominator

    return flux_phot


def get_effective_range(profile, mass_percentage_threshold=0.95):
    total_mass = np.sum(profile)
    if total_mass == 0:
        return 0, 0 # Or handle as an error/empty galaxy
    cumulative_mass = np.cumsum(profile)

    # Find the first index where cumulative mass exceeds (1 - threshold)/2 of total mass
    # and the last index where it exceeds total_mass * threshold + (1 - threshold)/2 of total mass
    # This centers the "significant" mass.

    # Find the start index: mass from beginning to this point is (1-threshold)/2
    # Find the end index: mass from beginning to this point is threshold + (1-threshold)/2

    # Alternative: find where cumulative mass first exceeds (1-fraction_to_trim_each_side) of total mass
    # And where it first exceeds fraction_to_trim_each_side

    # More robust for finding a central region:
    # Find the indices corresponding to the inner `mass_percentage_threshold` of mass.

    # Find the first index where cumulative mass is greater than or equal to (1 - mass_percentage_threshold) / 2 * total_mass
    # This is the left/bottom boundary
    lower_bound_idx = np.searchsorted(cumulative_mass, (1 - mass_percentage_threshold) / 2 * total_mass)

    # Find the first index where cumulative mass is greater than or equal to (1 + mass_percentage_threshold) / 2 * total_mass
    # This is the right/top boundary. We add 1 to include that pixel.
    upper_bound_idx = np.searchsorted(cumulative_mass, (1 + mass_percentage_threshold) / 2 * total_mass)

    # Ensure indices are within bounds
    lower_bound_idx = np.clip(lower_bound_idx, 0, len(profile) - 1)
    upper_bound_idx = np.clip(upper_bound_idx, lower_bound_idx + 1, len(profile)) # ensure at least 1 pixel wide

    return lower_bound_idx, upper_bound_idx


def determine_image_size(star_coords, particle_masses, pixel_size, output_dimension, polar_angle_deg, 
                         azimuth_angle_deg, gas_coords, gas_masses, mass_percentage=0.98, max_img_dim=100):

    # Temporarily call get_2d_density_projection_no_los_binning to get grid info for image sizing
    # Pass dummy velocity arrays as they are not needed for image sizing
    out = get_2d_density_projection_no_los_binning(star_coords, particle_masses, pixel_size, output_dimension, 
                                                   polar_angle_deg=polar_angle_deg, azimuth_angle_deg=azimuth_angle_deg, 
                                                   gas_coords=gas_coords, gas_masses=gas_masses, 
                                                   star_vels=np.zeros_like(star_coords), gas_vels=np.zeros_like(gas_coords) if gas_coords is not None else None)
    star_particle_membership, star_mass_density_map, central_pixel_coords, grid_info, gas_particle_membership, gas_mass_density_map, _, _ = out

    # Summing along columns (axis=0) gives profile along X-axis
    mass_profile_x = np.sum(star_mass_density_map, axis=0)

    # Summing along rows (axis=1) gives profile along Y-axis
    mass_profile_y = np.sum(star_mass_density_map, axis=1)

    x_start_idx, x_end_idx = get_effective_range(mass_profile_x, mass_percentage)
    y_start_idx, y_end_idx = get_effective_range(mass_profile_y, mass_percentage)

    # Calculate effective physical dimensions
    effective_width_pixels = x_end_idx - x_start_idx
    effective_height_pixels = y_end_idx - y_start_idx

    effective_width_physical = effective_width_pixels * pixel_size
    effective_height_physical = effective_height_pixels * pixel_size

    # A small buffer can be added to the effective dimensions to avoid cutting off edges exactly
    buffer_factor = 1.1 # e.g., add 10% buffer
    final_output_width = effective_width_physical * buffer_factor
    final_output_height = effective_height_physical * buffer_factor

    # Ensure minimum size to avoid issues with very sparse data
    min_dim_physical = 10 * pixel_size # e.g., at least 10 pixels wide
    final_output_width = max(final_output_width, min_dim_physical)
    final_output_height = max(final_output_height, min_dim_physical)

    img_dim = int(round(max(final_output_width, final_output_height)))
    
    if img_dim > max_img_dim:
         img_dim = max_img_dim

    return img_dim


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

		idx1 = np.where((wave>=3.0*lL) & (wave<lL*(1+z)))
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


def construct_SFH(stars_form_lbt, stars_init_mass, stars_metallicity, del_t=0.3, max_lbt=14.0):
    """
    Constructs the Star Formation History (SFH) from stellar formation times,
    initial masses, and metallicities.

    Calculates the mass-weighted average metallicity within each lookback time bin.
    Ensures all bins are returned, even if empty, to maintain consistent binning across pixels.

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
              - 'lbt' (np.ndarray): Midpoints of all lookback time bins.
              - 'sfr' (np.ndarray): Star formation rate in solar mass per year.
              - 'nstars' (np.ndarray): Number of stars formed in each bin.
              - 'mass' (np.ndarray): Total initial stellar mass formed within each bin.
              - 'cumul_mass' (np.ndarray): Cumulative initial stellar mass formed.
              - 'metallicity' (np.ndarray): Mass-weighted average metallicity in each bin.

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

    # Define the bins for lookback time.
    # The bins are inclusive of the lower bound and exclusive of the upper bound [t, t + del_t).
    # We add a small epsilon to max_lbt to ensure that max_lbt itself is included in a bin
    # if it falls exactly on a bin edge.
    bins = np.arange(0, max_lbt + del_t, del_t)
    sfh_lbt_midpoints = bins[:-1] + 0.5 * del_t

    # Handle empty particle set: return arrays of correct size filled with zeros/NaNs
    num_bins = len(sfh_lbt_midpoints)
    if stars_form_lbt.shape[0] == 0:
        return {
            'lbt': sfh_lbt_midpoints,
            'sfr': np.zeros(num_bins, dtype=np.float32),
            'nstars': np.zeros(num_bins, dtype=np.int32),
            'mass': np.zeros(num_bins, dtype=np.float32),
            'cumul_mass': np.zeros(num_bins, dtype=np.float32), # Will be filled later
            'metallicity': np.full(num_bins, np.nan, dtype=np.float32)
        }

    # Use np.histogram to efficiently bin the data
    # mass_in_bins will correspond to bins from youngest LBT to oldest LBT
    mass_in_bins, _ = np.histogram(stars_form_lbt, bins=bins, weights=stars_init_mass)
    nstars_in_bins, _ = np.histogram(stars_form_lbt, bins=bins)
    mass_times_metallicity_in_bins, _ = np.histogram(stars_form_lbt, bins=bins, weights=stars_init_mass * stars_metallicity)

    # Calculate SFR
    sfh_sfr = mass_in_bins / del_t / 1e9

    # Total stellar mass in each bin
    sfh_total_stellar_mass_in_bin = mass_in_bins

    # Calculate Cumulative Mass Formed (increasing with decreasing lookback time)
    # The bins are ordered from youngest lookback time to oldest lookback time.
    # To get cumulative mass from Big Bang towards present, we need to:
    # 1. Reverse the mass_in_bins array to get mass from oldest to youngest.
    # 2. Compute cumulative sum on this reversed array.
    # 3. Reverse the result back to match the original lbt order (youngest to oldest).
    sfh_cumul_mass = np.cumsum(mass_in_bins[::-1])[::-1]


    # Calculate Mass-Weighted Average Metallicity
    sfh_metallicity = np.full_like(mass_in_bins, np.nan, dtype=np.float32)
    non_zero_mass_mask = mass_in_bins > 0
    sfh_metallicity[non_zero_mass_mask] = mass_times_metallicity_in_bins[non_zero_mass_mask] / mass_in_bins[non_zero_mass_mask]

    # Construct the sfh dictionary
    sfh = {
        'lbt': sfh_lbt_midpoints,
        'sfr': sfh_sfr,
        'nstars': nstars_in_bins,
        'mass': sfh_total_stellar_mass_in_bin,
        'cumul_mass': sfh_cumul_mass,
        'metallicity': sfh_metallicity
    }

    return sfh


def make_filter_transmission_text_pixedfit(filters, output_dir="filters"):
    """
    Creates text files containing filter transmission functions and stores them
    in a specified directory. It uses the piXedfit library to get the filter curves.

    Parameters:
    -----------
    filters : list of str
        List of filter names recognized in piXedfit.
    output_dir : str, optional
        The directory where the filter transmission text files will be saved.
        Defaults to "filters".

    Returns:
    --------
    filter_transmission_path : dict
        A dictionary where keys are filter names and values are the paths to
        the generated text files containing the transmission function.
    """
    try:
        from piXedfit.utils.filtering import get_filter_curve
    except ImportError:
        print("Error: piXedfit library not found. Please install it to use this function.")
        sys.exit(1)

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    else:
        print(f"Directory '{output_dir}' already exists.")

    filter_transmission_path = {}

    for ff in filters:
        try:
            # Get filter curve from piXedfit
            w, t = get_filter_curve(ff)
            
            # Define the output file path for this filter
            file_name = f"{ff}.txt"
            file_path = os.path.join(output_dir, file_name)

            # Save the wavelength and transmission to a text file
            np.savetxt(file_path, np.column_stack((w, t)), fmt='%.6e', header='Wavelength (Angstrom) Transmission')
            
            filter_transmission_path[ff] = file_path
            #print(f"Saved filter transmission for '{ff}' to: {file_path}")

        except Exception as e:
            print(f"Error processing filter '{ff}': {e}")
            # Continue to next filter even if one fails, but log the error
            continue

    return filter_transmission_path


def get_2d_density_projection_no_los_binning(star_coords, particle_masses, pixel_size, output_dimension,
                                             polar_angle_deg=0, azimuth_angle_deg=0, gas_coords=None, gas_masses=None,
                                             star_vels=None, gas_vels=None):
    """
    Calculates the 2D mass density projection of star particles given 3D coordinates,
    their masses, a fixed pixel size, and a desired output dimension (cutout size).
    The output grid is geometrically centered around the most massive pixel of the
    full projected star data. It also calculates the projected distance along the line of sight
    for each star particles within its 2D pixel, and outputs the geometric central pixel coordinate of the cutout.

    If `gas_coords` and `gas_masses` are provided, it also calculates the membership of gas particles
    and their mass density within the *same* projected pixel gridding defined by the star particles.

    The line-of-sight distance for each particle (star or gas) is normalized such that the closest
    particle (considering both stars and gas, if gas is provided) has a line-of-sight
    distance of 0, and distances increase farther away from this closest point.

    The projection is defined by a camera/observer position (viewing angle)
    using polar and azimuth angles. The line-of-sight (LOS) direction is from the
    observer's position towards the origin (0,0,0).

    Args:
        star_coords (np.ndarray): A NumPy array of shape (N, 3) representing the
                                  (x, y, z) coordinates of N star particles.
        particle_masses (np.ndarray): A NumPy array of shape (N,) representing the
                                      masses of N star particles.
        pixel_size (float): The size of each pixel in the 2D grid. Must be positive.
        output_dimension (tuple): A tuple (width, height) specifying the desired total
                                  physical width and height of the 2D output map.
                                  Each dimension must be positive.
        polar_angle_deg (float, optional): The polar angle (inclination) in degrees,
                                           measured from the positive Z-axis. Defaults to 0.
        azimuth_angle_deg (float, optional): The azimuth angle (longitude) in degrees,
                                             measured counter-clockwise from the positive X-axis
                                             in the XY-plane. Defaults to 0.
        gas_coords (np.ndarray, optional): A NumPy array of shape (M, 3) representing the
                                           (x, y, z) coordinates of M gas particles.
                                           Defaults to None.
        gas_masses (np.ndarray, optional): A NumPy array of shape (M,) representing the
                                           masses of M gas particles. Must be provided if
                                           `gas_coords` is provided. Defaults to None.
        star_vels (np.ndarray, optional): A NumPy array of shape (N, 3) representing the
                                          (vx, vy, vz) velocities of N star particles.
                                          Defaults to None.
        gas_vels (np.ndarray, optional): A NumPy array of shape (M, 3) representing the
                                         (vx, vy, vz) velocities of M gas particles.
                                         Defaults to None.


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
            - stars_vel_los_proj (np.ndarray): 1D array of projected line-of-sight velocities for stars.
            - gas_vel_los_proj (np.ndarray): 1D array of projected line-of-sight velocities for gas.

    Raises:
        ValueError: If star_coords/particle_masses are invalid, pixel_size is not positive,
                    output_dimension is invalid, or `gas_coords` is provided without `gas_masses` (or vice-versa).
    """
    # Input Validation
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

    if (star_vels is not None and star_vels.shape != star_coords.shape):
        raise ValueError("star_vels must have the same shape as star_coords if provided.")
    if (gas_vels is not None and gas_vels.shape != gas_coords.shape):
        raise ValueError("gas_vels must have the same shape as gas_coords if provided.")


    # Handle Empty Particle Set (Stars)
    if star_coords.shape[0] == 0:
        print("Warning: No star particles provided. Returning empty results for no LOS binning.")
        return ([], np.array([[]]), (0, 0), {'min_x_proj': 0, 'min_y_proj': 0, 'num_pixels_x': 0, 'num_pixels_y': 0,
                                            'effective_pixel_size_x': pixel_size, 'effective_pixel_size_y': pixel_size}, 
                [], np.array([[]]), np.array([]), np.array([]))

    # Determine transformation based on polar and azimuth angles
    polar_rad = np.deg2rad(polar_angle_deg)
    azimuth_rad = np.deg2rad(azimuth_angle_deg)

    # The line of sight (LOS) vector points from the observer's position towards the origin.
    # The observer's position in spherical coordinates (r, theta, phi) looking at origin (0,0,0)
    # is defined by polar_angle_deg (phi) and azimuth_angle_deg (theta).
    # The vector from origin to observer is (sin(phi)*cos(theta), sin(phi)*sin(theta), cos(phi)).
    # We want this vector to become the new Z-axis in our transformed coordinate system.
    # This means the transformation matrix's third row will be this normalized vector.

    # Calculate the new Z-axis (LOS direction)
    z_axis_new = np.array([
        np.sin(polar_rad) * np.cos(azimuth_rad),
        np.sin(polar_rad) * np.sin(azimuth_rad),
        np.cos(polar_rad)
    ])
    z_axis_new = z_axis_new / np.linalg.norm(z_axis_new) # Ensure it's a unit vector

    # Calculate the new X-axis and Y-axis (forming the projection plane)
    # We need a temporary 'up' vector to cross with the new Z-axis to define the new X-axis.
    # Choose (0,0,1) as a default 'up'. If z_axis_new is parallel to (0,0,1), use (0,1,0) instead.
    temp_up_vector = np.array([0.0, 0.0, 1.0])
    # Check if z_axis_new is parallel to temp_up_vector
    if np.allclose(np.abs(np.dot(z_axis_new, temp_up_vector)), 1.0):
        temp_up_vector = np.array([0.0, 1.0, 0.0]) # Use original Y-axis if LOS is along original Z

    x_axis_new = np.cross(temp_up_vector, z_axis_new)
    # Handle the case where cross product might be zero if temp_up_vector is still parallel (e.g., if z_axis_new was (0,1,0) and temp_up_vector became (0,1,0))
    if np.linalg.norm(x_axis_new) < 1e-9: # If x_axis_new is effectively zero
        # This can happen if z_axis_new is (0,1,0) or (0,-1,0) and temp_up_vector was forced to (0,1,0)
        # In this specific case, use (1,0,0) as x_axis_new
        x_axis_new = np.array([1.0, 0.0, 0.0])
    else:
        x_axis_new = x_axis_new / np.linalg.norm(x_axis_new)

    y_axis_new = np.cross(z_axis_new, x_axis_new) # Ensure y_axis_new is orthogonal to both
    y_axis_new = y_axis_new / np.linalg.norm(y_axis_new)

    # The rotation matrix has the new basis vectors as its rows
    rotation_matrix = np.array([x_axis_new, y_axis_new, z_axis_new])

    # Apply the rotation to star and gas coordinates
    rotated_star_coords = np.dot(star_coords, rotation_matrix.T)
    
    stars_vel_los_proj = np.array([])
    if star_vels is not None:
        rotated_star_vels = np.dot(star_vels, rotation_matrix.T)
        stars_vel_los_proj = rotated_star_vels[:, 2] # Z' component of velocity

    gas_vel_los_proj = np.array([])
    if gas_coords is not None:
        rotated_gas_coords = np.dot(gas_coords, rotation_matrix.T)
        if gas_vels is not None:
            rotated_gas_vels = np.dot(gas_vels, rotation_matrix.T)
            gas_vel_los_proj = rotated_gas_vels[:, 2] # Z' component of velocity


    # Extract projected 2D coordinates and raw line-of-sight distances for STARS
    projected_star_2d_coords = rotated_star_coords[:, :2] # X' and Y'
    star_line_of_sight_distances_raw = rotated_star_coords[:, 2] # Z' (LOS)

    # Determine the global minimum LOS distance for normalization (from all particles)
    all_los_distances = star_line_of_sight_distances_raw
    if gas_coords is not None:
        gas_line_of_sight_distances_raw = rotated_gas_coords[:, 2]
        all_los_distances = np.concatenate((all_los_distances, gas_line_of_sight_distances_raw))

    # Handle case where all_los_distances might be empty (e.g., if both star_coords and gas_coords are empty after initial check)
    min_global_los = np.min(all_los_distances) if all_los_distances.size > 0 else 0.0

    # Normalize line-of-sight distances for STARS
    star_line_of_sight_distances_normalized = star_line_of_sight_distances_raw - min_global_los

    # Current logic to calculate dimensions
    min_x_full, min_y_full = np.min(projected_star_2d_coords, axis=0)
    max_x_full, max_y_full = np.max(projected_star_2d_coords, axis=0)
    
    dim_x = max_x_full - min_x_full
    dim_y = max_y_full - min_y_full

    max_allowed_dim = 3000.0  # kpc
    
    # loop to prune outliers until the bounding box fits
    while dim_x > max_allowed_dim or dim_y > max_allowed_dim:
        # Calculate the center of the current distribution
        center_x = (min_x_full + max_x_full) / 2.0
        center_y = (min_y_full + max_y_full) / 2.0
        
        # Calculate squared distance of all particles from the center
        dist_sq = (projected_star_2d_coords[:, 0] - center_x)**2 + \
                  (projected_star_2d_coords[:, 1] - center_y)**2
        
        # Find the index of the particle furthest from the center
        outlier_idx = np.argmax(dist_sq)
        
        # Remove the outlier from coordinates and masses
        # Note: If particle_masses is used later, ensure it stays synced
        projected_star_2d_coords = np.delete(projected_star_2d_coords, outlier_idx, axis=0)
        particle_masses = np.delete(particle_masses, outlier_idx)
        
        # Recalculate dimensions for the next iteration
        if len(projected_star_2d_coords) == 0:
            break # Safety break if all particles are removed
            
        min_x_full, min_y_full = np.min(projected_star_2d_coords, axis=0)
        max_x_full, max_y_full = np.max(projected_star_2d_coords, axis=0)
        dim_x = max_x_full - min_x_full
        dim_y = max_y_full - min_y_full

    # Proceed with the rest of the function using the pruned coordinates
    num_pixels_x_global = int(np.ceil(dim_x / pixel_size))
    num_pixels_y_global = int(np.ceil(dim_y / pixel_size))

    if num_pixels_x_global == 0: num_pixels_x_global = 1
    if num_pixels_y_global == 0: num_pixels_y_global = 1

    # Create a global mass density map (STARS) to find the most massive pixel
    global_star_mass_density_map = np.zeros((num_pixels_y_global, num_pixels_x_global), dtype=float)

    for i in range(len(projected_star_2d_coords)):
        x_coord_proj = projected_star_2d_coords[i, 0]
        y_coord_proj = projected_star_2d_coords[i, 1]

        x_idx_global = int(np.floor((x_coord_proj - min_x_full) / pixel_size))
        y_idx_global = int(np.floor((y_coord_proj - min_y_full) / pixel_size))

        # Clip indices to ensure they are within the valid range for the global map
        x_idx_global = np.clip(x_idx_global, 0, num_pixels_x_global - 1)
        y_idx_global = np.clip(y_idx_global, 0, num_pixels_y_global - 1)

        global_star_mass_density_map[y_idx_global][x_idx_global] += particle_masses[i]

    # Find the most massive pixel in the global STAR map
    most_massive_pixel_x_idx_global = 0
    most_massive_pixel_y_idx_global = 0

    if np.sum(global_star_mass_density_map) > 0: # Only search if there's mass
        # Find the index of the maximum value in the flattened array
        flat_idx = np.argmax(global_star_mass_density_map)
        most_massive_pixel_y_idx_global, most_massive_pixel_x_idx_global = np.unravel_index(flat_idx, global_star_mass_density_map.shape)

    # Convert most massive pixel index to its physical center coordinate in the global space
    most_massive_pixel_x_coord_global = min_x_full + (most_massive_pixel_x_idx_global + 0.5) * pixel_size
    most_massive_pixel_y_coord_global = min_y_full + (most_massive_pixel_y_idx_global + 0.5) * pixel_size


    # Define the cutout grid's extent and dimensions based on most massive STAR pixel
    num_pixels_x_cutout = int(np.ceil(output_dimension[0] / pixel_size))
    num_pixels_y_cutout = int(np.ceil(output_dimension[1] / pixel_size))

    if num_pixels_x_cutout == 0: num_pixels_x_cutout = 1
    if num_pixels_y_cutout == 0: num_pixels_y_cutout = 1

    # Calculate the minimum projected coordinates for the cutout,
    # ensuring its geometric center aligns with the most massive pixel's center.
    min_x_cutout = most_massive_pixel_x_coord_global - (num_pixels_x_cutout * pixel_size / 2.0)
    min_y_cutout = most_massive_pixel_y_coord_global - (num_pixels_y_cutout * pixel_size / 2.0)

    # Initialize output arrays for the cutout region (STARS)
    star_particle_membership = [[[] for _ in range(num_pixels_x_cutout)] for _ in range(num_pixels_y_cutout)]
    star_mass_density_map = np.zeros((num_pixels_y_cutout, num_pixels_x_cutout), dtype=float)

    # Assign STAR particles to the cutout grid
    for i in range(len(projected_star_2d_coords)):
        x_coord_proj = projected_star_2d_coords[i, 0]
        y_coord_proj = projected_star_2d_coords[i, 1]

        x_idx_cutout = int(np.floor((x_coord_proj - min_x_cutout) / pixel_size))
        y_idx_cutout = int(np.floor((y_coord_proj - min_y_cutout) / pixel_size))

        # Only include particles that fall within the cutout dimensions
        if 0 <= x_idx_cutout < num_pixels_x_cutout and 0 <= y_idx_cutout < num_pixels_y_cutout:
            star_particle_membership[y_idx_cutout][x_idx_cutout].append((i, star_line_of_sight_distances_normalized[i])) # Use normalized distance
            star_mass_density_map[y_idx_cutout][x_idx_cutout] += particle_masses[i]

    # Set central pixel coordinate to the geometric center of the cutout
    central_pixel_x = num_pixels_x_cutout // 2
    central_pixel_y = num_pixels_y_cutout // 2
    central_pixel_coords = (central_pixel_x, central_pixel_y)

    # Process GAS particles if provided
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

    # Prepare grid information for output
    grid_info = {
        'min_x_proj': min_x_cutout,
        'min_y_proj': min_y_cutout,
        'num_pixels_x': num_pixels_x_cutout,
        'num_pixels_y': num_pixels_y_cutout,
        'effective_pixel_size_x': pixel_size,
        'effective_pixel_size_y': pixel_size
    }

    return star_particle_membership, star_mass_density_map, central_pixel_coords, grid_info, gas_particle_membership, gas_mass_density_map, stars_vel_los_proj, gas_vel_los_proj

def doppler_shift_spectrum(wave_rest, flux_rest, vel_los_km_s):
    """
    Applies a Doppler shift to a spectrum.

    Parameters
    ----------
    wave_rest : np.ndarray
        Rest-frame wavelength array in Angstroms.
    flux_rest : np.ndarray
        Rest-frame flux array.
    vel_los_km_s : float
        Line-of-sight velocity in km/s. Positive for redshift (recession),
        negative for blueshift (approach).

    Returns
    -------
    tuple:
        - wave_shifted (np.ndarray): Doppler-shifted wavelength array.
        - flux_shifted (np.ndarray): Corresponding flux array (intensity is conserved).
    """
    c = 299792.458 # Speed of light in km/s

    # Calculate the Doppler factor (1 + z_doppler)
    doppler_factor = 1 + (vel_los_km_s / c)

    wave_shifted = wave_rest * doppler_factor
    flux_shifted = flux_rest # Intensity is conserved when just shifting wavelength.

    return wave_shifted, flux_shifted

def create_hdf5_file(filename, stars_init_mass, stars_form_z, stars_mass, stars_zmet, stars_coords,
    stars_vel, gas_mass, gas_zmet, gas_sfr_inst, gas_temp, gas_coords, gas_vel, gas_mass_H):
    """
    Creates an HDF5 file with stellar and gas particle properties.

    Args:
        filename (str): The name of the HDF5 file to create.
        stars_init_mass (np.ndarray): 1D array of initial stellar masses.
        stars_form_z (np.ndarray): 1D array of stellar formation redshifts.
        stars_mass (np.ndarray): 1D array of current stellar masses.
        stars_zmet (np.ndarray): 1D array of stellar metallicities.
        stars_coords (np.ndarray): (N,3) array of stellar coordinates in units of kpc.
        stars_vel (np.ndarray): (N,3) array of stellar peculiar velocities in units of km/s.
        gas_mass (np.ndarray): 1D array of gas masses.
        gas_zmet (np.ndarray): 1D array of gas metallicities.
        gas_sfr_inst (np.ndarray): 1D array of instantaneous SFR for gas.
        gas_temp (np.ndarray): 1D array of gas temperatures.
        gas_coords (np.ndarray): (N,3) array of gas coordinates in units of kpc.
        gas_vel (np.ndarray): (N,3) array of gas peculiar velocities in units of km/s.
        gas_mass_H (np.ndarray): hyrogen mass in unit of solar mass.
    """

    import h5py

    # Open the HDF5 file in write mode ('w').
    # The 'with' statement ensures the file is automatically closed.
    with h5py.File(filename, 'w') as f:

        # Create the 'star' group
        star_group = f.create_group('star')
        star_group.create_dataset('init_mass', data=stars_init_mass, compression="gzip")
        star_group.create_dataset('form_z', data=stars_form_z, compression="gzip")
        star_group.create_dataset('mass', data=stars_mass, compression="gzip")
        star_group.create_dataset('zmet', data=stars_zmet, compression="gzip")
        star_group.create_dataset('coords', data=stars_coords, compression="gzip")
        star_group.create_dataset('vel', data=stars_vel, compression="gzip")
        
        # Create the 'gas' group
        gas_group = f.create_group('gas')
        gas_group.create_dataset('mass', data=gas_mass, compression="gzip")
        gas_group.create_dataset('zmet', data=gas_zmet, compression="gzip")
        gas_group.create_dataset('sfr_inst', data=gas_sfr_inst, compression="gzip")
        gas_group.create_dataset('temp', data=gas_temp, compression="gzip")
        gas_group.create_dataset('coords', data=gas_coords, compression="gzip")
        gas_group.create_dataset('vel', data=gas_vel, compression="gzip")
        gas_group.create_dataset('mass_H', data=gas_mass_H, compression="gzip")

        print(f"HDF5 file '{filename}' created successfully.")
    
    