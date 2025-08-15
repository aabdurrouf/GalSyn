import os
import sys
import numpy as np
import h5py
from astropy.io import fits
from joblib import Parallel, delayed
from tqdm.auto import tqdm
from tqdm_joblib import tqdm_joblib
import multiprocessing

# Import necessary functions from galsyn's existing modules
from . import config
from .utils import define_cosmo, interp_age_univ_from_z, get_2d_density_projection_no_los_binning, construct_SFH
from .imgutils import angular_to_physical, determine_image_size

# Global variables for worker processes (to be initialized once per worker)
_worker_stars_form_lbt = None
_worker_stars_init_mass = None
_worker_stars_metallicity = None
_worker_sfh_del_t = None
_worker_sfh_max_lbt = None
_worker_sfh_lbt_bins = None # To store the common lookback time bins

class SFHReconstructor:
    """
    A class to reconstruct spatially resolved star formation histories (SFHs)
    from HDF5 simulation data. It projects star particles onto a 2D grid and
    calculates SFH-related quantities for each pixel, outputting the results
    as a 3D FITS file (spatial x spatial x lookback_time).

    Parameters:
        sim_file (str): Path to the HDF5 simulation file.
        z (float): Redshift of the galaxy. Defaults to 0.01.
        Z_sun (float, optional): The value of solar metallicity used for normalizing particle metallicities. Defaults to 0.019.

    """

    def __init__(self, sim_file, z, Z_sun=0.019):
        """
        Initializes the SFHReconstructor with basic simulation parameters.

        Parameters:
            sim_file (str): Path to the HDF5 simulation file.
            z (float): Redshift of the galaxy. Defaults to 0.01.
            Z_sun (float, optional): The value of solar metallicity used for
                normalizing particle metallicities. Defaults to 0.019.
        """
        self._sim_file = sim_file
        self._z = z
        self._Z_sun = Z_sun

        # Initialize with default values from config.py
        self._load_config_defaults()

        # Set other default parameters
        self._dim_kpc = None
        self._pix_arcsec = 0.02
        self._polar_angle_deg = 0
        self._azimuth_angle_deg = 0
        self._ncpu = 4
        self._initdim_kpc = 100
        self._initdim_mass_fraction = 0.99
        self._name_out_sfh = None
        self._sfh_del_t = 0.05 # Default SFH bin width in Gyr
        self._sfh_max_lbt = 14.0 # Default maximum lookback time for SFH in Gyr

    def _load_config_defaults(self):
        """
        Loads default parameter values from the config.py module.
        """
        self._cosmo_str = getattr(config, 'COSMO', "Planck18")

    @property
    def sim_file(self):
        """str: The path to the input HDF5 simulation file."""
        return self._sim_file

    @sim_file.setter
    def sim_file(self, value):
        if not isinstance(value, str):
            raise ValueError("sim_file must be a string.")
        if not os.path.exists(value):
            raise FileNotFoundError(f"Simulation file not found at: {value}")
        self._sim_file = value

    @property
    def z(self):
        """float: The redshift of the galaxy snapshot."""
        return self._z

    @z.setter
    def z(self, value):
        if not isinstance(value, (int, float)) or value < 0:
            raise ValueError("z must be a non-negative number.")
        self._z = value

    @property
    def Z_sun(self):
        """float: The value of solar metallicity for normalization."""
        return self._Z_sun

    @Z_sun.setter
    def Z_sun(self, value):
        if not isinstance(value, (int, float)) or value < 0:
            raise ValueError("Z_sun must be a non-negative number.")
        self._Z_sun = value

    @property
    def dim_kpc(self):
        """float or None: The physical side length of the output image in kpc.
        If set to None, it will be automatically determined during reconstruction.
        """
        return self._dim_kpc

    @dim_kpc.setter
    def dim_kpc(self, value):
        if value is not None:
            if not isinstance(value, (int, float)) or value <= 0:
                raise ValueError("dim_kpc must be a positive number or None.")
        self._dim_kpc = value

    @property
    def pix_arcsec(self):
        """float: The side length of each square pixel in arcseconds."""
        return self._pix_arcsec

    @pix_arcsec.setter
    def pix_arcsec(self, value):
        if not isinstance(value, (int, float)) or value <= 0:
            raise ValueError("pix_arcsec must be a positive number.")
        self._pix_arcsec = value

    @property
    def polar_angle_deg(self):
        """float: The polar (inclination) angle for 2D projection in degrees."""
        return self._polar_angle_deg

    @polar_angle_deg.setter
    def polar_angle_deg(self, value):
        if not isinstance(value, (int, float)):
            raise ValueError("polar_angle_deg must be a number.")
        self._polar_angle_deg = value

    @property
    def azimuth_angle_deg(self):
        """float: The azimuthal (rotation) angle for 2D projection in degrees."""
        return self._azimuth_angle_deg

    @azimuth_angle_deg.setter
    def azimuth_angle_deg(self, value):
        if not isinstance(value, (int, float)):
            raise ValueError("azimuth_angle_deg must be a number.")
        self._azimuth_angle_deg = value

    @property
    def ncpu(self):
        """int: The number of CPU cores to use for parallel processing."""
        return self._ncpu

    @ncpu.setter
    def ncpu(self, value):
        if not isinstance(value, int) or value <= 0:
            raise ValueError("ncpu must be a positive integer.")
        self._ncpu = value

    @property
    def initdim_kpc(self):
        """float: The initial search dimension (in kpc) for automatic image sizing."""
        return self._initdim_kpc

    @initdim_kpc.setter
    def initdim_kpc(self, value):
        if not isinstance(value, (int, float)) or value <= 0:
            raise ValueError("initdim_kpc must be a positive number.")
        self._initdim_kpc = value

    @property
    def initdim_mass_fraction(self):
        """float: The stellar mass fraction to enclose for automatic image sizing."""
        return self._initdim_mass_fraction

    @initdim_mass_fraction.setter
    def initdim_mass_fraction(self, value):
        if not isinstance(value, (int, float)) or not (0 < value <= 1):
            raise ValueError("initdim_mass_fraction must be a number between 0 and 1.")
        self._initdim_mass_fraction = value

    @property
    def name_out_sfh(self):
        """str or None: The path and filename for the output FITS file."""
        return self._name_out_sfh

    @name_out_sfh.setter
    def name_out_sfh(self, value):
        if value is not None and not isinstance(value, str):
            raise ValueError("name_out_sfh must be a string or None.")
        self._name_out_sfh = value

    @property
    def sfh_del_t(self):
        """float: The width of each lookback time bin in Gyr for the SFH."""
        return self._sfh_del_t

    @sfh_del_t.setter
    def sfh_del_t(self, value):
        if not isinstance(value, (int, float)) or value <= 0:
            raise ValueError("sfh_del_t must be a positive number.")
        self._sfh_del_t = value

    @property
    def sfh_max_lbt(self):
        """float: The maximum lookback time in Gyr for the SFH."""
        return self._sfh_max_lbt

    @sfh_max_lbt.setter
    def sfh_max_lbt(self, value):
        if not isinstance(value, (int, float)) or value <= 0:
            raise ValueError("sfh_max_lbt must be a positive number.")
        self._sfh_max_lbt = value

    @property
    def cosmo_str(self):
        """str: The name of the cosmological model to use (e.g., 'Planck18')."""
        return self._cosmo_str

    @cosmo_str.setter
    def cosmo_str(self, value):
        accepted_cosmo_models = ["planck18", "planck15", "planck13", "wmap5", "wmap7", "wmap9"]
        if not isinstance(value, str) or value.lower() not in accepted_cosmo_models:
            raise ValueError(f"cosmo_str must be one of {accepted_cosmo_models}.")
        self._cosmo_str = value.lower()

    def set_params(self, **kwargs):
        """
        Sets multiple parameters at once using keyword arguments.

        This provides a convenient way to update the reconstructor's configuration.
        Any invalid parameter names will be ignored with a warning.

        Args:
            **kwargs: Keyword arguments where keys match attribute names
                      (e.g., dim_kpc=50, sfh_del_t=0.1).

        Returns:
            None
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                try:
                    setattr(self, key, value)
                except ValueError as e:
                    print(f"Error setting '{key}': {e}")
                except FileNotFoundError as e:
                    print(f"Error setting '{key}': {e}")
            else:
                print(f"Warning: Parameter '{key}' not recognized.")

    def _init_worker(self, stars_form_lbt_global, stars_init_mass_global, stars_metallicity_global,
                     sfh_del_t_val, sfh_max_lbt_val, sfh_lbt_bins_val):
        """
        Initializes global variables for each worker process in parallel execution.
        """
        global _worker_stars_form_lbt, _worker_stars_init_mass, _worker_stars_metallicity
        global _worker_sfh_del_t, _worker_sfh_max_lbt, _worker_sfh_lbt_bins

        _worker_stars_form_lbt = stars_form_lbt_global
        _worker_stars_init_mass = stars_init_mass_global
        _worker_stars_metallicity = stars_metallicity_global
        _worker_sfh_del_t = sfh_del_t_val
        _worker_sfh_max_lbt = sfh_max_lbt_val
        _worker_sfh_lbt_bins = sfh_lbt_bins_val

    def _process_pixel_sfh(self, ii, jj, star_particle_membership_list):
        """
        Worker function to process SFH calculations for a single pixel (ii, jj).
        This function will be executed in parallel.

        Args:
            ii (int): Y-coordinate (row index) of the pixel.
            jj (int): X-coordinate (column index) of the pixel.
            star_particle_membership_list (list): List of (original_particle_index, line_of_sight_distance)
                                                  for star particles in THIS pixel.

        Returns:
            tuple: (ii, jj, sfh_dict) where sfh_dict is the result from construct_SFH,
                   now also including mass assembly times.
        """
        # Extract original indices of star particles in this pixel
        star_ids = np.asarray([x[0] for x in star_particle_membership_list], dtype=int)

        # Filter global arrays to get data for stars in this pixel
        pixel_stars_form_lbt = _worker_stars_form_lbt[star_ids]
        pixel_stars_init_mass = _worker_stars_init_mass[star_ids]
        pixel_stars_metallicity = _worker_stars_metallicity[star_ids]

        # Construct SFH for this pixel
        sfh_dict = construct_SFH(
            pixel_stars_form_lbt,
            pixel_stars_init_mass,
            pixel_stars_metallicity,
            del_t=_worker_sfh_del_t,
            max_lbt=_worker_sfh_max_lbt
        )

        # Calculate mass assembly times
        mass_assembly_times = {}
        percentages = [0.05, 0.10, 0.25, 0.50, 0.75, 0.95]

        # Use the cumulative mass returned by construct_SFH (which now includes all bins)
        # and the full lbt midpoints (which are also consistent across pixels)
        if sfh_dict['mass'].sum() > 0: # Check if there is any mass in this pixel
            pixel_total_initial_mass = sfh_dict['mass'].sum()
            cumulative_mass_formed_for_interp = sfh_dict['cumul_mass'] # This is already cumulative mass formed at end of each bin
            
            x_interp = sfh_dict['lbt'] # Lookback time midpoints for all bins
            y_interp = cumulative_mass_formed_for_interp # Cumulative mass formed at end of each bin

            for p in percentages:
                target_mass = p * pixel_total_initial_mass
                
                # np.interp requires x-coordinates (y_interp here) to be increasing.
                # cumulative_mass_formed should always be non-decreasing.
                # If there are zero-mass bins, y_interp will have plateaus.
                # np.interp handles this correctly.
                t_assemble = np.interp(target_mass, y_interp, x_interp)
                mass_assembly_times[f't_{int(p*100)}'] = t_assemble
        else:
            # If no stars in pixel, all assembly times are NaN
            mass_assembly_times = {f't_{int(p*100)}': np.nan for p in percentages}

        sfh_dict['mass_assembly_times'] = mass_assembly_times
        return ii, jj, sfh_dict

    def reconstruct_sfh(self):
        """
        Executes the spatially resolved SFH reconstruction process.
        """
        if self.sim_file is None:
            raise ValueError("sim_file must be set before calling reconstruct_sfh.")
        if self.name_out_sfh is None:
            raise ValueError("name_out_sfh must be set before calling reconstruct_sfh.")

        cosmo = define_cosmo(self.cosmo_str)
        print(f"Processing {self.sim_file} at redshift z={self.z}")

        # --- Data Loading and Initial Calculations (Sequential) ---
        try:
            f = h5py.File(self.sim_file, 'r')

            # Stellar particle data
            stars_init_mass_raw = f['star']['init_mass'][:]
            stars_form_z = f['star']['form_z'][:]
            stars_zmet = f['star']['zmet'][:]
            stars_zsol_raw = stars_zmet / self.Z_sun
            stars_coords_raw = f['star']['coords'][:]   # coordinates (N,3) in units of kpc

            f.close()
        except Exception as e:
            print(f"Error loading data from simulation file {self.sim_file}: {e}")
            sys.exit(1)

        # Convert formation time to lookback time
        snap_a = 1.0/(1.0 + self.z)
        stars_form_a = 1.0/(1.0 + stars_form_z)
        snap_univ_age = cosmo.age(self.z).value
        stars_form_age_univ = interp_age_univ_from_z(stars_form_z, cosmo)
        stars_form_lbt_raw = snap_univ_age - stars_form_age_univ

        # Filter out invalid stellar particles (e.g., formation time 0 or negative age)
        idx_valid_stars = np.where((stars_form_a > 0) & (stars_form_lbt_raw >= 0))[0]

        stars_init_mass = stars_init_mass_raw[idx_valid_stars]
        stars_zsol = stars_zsol_raw[idx_valid_stars]
        stars_form_lbt = stars_form_lbt_raw[idx_valid_stars] # This is the lookback time for SFH
        stars_coords = stars_coords_raw[idx_valid_stars,:]
        
        pix_kpc = angular_to_physical(self.z, self.pix_arcsec, cosmo)
        print(f"Pixel size: {self.pix_arcsec:.2f} arcsec or {pix_kpc:.2f} kpc")

        # Determine image dimension if not explicitly set
        if self.dim_kpc is None:
            self.dim_kpc = determine_image_size(stars_coords, stars_init_mass, pix_kpc, 
                                                (self.initdim_kpc, self.initdim_kpc),
                                                self.polar_angle_deg, self.azimuth_angle_deg,
                                                gas_coords=None, gas_masses=None, # SFH reconstruction only needs stars
                                                mass_percentage=self.initdim_mass_fraction,
                                                max_img_dim=self.initdim_kpc)
            print(f"Automatically determined image dimension: {self.dim_kpc:.2f} kpc")

        output_dimension_physical = (self.dim_kpc, self.dim_kpc)

        # Get 2D projection and pixel membership for star particles
        star_particle_membership, _, _, grid_info, _, _, _, _ = \
            get_2d_density_projection_no_los_binning(
                stars_coords,
                stars_init_mass,
                pix_kpc,
                output_dimension_physical,
                polar_angle_deg=self.polar_angle_deg,
                azimuth_angle_deg=self.azimuth_angle_deg,
                gas_coords=None,
                gas_masses=None
            )
        
        dimx, dimy = grid_info['num_pixels_x'], grid_info['num_pixels_y']
        print(f"Cutout size: {dimx} x {dimy} pix or {self.dim_kpc:.2f} x {self.dim_kpc:.2f} kpc")

        # Define common lookback time bins for all pixels
        # Ensure max_lbt does not exceed the age of the universe at the galaxy's redshift
        age_universe_at_galaxy_z = cosmo.age(self.z).value # Age of the universe at the galaxy's redshift
        self.sfh_max_lbt = min(self.sfh_max_lbt, age_universe_at_galaxy_z)

        sfh_lbt_bins = np.arange(0, self.sfh_max_lbt + self.sfh_del_t, self.sfh_del_t)
        # Midpoints of the bins for FITS header
        sfh_lbt_midpoints = sfh_lbt_bins[:-1] + 0.5 * self.sfh_del_t
        num_lbt_bins = len(sfh_lbt_midpoints)

        # Initialize 3D arrays to store SFH quantities
        # (dimy, dimx, num_lbt_bins)
        map_sfr = np.zeros((dimy, dimx, num_lbt_bins), dtype=np.float32)
        map_nstars = np.zeros((dimy, dimx, num_lbt_bins), dtype=np.int32)
        map_mass = np.zeros((dimy, dimx, num_lbt_bins), dtype=np.float32)
        map_cumul_mass = np.zeros((dimy, dimx, num_lbt_bins), dtype=np.float32)
        map_metallicity = np.zeros((dimy, dimx, num_lbt_bins), dtype=np.float32)

        # Initialize 2D arrays for mass assembly times
        map_t_05 = np.full((dimy, dimx), np.nan, dtype=np.float32)
        map_t_10 = np.full((dimy, dimx), np.nan, dtype=np.float32)
        map_t_25 = np.full((dimy, dimx), np.nan, dtype=np.float32)
        map_t_50 = np.full((dimy, dimx), np.nan, dtype=np.float32)
        map_t_75 = np.full((dimy, dimx), np.nan, dtype=np.float32)
        map_t_95 = np.full((dimy, dimx), np.nan, dtype=np.float32)

        # Prepare tasks for parallel processing
        tasks = []
        for ii in range(dimy):
            for jj in range(dimx):
                num_stars_in_pixel = len(star_particle_membership[ii][jj])
                # SFH reconstruction only depends on stars, so complexity is just num_stars_in_pixel
                complexity = num_stars_in_pixel 
                tasks.append({'coords': (ii, jj), 'complexity': complexity, 
                              'star_part_mem': star_particle_membership[ii][jj]})

        # Sort tasks from heaviest (largest number of stars) to lightest
        tasks.sort(key=lambda x: x['complexity'], reverse=True)

        processed_tasks_args = []
        for task in tasks:
            ii, jj = task['coords']
            processed_tasks_args.append((ii, jj, task['star_part_mem']))

        num_cores = self.ncpu
        if num_cores == -1:
            num_cores = multiprocessing.cpu_count()

        print(f"\nStarting parallel SFH reconstruction on {num_cores} cores...")

        with tqdm_joblib(total=len(tasks), desc="Processing pixels for SFH") as progress_bar:
            results = Parallel(n_jobs=num_cores, verbose=0, initializer=self._init_worker,
                               initargs=(stars_form_lbt, stars_init_mass, stars_zsol,
                                         self.sfh_del_t, self.sfh_max_lbt, sfh_lbt_bins))(
                delayed(self._process_pixel_sfh)(*task_args) for task_args in processed_tasks_args
            )
        print("\nFinished parallel SFH reconstruction.")

        # Populate the 3D maps with results from parallel processing
        for pixel_result_tuple in results:
            ii, jj, sfh_data = pixel_result_tuple

            # Ensure data is correctly assigned based on the common sfh_lbt_midpoints order
            map_sfr[ii, jj, :] = sfh_data['sfr']
            map_nstars[ii, jj, :] = sfh_data['nstars']
            map_mass[ii, jj, :] = sfh_data['mass']
            map_cumul_mass[ii, jj, :] = sfh_data['cumul_mass']
            map_metallicity[ii, jj, :] = sfh_data['metallicity']

            # Populate new mass assembly time maps
            if 'mass_assembly_times' in sfh_data:
                map_t_05[ii, jj] = sfh_data['mass_assembly_times'].get('t_5', np.nan)
                map_t_10[ii, jj] = sfh_data['mass_assembly_times'].get('t_10', np.nan)
                map_t_25[ii, jj] = sfh_data['mass_assembly_times'].get('t_25', np.nan)
                map_t_50[ii, jj] = sfh_data['mass_assembly_times'].get('t_50', np.nan)
                map_t_75[ii, jj] = sfh_data['mass_assembly_times'].get('t_75', np.nan)
                map_t_95[ii, jj] = sfh_data['mass_assembly_times'].get('t_95', np.nan)


        print("All SFH calculations complete. Maps populated.")

        # --- Save results to FITS file ---
        try:
            hdul = fits.HDUList()

            # Primary HDU (can be empty or contain general info)
            prihdr = fits.Header()
            prihdr['COMMENT'] = 'Spatially Resolved Star Formation History'
            prihdr['REDSHIFT'] = self.z
            prihdr['POLAR'] = self.polar_angle_deg
            prihdr['AZIMUTH'] = self.azimuth_angle_deg
            prihdr['DIM_KPC'] = self.dim_kpc
            prihdr['PIX_KPC'] = pix_kpc
            prihdr['PIXSIZE'] = self.pix_arcsec
            prihdr['SFH_DELT'] = self.sfh_del_t
            # Removed SFH_MAXLBT from header as it's now dynamically adjusted
            prihdr['COSMO'] = self.cosmo_str
            prihdr['H0'] = cosmo.H0.value
            prihdr['OM0'] = cosmo.Om0
            prihdr['OL0'] = cosmo.Ode0
            hdul.append(fits.PrimaryHDU(header=prihdr))

            # Helper to add ImageHDU with WCS for 3D data
            def add_sfh_hdu(data, name, comment, unit):
                ext_hdr = fits.Header()
                ext_hdr['EXTNAME'] = name
                ext_hdr['COMMENT'] = comment
                ext_hdr['BUNIT'] = unit
                
                # Spatial WCS
                ext_hdr['CRPIX1'] = dimx / 2.0 + 0.5
                ext_hdr['CRPIX2'] = dimy / 2.0 + 0.5
                ext_hdr['CDELT1'] = pix_kpc
                ext_hdr['CDELT2'] = pix_kpc
                ext_hdr['CUNIT1'] = 'kpc'
                ext_hdr['CUNIT2'] = 'kpc'

                # Lookback time WCS (3rd dimension)
                ext_hdr['CRPIX3'] = 1.0 # Reference pixel for 3rd axis
                ext_hdr['CDELT3'] = self.sfh_del_t # Step size in Gyr
                ext_hdr['CRVAL3'] = sfh_lbt_midpoints[0] if num_lbt_bins > 0 else 0.0 # Value at reference pixel
                ext_hdr['CUNIT3'] = 'Gyr'
                ext_hdr['CTYPE3'] = 'LOOKBACK_TIME' # Custom type for clarity

                hdul.append(fits.ImageHDU(data=data, header=ext_hdr))

            # Add each SFH quantity as a separate extension
            add_sfh_hdu(map_sfr, 'SFR', 'Star Formation Rate', 'Msun/yr')
            add_sfh_hdu(map_nstars, 'N_STARS', 'Number of stars in SFH bin', 'count')
            add_sfh_hdu(map_mass, 'MASS', 'Initial stellar mass formed', 'Msun')
            add_sfh_hdu(map_cumul_mass, 'CUMUL_MASS', 'Cumulative initial stellar mass formed', 'Msun')
            add_sfh_hdu(map_metallicity, 'METALLICITY', 'Mass-weighted average metallicity', 'Z/Zsun')

            # Add 2D maps for mass assembly times
            # Helper to add 2D ImageHDU
            def add_2d_map_hdu(data, name, comment, unit):
                ext_hdr = fits.Header()
                ext_hdr['EXTNAME'] = name
                ext_hdr['COMMENT'] = comment
                ext_hdr['BUNIT'] = unit
                ext_hdr['CRPIX1'] = dimx / 2.0 + 0.5
                ext_hdr['CRPIX2'] = dimy / 2.0 + 0.5
                ext_hdr['CDELT1'] = pix_kpc
                ext_hdr['CDELT2'] = pix_kpc
                ext_hdr['CUNIT1'] = 'kpc'
                ext_hdr['CUNIT2'] = 'kpc'
                hdul.append(fits.ImageHDU(data=data, header=ext_hdr))

            add_2d_map_hdu(map_t_05, 'T_5_PERCENT', 'Lookback time for 5% mass assembly', 'Gyr')
            add_2d_map_hdu(map_t_10, 'T_10_PERCENT', 'Lookback time for 10% mass assembly', 'Gyr')
            add_2d_map_hdu(map_t_25, 'T_25_PERCENT', 'Lookback time for 25% mass assembly', 'Gyr')
            add_2d_map_hdu(map_t_50, 'T_50_PERCENT', 'Lookback time for 50% mass assembly', 'Gyr')
            add_2d_map_hdu(map_t_75, 'T_75_PERCENT', 'Lookback time for 75% mass assembly', 'Gyr')
            add_2d_map_hdu(map_t_95, 'T_95_PERCENT', 'Lookback time for 95% mass assembly', 'Gyr')

            # Add Lookback Time bins as a 1D extension
            ext_hdr_lbt_bins = fits.Header()
            ext_hdr_lbt_bins['EXTNAME'] = 'LOOKBACK_TIME_BINS'
            ext_hdr_lbt_bins['COMMENT'] = 'Midpoints of lookback time bins for SFH'
            ext_hdr_lbt_bins['BUNIT'] = 'Gyr'
            ext_hdr_lbt_bins['CRPIX1'] = 1.0
            ext_hdr_lbt_bins['CDELT1'] = self.sfh_del_t
            ext_hdr_lbt_bins['CRVAL1'] = sfh_lbt_midpoints[0] if num_lbt_bins > 0 else 0.0
            ext_hdr_lbt_bins['CUNIT1'] = 'Gyr'
            hdul.append(fits.ImageHDU(data=sfh_lbt_midpoints, header=ext_hdr_lbt_bins))

            output_dir = os.path.dirname(self.name_out_sfh)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)

            hdul.writeto(self.name_out_sfh, overwrite=True, output_verify='fix')
            print(f"Spatially resolved SFH reconstruction completed successfully and results saved to FITS file: {self.name_out_sfh}")

        except Exception as e:
            print(f"Error saving FITS file {self.name_out_sfh}: {e}")
            import traceback
            traceback.print_exc()