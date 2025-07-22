import os, sys
import numpy as np
from . import config
from .galsyn_run import generate_images
from .ssp_generator import generate_ssp_grid, FSPS_Z_SUN # Import the new function and constant


class GalaxySynthesizer:

    def __init__(self, sim_file=None, z=0.01, filters=[], filter_transmission={}, filter_wave_eff={}):
        self._sim_file = sim_file
        self._z = z
        self._filters = filters
        self._filter_transmission = filter_transmission
        self._filter_wave_eff = filter_wave_eff

        # Initialize with default values from config.py
        self._load_config_defaults()

        # Set other default parameters
        self._dim_kpc = None
        self._pix_arcsec = 0.02
        self._flux_unit = 'MJy/sr'
        self._polar_angle_deg = 0
        self._azimuth_angle_deg = 0
        self._ncpu = 4 # Default to 4, can be overridden by config or set_params

        self._initdim_kpc = 100
        self._initdim_mass_fraction = 0.92

        self._name_out_img = None
        self._ssp_filepath = "ssp_spectra.hdf5" # Default SSP file path
        self._use_precomputed_ssp = True # New flag, default to True

        # New parameters for pixel spectra output
        self._output_pixel_spectra = False # Default to not output spectra
        self._rest_wave_min = 1000.0 # Default min wavelength in Angstrom
        self._rest_wave_max = 16000.0 # Default max wavelength in Angstrom


    def _load_config_defaults(self):
        """
        Loads default parameter values from the config.py module.
        """
        # Hydrogen fraction
        self._XH = getattr(config, 'XH', 0.76)

        # Cosmology
        self._cosmo_str = getattr(config, 'COSMO', "Planck18")
        self._cosmo_h = getattr(config, 'COSMO_LITTLE_H', 0.6774)

        # SPS setup
        self._imf_type = getattr(config, 'IMF_TYPE', 1)
        self._add_neb_emission = getattr(config, 'ADD_NEB_EMISSION', 1)
        self._gas_logu = getattr(config, 'GAS_LOGU', -2.0)

        # IGM absorption
        self._add_igm_absorption = getattr(config, 'ADD_IGM_ABSORPTION', 1)
        self._igm_type = getattr(config, 'IGM_TYPE', 0)

        # Dust attenuation tau normalization as function of redshift
        self._norm_dust_z = getattr(config, 'NORM_DUST_Z', [0, 2, 3, 4, 5, 6, 7, 8, 12])
        self._norm_dust_tau = getattr(config, 'NORM_DUST_TAU', [0.46, 0.46, 0.20, 0.13, 0.08, 0.06, 0.04, 0.03, 0.03])

        # Dust attenuation setup
        self._dust_law = getattr(config, 'DUST_LAW', 0)
        self._dust_index = getattr(config, 'DUST_INDEX', 0.0)
        self._bump_amp = getattr(config, 'BUMP_AMP', 0.85)
        self._dust_index_bc = getattr(config, 'DUST_INDEX_BC', -0.7)
        self._t_esc = getattr(config, 'T_ESC', 0.01)

        self._dustindexAV_AV = getattr(config, 'DUSTINDEXAV_AV', [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0])
        self._dustindexAV_dust_index = getattr(config, 'DUSTINDEXAV_DUST_INDEX', [-1.005, -0.548, -0.264, -0.135, -0.067, 0.048, 0.166, 0.217, 0.239, 0.261, 0.283, 0.306, 
                                                                                  0.328, 0.35, 0.373, 0.395, 0.417, 0.44, 0.462, 0.484, 0.507])

        self._salim_a0 = getattr(config, 'SALIM_A0', -4.30)
        self._salim_a1 = getattr(config, 'SALIM_A1', 2.71)
        self._salim_a2 = getattr(config, 'SALIM_A2', -0.191)
        self._salim_a3 = getattr(config, 'SALIM_A3', 0.0121)
        self._salim_RV = getattr(config, 'SALIM_RV', 3.15)
        self._salim_B = getattr(config, 'SALIM_B', 1.57)
        

    @property
    def sim_file(self):
        return self._sim_file

    @sim_file.setter
    def sim_file(self, value):
        if not isinstance(value, str):
            raise ValueError("sim_file must be string.")
        self._sim_file = value

    @property
    def z(self):
        return self._z

    @z.setter
    def z(self, value):
        if not isinstance(value, (int, float)) or value < 0:
            raise ValueError("z must be a non-negative number.")
        self._z = value

    @property
    def filters(self):
        return self._filters

    @filters.setter
    def filters(self, value):
        if not isinstance(value, list):
            raise ValueError("filters must be a list.")
        for i, item in enumerate(value):
            if not isinstance(item, str):
                raise ValueError(f"filters list must contain only strings. Item at index {i} is not a string ({type(item).__name__}).")
        self._filters = value[:]

    @property
    def filter_transmission(self):
        return self._filter_transmission

    @filter_transmission.setter
    def filter_transmission(self, value):
        if not isinstance(value, dict):
            raise ValueError("filter_transmission must be a dictionary.")
        self._filter_transmission = value

    @property
    def filter_wave_eff(self):
        return self._filter_wave_eff

    @filter_wave_eff.setter
    def filter_wave_eff(self, value):
        if not isinstance(value, dict):
            raise ValueError("filter_wave_eff must be a dictionary.")
        self._filter_wave_eff = value

    @property
    def dim_kpc(self):
        return self._dim_kpc

    @dim_kpc.setter
    def dim_kpc(self, value):
        if value is not None:
            if not isinstance(value, (int, float)) or value <= 0:
                raise ValueError("dim_kpc must be a positive number or None.")
        self._dim_kpc = value

    @property
    def pix_arcsec(self):
        return self._pix_arcsec

    @pix_arcsec.setter
    def pix_arcsec(self, value):
        if not isinstance(value, (int, float)) or value <= 0:
            raise ValueError("pix_arcsec must be a positive number.")
        self._pix_arcsec = value

    @property
    def flux_unit(self):
        return self._flux_unit

    @flux_unit.setter
    def flux_unit(self, value):
        accepted_units = ['MJy/sr', 'nJy', 'AB magnitude', 'erg/s/cm2/A']
        if not isinstance(value, str) or value not in accepted_units:
            raise ValueError(f"flux_unit must be one of {accepted_units}.")
        self._flux_unit = value

    @property
    def polar_angle_deg(self):
        return self._polar_angle_deg

    @polar_angle_deg.setter
    def polar_angle_deg(self, value):
        if not isinstance(value, (int, float)):
            raise ValueError("polar_angle_deg must be a number.")
        self._polar_angle_deg = value

    @property
    def azimuth_angle_deg(self):
        return self._azimuth_angle_deg

    @azimuth_angle_deg.setter
    def azimuth_angle_deg(self, value):
        if not isinstance(value, (int, float)):
            raise ValueError("azimuth_angle_deg must be a number.")
        self._azimuth_angle_deg = value

    @property
    def ncpu(self):
        return self._ncpu

    @ncpu.setter
    def ncpu(self, value):
        if not isinstance(value, int) or value <= 0:
            raise ValueError("ncpu must be a positive integer.")
        self._ncpu = value

    @property
    def initdim_kpc(self):
        return self._initdim_kpc

    @initdim_kpc.setter
    def initdim_kpc(self, value):
        if not isinstance(value, (int, float)) or value <= 0:
            raise ValueError("initdim_kpc must be a positive number.")
        self._initdim_kpc = value

    @property
    def initdim_mass_fraction(self):
        return self._initdim_mass_fraction

    @initdim_mass_fraction.setter
    def initdim_mass_fraction(self, value):
        if not isinstance(value, (int, float)) or value <= 0:
            raise ValueError("initdim_mass_fraction must be a positive number.")
        self._initdim_mass_fraction = value

    @property
    def name_out_img(self):
        return self._name_out_img

    @name_out_img.setter
    def name_out_img(self, value):
        if value is not None and not isinstance(value, str):
            raise ValueError("name_out_img must be a string or None.")
        self._name_out_img = value

    @property
    def imf_type(self):
        return self._imf_type

    @imf_type.setter
    def imf_type(self, value):
        if not isinstance(value, int) or value not in [0, 1, 2]:
            raise ValueError("imf_type must be an integer (e.g., 0, 1, or 2).")
        self._imf_type = value

    @property
    def add_neb_emission(self):
        return self._add_neb_emission

    @add_neb_emission.setter
    def add_neb_emission(self, value):
        if not isinstance(value, (int, bool)):
            raise ValueError("add_neb_emission must be a boolean (True/False) or integer (0/1).")
        self._add_neb_emission = int(value)

    @property
    def gas_logu(self):
        return self._gas_logu

    @gas_logu.setter
    def gas_logu(self, value):
        if not isinstance(value, (int, float)):
            raise ValueError("gas_logu must be a number.")
        self._gas_logu = value

    @property
    def add_igm_absorption(self):
        return self._add_igm_absorption

    @add_igm_absorption.setter
    def add_igm_absorption(self, value):
        if not isinstance(value, (int, bool)):
            raise ValueError("add_igm_absorption must be a boolean (True/False) or integer (0/1).")
        self._add_igm_absorption = int(value)

    @property
    def igm_type(self):
        return self._igm_type

    @igm_type.setter
    def igm_type(self, value):
        if not isinstance(value, int):
            raise ValueError("igm_type must be an integer.")
        self._igm_type = value

    @property
    def dust_index_bc(self):
        return self._dust_index_bc

    @dust_index_bc.setter
    def dust_index_bc(self, value):
        if not isinstance(value, (int, float)):
            raise ValueError("dust_index_bc must be a number.")
        self._dust_index_bc = value

    @property
    def dust_index(self):
        return self._dust_index

    @dust_index.setter
    def dust_index(self, value):
        if not isinstance(value, (int, float)):
            raise ValueError("dust_index must be a number.")
        self._dust_index = value

    @property
    def t_esc(self):
        return self._t_esc

    @t_esc.setter
    def t_esc(self, value):
        if not isinstance(value, (int, float)) or value < 0:
            raise ValueError("t_esc must be a non-negative number.")
        self._t_esc = value

    @property
    def norm_dust_z(self):
        return self._norm_dust_z

    @norm_dust_z.setter
    def norm_dust_z(self, value):
        if not isinstance(value, list):
            raise ValueError("norm_dust_z must be a list.")
        for i, item in enumerate(value):
            if not isinstance(item, (int, float)):
                raise ValueError(f"norm_dust_z list must contain only numbers. Item at index {i} is not a number ({type(item).__name__}).")
        self._norm_dust_z = value[:]

    @property
    def norm_dust_tau(self):
        return self._norm_dust_tau

    @norm_dust_tau.setter
    def norm_dust_tau(self, value):
        if not isinstance(value, list):
            raise ValueError("norm_dust_tau must be a list.")
        for i, item in enumerate(value):
            if not isinstance(item, (int, float)):
                raise ValueError(f"norm_dust_tau list must contain only numbers. Item at index {i} is not a number ({type(item).__name__}).")
        self._norm_dust_tau = value[:]

    @property
    def cosmo_str(self):
        return self._cosmo_str

    @cosmo_str.setter
    def cosmo_str(self, value):
        accepted_cosmo_models = ["planck18", "planck15", "planck13", "wmap5", "wmap7", "wmap9"]
        if not isinstance(value, str) or value.lower() not in accepted_cosmo_models:
            raise ValueError(f"cosmo_str must be one of {accepted_cosmo_models}.")
        self._cosmo_str = value.lower() # Store in lowercase for consistency

    @property
    def cosmo_h(self):
        return self._cosmo_h

    @cosmo_h.setter
    def cosmo_h(self, value):
        if not isinstance(value, (int, float)) or value <= 0:
            raise ValueError("cosmo_h must be a positive number.")
        self._cosmo_h = value

    @property
    def XH(self):
        return self._XH

    @XH.setter
    def XH(self, value):
        if not isinstance(value, (int, float)) or not (0 <= value <= 1):
            raise ValueError("XH must be a number between 0 and 1.")
        self._XH = value

    @property
    def dust_law(self):
        return self._dust_law

    @dust_law.setter
    def dust_law(self, value):
        if not isinstance(value, int) or value not in [0, 1, 2, 3, 4, 5]:
            raise ValueError("dust_law must be integer in the range of 0 to 5.")
        self._dust_law = value

    @property
    def bump_amp(self):
        return self._bump_amp

    @bump_amp.setter
    def bump_amp(self, value):
        if not isinstance(value, (int, float)):
            raise ValueError("bump_amp must be a number.")
        self._bump_amp = value

    @property
    def dustindexAV_AV(self):
        return self._dustindexAV_AV

    @dustindexAV_AV.setter
    def dustindexAV_AV(self, value):
        if not isinstance(value, list):
            raise ValueError("dustindexAV_AV must be a list.")
        for i, item in enumerate(value):
            if not isinstance(item, (int, float)):
                raise ValueError(f"norm_dust_z list must contain only numbers. Item at index {i} is not a number ({type(item).__name__}).")
        self._dustindexAV_AV = value[:]

    @property
    def dustindexAV_dust_index(self):
        return self._dustindexAV_dust_index

    @dustindexAV_dust_index.setter
    def dustindexAV_dust_index(self, value):
        if not isinstance(value, list):
            raise ValueError("dustindexAV_dust_index must be a list.")
        for i, item in enumerate(value):
            if not isinstance(item, (int, float)):
                raise ValueError(f"dustindexAV_dust_index list must contain only numbers. Item at index {i} is not a number ({type(item).__name__}).")
        self._dustindexAV_dust_index = value[:]

    @property
    def salim_a0(self):
        return self._salim_a0

    @salim_a0.setter
    def salim_a0(self, value):
        if not isinstance(value, (int, float)):
            raise ValueError("salim_a0 must be a number.")
        self._salim_a0 = value

    @property
    def salim_a1(self):
        return self._salim_a1

    @salim_a1.setter
    def salim_a1(self, value):
        if not isinstance(value, (int, float)):
            raise ValueError("salim_a1 must be a number.")
        self._salim_a1 = value

    @property
    def salim_a2(self):
        return self._salim_a2

    @salim_a2.setter
    def salim_a2(self, value):
        if not isinstance(value, (int, float)):
            raise ValueError("salim_a2 must be a number.")
        self._salim_a2 = value

    @property
    def salim_a3(self):
        return self._salim_a3

    @salim_a3.setter
    def salim_a3(self, value):
        if not isinstance(value, (int, float)):
            raise ValueError("salim_a3 must be a number.")
        self._salim_a3 = value

    @property
    def salim_RV(self):
        return self._salim_RV

    @salim_RV.setter
    def salim_RV(self, value):
        if not isinstance(value, (int, float)):
            raise ValueError("salim_RV must be a number.")
        self._salim_RV = value

    @property
    def salim_B(self):
        return self._salim_B

    @salim_B.setter
    def salim_B(self, value):
        if not isinstance(value, (int, float)):
            raise ValueError("salim_B must be a number.")
        self._salim_B = value
    
    @property
    def ssp_filepath(self):
        return self._ssp_filepath

    @ssp_filepath.setter
    def ssp_filepath(self, value):
        if not isinstance(value, str):
            raise ValueError("ssp_filepath must be a string.")
        self._ssp_filepath = value

    @property
    def use_precomputed_ssp(self):
        return self._use_precomputed_ssp

    @use_precomputed_ssp.setter
    def use_precomputed_ssp(self, value):
        if not isinstance(value, bool):
            raise ValueError("use_precomputed_ssp must be a boolean (True/False).")
        self._use_precomputed_ssp = value

    @property
    def output_pixel_spectra(self):
        return self._output_pixel_spectra

    @output_pixel_spectra.setter
    def output_pixel_spectra(self, value):
        if not isinstance(value, bool):
            raise ValueError("output_pixel_spectra must be a boolean (True/False).")
        self._output_pixel_spectra = value

    @property
    def rest_wave_min(self):
        return self._rest_wave_min

    @rest_wave_min.setter
    def rest_wave_min(self, value):
        if not isinstance(value, (int, float)) or value <= 0:
            raise ValueError("rest_wave_min must be a positive number.")
        self._rest_wave_min = float(value)

    @property
    def rest_wave_max(self):
        return self._rest_wave_max

    @rest_wave_max.setter
    def rest_wave_max(self, value):
        if not isinstance(value, (int, float)) or value <= 0:
            raise ValueError("rest_wave_max must be a positive number.")
        self._rest_wave_max = float(value)
        if self._rest_wave_max <= self._rest_wave_min:
            raise ValueError("rest_wave_max must be greater than rest_wave_min.")


    # --- Convenience method for setting multiple parameters ---

    def set_params(self, **kwargs):
        """
        Sets multiple parameters at once using keyword arguments.
        Example: synthesizer.set_params(imf_type=1, filters=['FUV', 'NUV'])
        """
        for key, value in kwargs.items():
            # Check if the attribute exists as a property setter (e.g., _imf_type has imf_type.setter)
            if hasattr(self, key):
                try:
                    setattr(self, key, value)
                except ValueError as e:
                    print(f"Error setting '{key}': {e}")
            else:
                print(f"Warning: Parameter '{key}' not recognized.")

    def __repr__(self):
        params_list = []
        # Iterate over all public properties (those with getters)
        for key in dir(self):
            if not key.startswith('_') and not key.endswith('__') and hasattr(getattr(type(self), key), 'fget'):
                try:
                    val = getattr(self, key)
                    if isinstance(val, (list, dict)):
                        # Use repr for accurate representation of lists/dictionaries
                        params_list.append(f"{key}={repr(val)}")
                    elif isinstance(val, str):
                        # Add quotes for string values
                        params_list.append(f"{key}='{val}'")
                    else:
                        params_list.append(f"{key}={val}")
                except AttributeError:
                    # This can happen if a property raises an error during access (e.g., if a required init param is missing)
                    pass
        return f"GalaxySynthesizer({', '.join(params_list)})"

    def generate_ssp_data(self, overwrite=False):
        """
        Generates the SSP spectra grid and saves it to the specified ssp_filepath.
        Parameters like imf_type, add_neb_emission, gas_logu from the synthesizer
        instance are used for SSP generation to ensure consistency.
        This method is only called if use_precomputed_ssp is True.
        """
        if not self.use_precomputed_ssp:
            print("Skipping SSP grid generation as use_precomputed_ssp is False.")
            return

        print(f"Checking for pre-computed SSP spectra at: {self.ssp_filepath}")
        if not os.path.exists(self.ssp_filepath) or overwrite:
            print("Pre-computed SSP spectra not found or overwrite requested. Generating now...")
            self.ssp_filepath = generate_ssp_grid(
                output_filename=self.ssp_filepath,
                imf_type=self.imf_type,
                add_neb_emission=self.add_neb_emission,
                gas_logu=self.gas_logu,
                overwrite=overwrite
            )
        else:
            print("Pre-computed SSP spectra found. Skipping generation.")
        
        pass


    # --- Method to run the synthesis process ---
    def run_synthesis(self):
        """
        Executes the galaxy image synthesis process using the current parameters
        set in this GalaxySynthesizer instance.
        """
        # Ensure SSP data is generated/available if pre-computed option is chosen
        if self.use_precomputed_ssp:
            self.generate_ssp_data()

        try:
            # Call the generate_images function from galsyn.py
            # Pass all relevant parameters from the current instance
            generate_images(
                sim_file=self.sim_file,
                z=self.z,
                filters=self.filters,
                filter_transmission=self.filter_transmission,
                filter_wave_eff=self.filter_wave_eff,
                dim_kpc=self.dim_kpc,
                pix_arcsec=self.pix_arcsec,
                flux_unit=self.flux_unit,
                polar_angle_deg=self.polar_angle_deg,
                azimuth_angle_deg=self.azimuth_angle_deg,
                name_out_img=self.name_out_img,
                n_jobs=self.ncpu,
                imf_type=self.imf_type,
                add_neb_emission = self.add_neb_emission,
                gas_logu = self.gas_logu,
                add_igm_absorption = self.add_igm_absorption,
                igm_type = self.igm_type,
                dust_index_bc = self.dust_index_bc,
                dust_index = self.dust_index,
                t_esc = self.t_esc,
                norm_dust_z = self.norm_dust_z,
                norm_dust_tau = self.norm_dust_tau,
                cosmo_str = self.cosmo_str,
                cosmo_h = self.cosmo_h,
                XH = self.XH,
                dust_law = self.dust_law,
                bump_amp = self.bump_amp,
                dustindexAV_AV = self.dustindexAV_AV,
                dustindexAV_dust_index = self.dustindexAV_dust_index,
                salim_a0 = self.salim_a0,
                salim_a1 = self.salim_a1,
                salim_a2 = self.salim_a2,
                salim_a3 = self.salim_a3,
                salim_RV = self.salim_RV,
                salim_B = self.salim_B,
                initdim_kpc = self.initdim_kpc,
                initdim_mass_fraction = self.initdim_mass_fraction,
                use_precomputed_ssp = self.use_precomputed_ssp, # Pass the new flag
                ssp_filepath = self.ssp_filepath, # Pass the SSP file path
                output_pixel_spectra = self.output_pixel_spectra, # Pass new flag
                rest_wave_min = self.rest_wave_min, # Pass new param
                rest_wave_max = self.rest_wave_max # Pass new param
            )
            #print("\nGalaxy image synthesis completed successfully.")
        except Exception as e:
            print(f"\nError during galaxy image synthesis: {e}")
            import traceback
            traceback.print_exc() # Print full traceback for debugging