import os, sys
import numpy as np
from . import config
# Import both run modules
from . import galsyn_run_fsps
from . import galsyn_run_bagpipes
import importlib.resources

class GalaxySynthesizer:

    def __init__(self, sim_file=None, z=0.01, filters=[], filter_transmission_path={}):
        self._sim_file = sim_file
        self._z = z
        self._filters = filters
        self._filter_transmission_path = filter_transmission_path

        # Initialize with default values from config.py
        self._load_config_defaults()

        # Set other default parameters
        self._dim_kpc = None
        self._pix_arcsec = 0.02
        self._flux_unit = 'MJy/sr'
        self._polar_angle_deg = 0
        self._azimuth_angle_deg = 0
        self._ncpu = 4
        self._initdim_kpc = 100
        self._initdim_mass_fraction = 0.99
        self._name_out_img = None
        self._use_precomputed_ssp = True
        self._ssp_filepath = None
        self._ssp_interpolation_method = 'nearest'
        self._output_pixel_spectra = False
        self._rest_wave_min = 1000.0
        self._rest_wave_max = 16000.0
        self._ssp_code = 'FSPS' # New parameter: default to FSPS

    def _load_config_defaults(self):
        """
        Loads default parameter values from the config.py module.
        """
        # Hydrogen fraction
        self._XH = getattr(config, 'XH', 0.76)

        # Cosmology
        self._cosmo_str = getattr(config, 'COSMO', "Planck18")
        self._cosmo_h = getattr(config, 'COSMO_LITTLE_H', 0.6774)

        # IMF setup (FSPS specific, but kept for consistency if FSPS is chosen)
        self._imf_type = getattr(config, 'IMF_TYPE', 1)
        self._imf_upper_limit = getattr(config, 'IMF_UPPER_LIMIT', 120.0)
        self._imf_lower_limit = getattr(config, 'IMF_LOWER_LIMIT', 0.08)
        self._imf1 = getattr(config, 'IMF1', 1.3)
        self._imf2 = getattr(config, 'IMF2', 2.3)
        self._imf3 = getattr(config, 'IMF3', 2.3)
        self._vdmc = getattr(config, 'VDMC', 0.08)
        self._mdave = getattr(config, 'MDAVE', 0.5)

        # nebular emission
        self._add_neb_emission = getattr(config, 'ADD_NEB_EMISSION', 1)
        self._gas_logu = getattr(config, 'GAS_LOGU', -2.0)

        # IGM absorption
        self._add_igm_absorption = getattr(config, 'ADD_IGM_ABSORPTION', 1)
        self._igm_type = getattr(config, 'IGM_TYPE', 0)

        # Dust attenuation tau normalization as function of redshift
        self._scale_dust_redshift = getattr(config, 'SCALE_DUST_REDSHIFT', "Vogelsberger20")

        # Dust attenuation setup
        self._dust_law = getattr(config, 'DUST_LAW', 0)
        self._dust_index = getattr(config, 'DUST_INDEX', 0.0)
        self._bump_amp = getattr(config, 'BUMP_AMP', 0.85)
        self._dust_index_bc = getattr(config, 'DUST_INDEX_BC', -0.7)
        self._t_esc = getattr(config, 'T_ESC', 0.01)

        # New parameter: A_V vs dust_index relation
        self._relation_AVslope = getattr(config, 'RELATION_AVSLOPE', "Salim18")

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
    def filter_transmission_path(self):
        return self._filter_transmission_path

    @filter_transmission_path.setter
    def filter_transmission_path(self, value):
        if not isinstance(value, dict):
            raise ValueError("filter_transmission_path must be a dictionary.")
        for key, path in value.items():
            if not isinstance(key, str) or not isinstance(path, str):
                raise ValueError("Keys and values in filter_transmission_path must be strings.")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Filter transmission file not found at: {path}")
        self._filter_transmission_path = value

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
    def ssp_code(self):
        return self._ssp_code

    @ssp_code.setter
    def ssp_code(self, value):
        if not isinstance(value, str) or value.upper() not in ['FSPS', 'BAGPIPES']:
            raise ValueError("ssp_code must be 'FSPS' or 'Bagpipes'.")
        self._ssp_code = value.upper()

    @property
    def imf_type(self):
        return self._imf_type

    @imf_type.setter
    def imf_type(self, value):
        if not isinstance(value, int) or value not in [0, 1, 2, 3, 4]:
            raise ValueError("imf_type must be an integer (0, 1, 2, 3, or 4).")
        self._imf_type = value

    @property
    def imf_upper_limit(self):
        return self._imf_upper_limit

    @imf_upper_limit.setter
    def imf_upper_limit(self, value):
        if not isinstance(value, (int, float)) or value <= 0:
            raise ValueError("imf_upper_limit must be a positive number.")
        self._imf_upper_limit = value

    @property
    def imf_lower_limit(self):
        return self._imf_lower_limit

    @imf_lower_limit.setter
    def imf_lower_limit(self, value):
        if not isinstance(value, (int, float)) or value <= 0:
            raise ValueError("imf_lower_limit must be a positive number.")
        self._imf_lower_limit = value

    @property
    def imf1(self):
        return self._imf1

    @imf1.setter
    def imf1(self, value):
        if not isinstance(value, (int, float)):
            raise ValueError("imf1 must be a number.")
        self._imf1 = value

    @property
    def imf2(self):
        return self._imf2

    @imf2.setter
    def imf2(self, value):
        if not isinstance(value, (int, float)):
            raise ValueError("imf2 must be a number.")
        self._imf2 = value

    @property
    def imf3(self):
        return self._imf3

    @imf3.setter
    def imf3(self, value):
        if not isinstance(value, (int, float)):
            raise ValueError("imf3 must be a number.")
        self._imf3 = value

    @property
    def vdmc(self):
        return self._vdmc

    @vdmc.setter
    def vdmc(self, value):
        if not isinstance(value, (int, float)):
            raise ValueError("vdmc must be a number.")
        self._vdmc = value

    @property
    def mdave(self):
        return self._mdave

    @mdave.setter
    def mdave(self, value):
        if not isinstance(value, (int, float)):
            raise ValueError("mdave must be a number.")
        self._mdave = value

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
    def scale_dust_redshift(self):
        return self._scale_dust_redshift

    @scale_dust_redshift.setter
    def scale_dust_redshift(self, value):
        if isinstance(value, str):
            if value not in ["Vogelsberger20"]:
                raise ValueError("scale_dust_redshift string must be 'Vogelsberger20'.")
        elif isinstance(value, dict):
            if "z" not in value or "tau_dust" not in value:
                raise ValueError("scale_dust_redshift dictionary must contain 'z' and 'tau_dust' keys.")
            if not isinstance(value["z"], (list, np.ndarray)) or not isinstance(value["tau_dust"], (list, np.ndarray)):
                raise ValueError("Both 'z' and 'tau_dust' in scale_dust_redshift dictionary must be lists or numpy arrays.")
            if len(value["z"]) != len(value["tau_dust"]):
                raise ValueError("'z' and 'tau_dust' arrays in scale_dust_redshift must have the same length.")
            try:
                np.asarray(value["z"], dtype=float)
                np.asarray(value["tau_dust"], dtype=float)
            except ValueError:
                raise ValueError("All elements in 'z' and 'tau_dust' arrays must be numeric.")
        else:
            raise ValueError("scale_dust_redshift must be a string ('Vogelsberger20') or a dictionary with 'z' and 'tau_dust' keys.")
        self._scale_dust_redshift = value

    @property
    def cosmo_str(self):
        return self._cosmo_str

    @cosmo_str.setter
    def cosmo_str(self, value):
        accepted_cosmo_models = ["planck18", "planck15", "planck13", "wmap5", "wmap7", "wmap9"]
        if not isinstance(value, str) or value.lower() not in accepted_cosmo_models:
            raise ValueError(f"cosmo_str must be one of {accepted_cosmo_models}.")
        self._cosmo_str = value.lower()

    @property
    def cosmo_h(self):
        return self._cosmo_h

    @cosmo_h.setter
    def cosmo_h(self, value):
        if not isinstance(value, (int, float)):
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
    def relation_AVslope(self):
        return self._relation_AVslope

    @relation_AVslope.setter
    def relation_AVslope(self, value):
        if isinstance(value, str):
            if value not in ["Salim18", "Nagaraj22", "Battisti19"]:
                raise ValueError("relation_AVslope string must be 'Salim18', 'Nagaraj22', or 'Battisti19'.")
        elif isinstance(value, dict):
            if "AV" not in value or "dust_index" not in value:
                raise ValueError("relation_AVslope dictionary must contain 'AV' and 'dust_index' keys.")
            if not isinstance(value["AV"], (list, np.ndarray)) or not isinstance(value["dust_index"], (list, np.ndarray)):
                raise ValueError("Both 'AV' and 'dust_index' in relation_AVslope dictionary must be lists or numpy arrays.")
            if len(value["AV"]) != len(value["dust_index"]):
                raise ValueError("'AV' and 'dust_index' arrays in relation_AVslope must have the same length.")
            try:
                # Attempt to convert to numpy arrays and check if elements are numeric
                np.asarray(value["AV"], dtype=float)
                np.asarray(value["dust_index"], dtype=float)
            except ValueError:
                raise ValueError("All elements in 'AV' and 'dust_index' arrays must be numeric.")
        else:
            raise ValueError("relation_AVslope must be a string ('Salim18', 'Nagaraj22', 'Battisti19') or a dictionary with 'AV' and 'dust_index' keys.")
        self._relation_AVslope = value

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
        if value is not None and not isinstance(value, str):
            raise ValueError("ssp_filepath must be a string or None.")
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
    def ssp_interpolation_method(self):
        return self._ssp_interpolation_method

    @ssp_interpolation_method.setter
    def ssp_interpolation_method(self, value):
        if not isinstance(value, str) or value.lower() not in ['nearest', 'linear', 'cubic']:
            raise ValueError("ssp_interpolation_method must one of these options: 'nearest', 'linear', 'cubic'.")
        self._ssp_interpolation_method = value.lower()

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
                    # This can happen if a property raises an error during access (e.e., if a required init param is missing)
                    pass
        return f"GalaxySynthesizer({', '.join(params_list)})"

    def generate_ssp_data(self):
        """
        Determines the SSP spectra grid filepath and ensures it exists.
        If _ssp_filepath is None, it defaults to the packaged data file based on ssp_code.
        If the determined SSP file does not exist, it raises a FileNotFoundError.
        """
        if not self.use_precomputed_ssp:
            print("SSP grid generation will be done on-the-fly. Skipping pre-computed SSP grid check.")
            return

        # Determine the effective SSP filepath based on ssp_code
        if self._ssp_filepath is None:
            try:
                if self.ssp_code == 'FSPS':
                    effective_ssp_filepath = str(importlib.resources.files('galsyn.data').joinpath('ssp_spectra.hdf5'))
                    print("Using packaged FSPS SSP data (Chabrier et al. 2003 IMF, MIST isochrone, MILES spectral library, Cloudy nebular emission).")
                elif self.ssp_code == 'BAGPIPES':
                    effective_ssp_filepath = str(importlib.resources.files('galsyn.data').joinpath('ssp_spectra_bagpipes.hdf5'))
                    print("Using packaged Bagpipes SSP data (Kroupa 2001 IMF).")
                else:
                    raise ValueError(f"Unknown ssp_code: {self.ssp_code}. Cannot determine default SSP filepath.")
            except Exception as e:
                print(f"Warning: Could not locate packaged SSP data via importlib.resources. Error: {e}. Falling back to a direct filename check.")
                # Fallback to current directory if packaged data path cannot be determined
                if self.ssp_code == 'FSPS':
                    effective_ssp_filepath = "ssp_spectra.hdf5"
                elif self.ssp_code == 'BAGPIPES':
                    effective_ssp_filepath = "ssp_spectra_bagpipes.hdf5"
                else: # Should not happen due to earlier check, but for safety
                    raise ValueError(f"Unknown ssp_code: {self.ssp_code}. Cannot determine default SSP filepath.")
        else:
            effective_ssp_filepath = self._ssp_filepath
            print(f"Using specified SSP filepath: {effective_ssp_filepath}")

        # Check if the SSP file exists
        if not os.path.exists(effective_ssp_filepath):
            raise FileNotFoundError(
                f"SSP file not found at '{effective_ssp_filepath}'. "
                f"Please ensure the SSP data file exists at this path or is correctly packaged within 'galsyn.data'. "
                f"Alternatively, set use_precomputed_ssp=False to generate SSPs on-the-fly."
            )
        else:
            # Ensure _ssp_filepath is set to the resolved path if it was initially None
            if self._ssp_filepath is None:
                self._ssp_filepath = effective_ssp_filepath
        
        pass


    # --- Method to run the synthesis process ---
    def run_synthesis(self):
        """
        Executes the galaxy image synthesis process using the current parameters
        set in this GalaxySynthesizer instance.
        """
        # Ensure SSP data is ready (checks file existence if pre-computed)
        self.generate_ssp_data()

        try:
            if self.ssp_code == 'FSPS':
                # Call the generate_images function from galsyn_run_fsps
                generate_images_func = galsyn_run_fsps.generate_images
                # FSPS-specific IMF parameters are passed
                ssp_params = {
                    'imf_type': self.imf_type,
                    'imf_upper_limit': self.imf_upper_limit,
                    'imf_lower_limit': self.imf_lower_limit,
                    'imf1': self.imf1,
                    'imf2': self.imf2,
                    'imf3': self.imf3,
                    'vdmc': self.vdmc,
                    'mdave': self.mdave,
                }
            elif self.ssp_code == 'BAGPIPES':
                # Call the generate_images function from galsyn_run_bagpipes
                generate_images_func = galsyn_run_bagpipes.generate_images
                # Bagpipes does not use these IMF parameters directly for SSP generation
                # but they are still part of the GalaxySynthesizer, so we pass only relevant ones.
                ssp_params = {} # Bagpipes IMF type is fixed to Kroupa (2001)
            else:
                raise ValueError(f"Unsupported SSP code: {self.ssp_code}")

            # Common parameters for both SSP codes
            common_params = {
                'sim_file': self.sim_file,
                'z': self.z,
                'filters': self.filters,
                'filter_transmission_path': self.filter_transmission_path,
                'dim_kpc': self.dim_kpc,
                'pix_arcsec': self.pix_arcsec,
                'flux_unit': self.flux_unit,
                'polar_angle_deg': self.polar_angle_deg,
                'azimuth_angle_deg': self.azimuth_angle_deg,
                'name_out_img': self.name_out_img,
                'n_jobs': self.ncpu,
                'ssp_code': self.ssp_code, # Pass ssp_code to the run module for FITS header
                'add_neb_emission': self.add_neb_emission,
                'gas_logu': self.gas_logu,
                'add_igm_absorption': self.add_igm_absorption,
                'igm_type': self.igm_type,
                'dust_index_bc': self.dust_index_bc,
                'dust_index': self.dust_index,
                't_esc': self.t_esc,
                'scale_dust_redshift': self.scale_dust_redshift, # Pass the new parameter
                'cosmo_str': self.cosmo_str,
                'cosmo_h': self.cosmo_h,
                'XH': self.XH,
                'dust_law': self.dust_law,
                'bump_amp': self.bump_amp,
                'relation_AVslope': self.relation_AVslope,
                'salim_a0': self.salim_a0,
                'salim_a1': self.salim_a1,
                'salim_a2': self.salim_a2,
                'salim_a3': self.salim_a3,
                'salim_RV': self.salim_RV,
                'salim_B': self.salim_B,
                'initdim_kpc': self.initdim_kpc,
                'initdim_mass_fraction': self.initdim_mass_fraction,
                'use_precomputed_ssp': self.use_precomputed_ssp, 
                'ssp_filepath': self.ssp_filepath, 
                'ssp_interpolation_method': self.ssp_interpolation_method, 
                'output_pixel_spectra': self.output_pixel_spectra, 
                'rest_wave_min': self.rest_wave_min, 
                'rest_wave_max': self.rest_wave_max
            }
            
            # Merge SSP-specific parameters with common parameters
            all_params = {**common_params, **ssp_params}

            generate_images_func(**all_params)
            
        except Exception as e:
            print(f"\nError during galaxy image synthesis: {e}")
            import traceback
            traceback.print_exc()