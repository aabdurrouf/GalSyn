import os, sys
import numpy as np
from . import config
from . import galsyn_run_fsps
from . import galsyn_run_bagpipes
import importlib.resources

class GalaxySynthesizer:
    """
    A class to synthesize multi-wavelength galaxy images from simulation data.

    This class orchestrates the process of taking particle data from a
    cosmological simulation, applying a stellar population synthesis (SSP) model
    (like FSPS or Bagpipes), accounting for dust attenuation and cosmological
    effects, and projecting the resulting light onto a 2D grid to create a
    synthetic astronomical image in specified filters.

    Args:
        sim_file (str, optional): Path to the HDF5 simulation file. Defaults to None.
        z (float, optional): Redshift of the galaxy. Defaults to 0.01.
        filters (list, optional): List of filter names for which to generate images. e.g., ['FUV', 'NUV']. Defaults to [].
        filter_transmission_path (dict, optional): Dictionary mapping custom filter names to their transmission curve file paths. Defaults to {}.

    """

    def __init__(self, sim_file=None, z=0.01, filters=[], filter_transmission_path={}):
        """
        Initializes the GalaxySynthesizer with core settings.

        Args:
            sim_file (str, optional): Path to the HDF5 simulation file. Defaults to None.
            z (float, optional): Redshift of the galaxy. Defaults to 0.01.
            filters (list, optional): List of filter names for which to generate images.
                                      e.g., ['FUV', 'NUV']. Defaults to [].
            filter_transmission_path (dict, optional): Dictionary mapping custom filter
                                                       names to their transmission curve
                                                       file paths. Defaults to {}.
        """
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
        self._rest_wave_max = 30000.0
        self._rest_delta_wave = 5.0
        self._ssp_code = 'FSPS'

    def _load_config_defaults(self):
        """
        Loads default parameter values from the config.py module.
        """
        # Cosmology
        self._cosmo_str = getattr(config, 'COSMO', "Planck18")

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
        self._gas_logu = getattr(config, 'GAS_LOGU', -2.0)

        # IGM absorption
        self._igm_type = getattr(config, 'IGM_TYPE', 0)


        # Dust attenuation tau normalization as function of redshift
        self._scale_dust_redshift = getattr(config, 'SCALE_DUST_REDSHIFT', "Vogelsberger20")

        # Dust attenuation setup
        self._dust_law = getattr(config, 'DUST_LAW', 0)
        self._dust_index = getattr(config, 'DUST_INDEX', 0.0)
        self._bump_amp = getattr(config, 'BUMP_AMP', 0.85)
        self._dust_index_bc = getattr(config, 'DUST_INDEX_BC', -0.7)
        self._t_esc = getattr(config, 'T_ESC', 0.01)
        self._dust_eta = getattr(config, 'DUST_ETA', 1.0)

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
        """str: Path to the input HDF5 simulation data file."""
        return self._sim_file

    @sim_file.setter
    def sim_file(self, value):
        if not isinstance(value, str):
            raise ValueError("sim_file must be string.")
        self._sim_file = value

    @property
    def z(self):
        """float: The redshift of the galaxy being synthesized."""
        return self._z

    @z.setter
    def z(self, value):
        if not isinstance(value, (int, float)) or value < 0:
            raise ValueError("z must be a non-negative number.")
        self._z = value

    @property
    def filters(self):
        """list: A list of filter names for which to generate photometric images."""
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
        """dict: Maps custom filter names to their transmission file paths."""
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
        """float or None: The physical side length of the output image in kiloparsecs."""
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
    def flux_unit(self):
        """str: The unit for the output flux in the FITS image."""
        return self._flux_unit

    @flux_unit.setter
    def flux_unit(self, value):
        accepted_units = ['MJy/sr', 'nJy', 'AB magnitude', 'erg/s/cm2/A']
        if not isinstance(value, str) or value not in accepted_units:
            raise ValueError(f"flux_unit must be one of {accepted_units}.")
        self._flux_unit = value

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
        """float: Initial search dimension (kpc) for automatic image sizing."""
        return self._initdim_kpc

    @initdim_kpc.setter
    def initdim_kpc(self, value):
        if not isinstance(value, (int, float)) or value <= 0:
            raise ValueError("initdim_kpc must be a positive number.")
        self._initdim_kpc = value

    @property
    def initdim_mass_fraction(self):
        """float: Mass fraction to enclose for automatic image sizing."""
        return self._initdim_mass_fraction

    @initdim_mass_fraction.setter
    def initdim_mass_fraction(self, value):
        if not isinstance(value, (int, float)) or value <= 0:
            raise ValueError("initdim_mass_fraction must be a positive number.")
        self._initdim_mass_fraction = value

    @property
    def name_out_img(self):
        """str or None: The path and filename for the output FITS image."""
        return self._name_out_img

    @name_out_img.setter
    def name_out_img(self, value):
        if value is not None and not isinstance(value, str):
            raise ValueError("name_out_img must be a string or None.")
        self._name_out_img = value

    @property
    def ssp_code(self):
        """str: The stellar population synthesis code to use ('FSPS' or 'BAGPIPES')."""
        return self._ssp_code

    @ssp_code.setter
    def ssp_code(self, value):
        if not isinstance(value, str) or value.upper() not in ['FSPS', 'BAGPIPES']:
            raise ValueError("ssp_code must be 'FSPS' or 'Bagpipes'.")
        self._ssp_code = value.upper()

    @property
    def imf_type(self):
        """int: FSPS initial mass function (IMF) type (e.g., 1 for Chabrier)."""
        return self._imf_type

    @imf_type.setter
    def imf_type(self, value):
        if not isinstance(value, int) or value not in [0, 1, 2, 3, 4]:
            raise ValueError("imf_type must be an integer (0, 1, 2, 3, or 4).")
        self._imf_type = value

    @property
    def imf_upper_limit(self):
        """The upper limit of the IMF for FSPS, in solar masses. Defaults to 120"""
        return self._imf_upper_limit

    @imf_upper_limit.setter
    def imf_upper_limit(self, value):
        if not isinstance(value, (int, float)) or value <= 0:
            raise ValueError("imf_upper_limit must be a positive number.")
        self._imf_upper_limit = value

    @property
    def imf_lower_limit(self):
        """The lower limit of the IMF for FSPS, in solar masse. Defaults to 0.08"""
        return self._imf_lower_limit

    @imf_lower_limit.setter
    def imf_lower_limit(self, value):
        if not isinstance(value, (int, float)) or value <= 0:
            raise ValueError("imf_lower_limit must be a positive number.")
        self._imf_lower_limit = value

    @property
    def imf1(self):
        """float: Logarithmic slope of the IMF over the range 0.08 < M < 0.5 Msun.

        Only used if `ssp_code=FSPS` and `imf_type=2` (a Kroupa-like broken power-law). Default: 1.3.
        """
        return self._imf1

    @imf1.setter
    def imf1(self, value):
        if not isinstance(value, (int, float)):
            raise ValueError("imf1 must be a number.")
        self._imf1 = value

    @property
    def imf2(self):
        """float: Logarithmic slope of the IMF over the range 0.5 < M < 1.0 Msun.

        Only used if `ssp_code=FSPS` and `imf_type=2` (a Kroupa-like broken power-law). Default: 2.3.
        """
        return self._imf2

    @imf2.setter
    def imf2(self, value):
        if not isinstance(value, (int, float)):
            raise ValueError("imf2 must be a number.")
        self._imf2 = value

    @property
    def imf3(self):
        """float: Logarithmic slope of the IMF over the range 1.0 < M < 120 Msun.

        Only used if `ssp_code=FSPS` and `imf_type=2` (a Kroupa-like broken power-law). Default: 2.3.
        """
        return self._imf3

    @imf3.setter
    def imf3(self, value):
        if not isinstance(value, (int, float)):
            raise ValueError("imf3 must be a number.")
        self._imf3 = value

    @property
    def vdmc(self):
        """float: IMF parameter defined in van Dokkum (2008).

        Only used if `ssp_code=FSPS` and `imf_type=3`. Default: 0.08.
        """
        return self._vdmc

    @vdmc.setter
    def vdmc(self, value):
        if not isinstance(value, (int, float)):
            raise ValueError("vdmc must be a number.")
        self._vdmc = value

    @property
    def mdave(self):
        """float: IMF parameter defined in Dave (2008).

        Only used if `ssp_code=FSPS` and `imf_type=4`. Default: 0.5.
        """
        return self._mdave

    @mdave.setter
    def mdave(self, value):
        if not isinstance(value, (int, float)):
            raise ValueError("mdave must be a number.")
        self._mdave = value

    @property
    def gas_logu(self):
        """float: The gas ionization parameter for nebular emission modeling."""
        return self._gas_logu

    @gas_logu.setter
    def gas_logu(self, value):
        if not isinstance(value, (int, float)):
            raise ValueError("gas_logu must be a number.")
        self._gas_logu = value

    @property
    def igm_type(self):
        """IGM absorption model type. Options are: 0 for Madau et al. 1995 and 1 for Inoue et al. 2014. Defaults to 0. """
        return self._igm_type

    @igm_type.setter
    def igm_type(self, value):
        if not isinstance(value, int):
            raise ValueError("igm_type must be an integer.")
        self._igm_type = value

    @property
    def dust_index_bc(self):
        """Dust index for birth clouds. Defaults to -0.7."""
        return self._dust_index_bc

    @dust_index_bc.setter
    def dust_index_bc(self, value):
        if not isinstance(value, (int, float)):
            raise ValueError("dust_index_bc must be a number.")
        self._dust_index_bc = value

    @property
    def dust_index(self):
        """Dust index for diffuse ISM. Defaults to 0.0."""
        return self._dust_index

    @dust_index.setter
    def dust_index(self, value):
        if not isinstance(value, (int, float)):
            raise ValueError("dust_index must be a number.")
        self._dust_index = value

    @property
    def t_esc(self):
        """Escape time for young stars in Gyr. Defaults to 0.01."""
        return self._t_esc

    @t_esc.setter
    def t_esc(self, value):
        if not isinstance(value, (int, float)) or value < 0:
            raise ValueError("t_esc must be a non-negative number.")
        self._t_esc = value

    @property
    def dust_eta(self):
        """Ratio of the dust attenuation A_V in the birth clouds and the diffuse ISM. Defaults to 1.0."""
        return self._dust_eta

    @dust_eta.setter
    def dust_eta(self, value):
        if not isinstance(value, (int, float)) or value < 0:
            raise ValueError("dust_eta must be a non-negative number.")
        self._dust_eta = value

    @property
    def scale_dust_redshift(self):
        """
        str or dict: Redshift scaling/normalization for dust optical depth.
        Defaults to "Vogelsberger20". Otherwise, provide a dictionary type input with keys of 'z' and 'tau_dust'.
        """
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
        """
        str: The name of the cosmology model to be used. Valid options are: "planck18", "planck15", "planck13",
        "wmap5", "wmap7", or "wmap9". The check is case-insensitive.
        """

        return self._cosmo_str

    @cosmo_str.setter
    def cosmo_str(self, value):
        accepted_cosmo_models = ["planck18", "planck15", "planck13", "wmap5", "wmap7", "wmap9"]
        if not isinstance(value, str) or value.lower() not in accepted_cosmo_models:
            raise ValueError(f"cosmo_str must be one of {accepted_cosmo_models}.")
        self._cosmo_str = value.lower()

    @property
    def dust_law(self):
        """
        int: The dust attenuation law to apply.

        Options:
            0: Modified Calzetti+00 with Bump strength tied to `dust_index`,
               where `dust_index` itself depends on the line-of-sight A_V.
            1: Modified Calzetti+00 with a free Bump strength (`bump_amp`),
               where `dust_index` depends on the line-of-sight A_V.
            2: Modified Calzetti+00 with Bump strength tied to `dust_index`,
               where `dust_index` is a single free parameter for all stars.
            3: Modified Calzetti+00 with both `bump_amp` and `dust_index` as
               free parameters, applied uniformly to all stars.
            4: Salim+18 attenuation law.
            5: The original Calzetti+00 starburst attenuation law.
            6: Small Magellanic Cloud (SMC) extinction law from Gordon+03.
            7: Large Magellanic Cloud (LMC) extinction law from Gordon+03.
            8: Milky Way (MW) extinction law from Cardelli, Clayton, & Mathis (1989).
            9: Milky Way (MW) extinction law from Fitzpatrick (1999).
        """
        return self._dust_law

    @dust_law.setter
    def dust_law(self, value):
        if not isinstance(value, int) or value not in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
            raise ValueError("dust_law must be integer in the range of 0 to 9!")
        self._dust_law = value

    @property
    def bump_amp(self):
        """UV bump amplitude. Defaults to 0.85."""
        return self._bump_amp

    @bump_amp.setter
    def bump_amp(self, value):
        if not isinstance(value, (int, float)):
            raise ValueError("bump_amp must be a number.")
        self._bump_amp = value

    @property
    def relation_AVslope(self):
        """Defines the A_V vs dust_index relation.
            Can be a string with options of "Salim18", "Nagaraj22", and "Battisti19",
            or a dictionary with "AV" and "dust_index" keys (1D arrays).
            Defaults to "Salim18".
        """
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
        """Parameters for Salim et al. (2018) dust law."""
        return self._salim_a0

    @salim_a0.setter
    def salim_a0(self, value):
        if not isinstance(value, (int, float)):
            raise ValueError("salim_a0 must be a number.")
        self._salim_a0 = value

    @property
    def salim_a1(self):
        """Parameters for Salim et al. (2018) dust law."""
        return self._salim_a1

    @salim_a1.setter
    def salim_a1(self, value):
        if not isinstance(value, (int, float)):
            raise ValueError("salim_a1 must be a number.")
        self._salim_a1 = value

    @property
    def salim_a2(self):
        """Parameters for Salim et al. (2018) dust law."""
        return self._salim_a2

    @salim_a2.setter
    def salim_a2(self, value):
        if not isinstance(value, (int, float)):
            raise ValueError("salim_a2 must be a number.")
        self._salim_a2 = value

    @property
    def salim_a3(self):
        """Parameters for Salim et al. (2018) dust law."""
        return self._salim_a3

    @salim_a3.setter
    def salim_a3(self, value):
        if not isinstance(value, (int, float)):
            raise ValueError("salim_a3 must be a number.")
        self._salim_a3 = value

    @property
    def salim_RV(self):
        """Parameters for Salim et al. (2018) dust law."""
        return self._salim_RV

    @salim_RV.setter
    def salim_RV(self, value):
        if not isinstance(value, (int, float)):
            raise ValueError("salim_RV must be a number.")
        self._salim_RV = value

    @property
    def salim_B(self):
        """Parameters for Salim et al. (2018) dust law."""
        return self._salim_B

    @salim_B.setter
    def salim_B(self, value):
        if not isinstance(value, (int, float)):
            raise ValueError("salim_B must be a number.")
        self._salim_B = value
    
    @property
    def ssp_filepath(self):
        """str or None: Path to a pre-computed SSP grid HDF5 file.

        If None, the default SSP grid packaged with galsyn will be used,
        selected based on the `ssp_code` ('FSPS' or 'BAGPIPES').
        """
        return self._ssp_filepath

    @ssp_filepath.setter
    def ssp_filepath(self, value):
        if value is not None and not isinstance(value, str):
            raise ValueError("ssp_filepath must be a string or None.")
        self._ssp_filepath = value

    @property
    def use_precomputed_ssp(self):
        """bool: If True, use a pre-computed SSP grid file for synthesis."""
        return self._use_precomputed_ssp

    @use_precomputed_ssp.setter
    def use_precomputed_ssp(self, value):
        if not isinstance(value, bool):
            raise ValueError("use_precomputed_ssp must be a boolean (True/False).")
        self._use_precomputed_ssp = value

    @property
    def ssp_interpolation_method(self):
        """str: Interpolation method for the SSP grid ('nearest', 'linear', 'cubic')."""
        return self._ssp_interpolation_method

    @ssp_interpolation_method.setter
    def ssp_interpolation_method(self, value):
        if not isinstance(value, str) or value.lower() not in ['nearest', 'linear', 'cubic']:
            raise ValueError("ssp_interpolation_method must one of these options: 'nearest', 'linear', 'cubic'.")
        self._ssp_interpolation_method = value.lower()

    @property
    def output_pixel_spectra(self):
        """bool: If True, output a 3D spectral cube in addition to filter images."""
        return self._output_pixel_spectra

    @output_pixel_spectra.setter
    def output_pixel_spectra(self, value):
        if not isinstance(value, bool):
            raise ValueError("output_pixel_spectra must be a boolean (True/False).")
        self._output_pixel_spectra = value

    @property
    def rest_wave_min(self):
        """float: Minimum rest-frame wavelength (Angstroms) for the spectral cube."""
        return self._rest_wave_min

    @rest_wave_min.setter
    def rest_wave_min(self, value):
        if not isinstance(value, (int, float)) or value <= 0:
            raise ValueError("rest_wave_min must be a positive number.")
        self._rest_wave_min = float(value)

    @property
    def rest_wave_max(self):
        """float: Maximum rest-frame wavelength (Angstroms) for the spectral cube."""
        return self._rest_wave_max

    @rest_wave_max.setter
    def rest_wave_max(self, value):
        if not isinstance(value, (int, float)) or value <= 0:
            raise ValueError("rest_wave_max must be a positive number.")
        self._rest_wave_max = float(value)
        if self._rest_wave_max <= self._rest_wave_min:
            raise ValueError("rest_wave_max must be greater than rest_wave_min.")
        
    @property
    def rest_delta_wave(self):
        """float: Wavelength step (Angstroms) for the spectral cube."""
        return self._rest_delta_wave
    
    @rest_delta_wave.setter
    def rest_delta_wave(self, value):
        if not isinstance(value, (int, float)) or value <= 0:
            raise ValueError("rest_delta_wave must be a positive number.")
        self._rest_delta_wave = float(value)
        if self._rest_delta_wave <= self._rest_wave_min:
            raise ValueError("rest_delta_wave must be greater than rest_wave_min.")


    # --- Convenience method for setting multiple parameters ---
    def set_params(self, **kwargs):
        """
        Sets multiple parameters at once using keyword arguments.

        This provides a convenient way to update the synthesizer's configuration.
        Any invalid parameter names will be ignored with a warning.

        Args:
            **kwargs: Keyword arguments where keys match attribute names
                      (e.g., dim_kpc=50, ssp_code='BAGPIPES').
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
        Checks for and resolves the path to the required SSP grid file.

        If `use_precomputed_ssp` is True, this method verifies that the SSP data
        file exists. If `ssp_filepath` is not specified, it defaults to the
        packaged data file corresponding to the selected `ssp_code`.

        Raises:
            FileNotFoundError: If the required SSP file cannot be found.
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
        Executes the full galaxy image synthesis process.

        This is the main method to run the pipeline. It gathers all configured
        parameters, selects the appropriate backend (FSPS or Bagpipes), calls the
        corresponding engine to perform the synthesis, and saves the resulting
        FITS file.
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
                'gas_logu': self.gas_logu,
                'igm_type': self.igm_type,
                'dust_index_bc': self.dust_index_bc,
                'dust_index': self.dust_index,
                't_esc': self.t_esc,
                'dust_eta': self.dust_eta,
                'scale_dust_redshift': self.scale_dust_redshift, # Pass the new parameter
                'cosmo_str': self.cosmo_str,
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
                'rest_wave_max': self.rest_wave_max,
                'rest_delta_wave': self.rest_delta_wave
            }
            
            # Merge SSP-specific parameters with common parameters
            all_params = {**common_params, **ssp_params}

            generate_images_func(**all_params)
            
        except Exception as e:
            print(f"\nError during galaxy image synthesis: {e}")
            import traceback
            traceback.print_exc()