Introduction and Capabilities
==============================

What is GalSyn?
---------------
While the current influx of observational data from state-of-the-art facilities (such as the JWST, Euclid, Vera C. Rubin Observatory) provides
a remarkably deep and complex view of galaxy evolution, it also present a challenge in bridging the gap between observations and 
the theoretical physics encoded in the numerical simulations. A critical tool for addressing this challange is the forward modeling 
of galaxy simulations to generate synthetic data cubes that directly mimic the real astronomical observations. 

Two primary methods exist for this: physically rigorous but computationally expensive radiative transfer codes, 
and more efficient but less complex particle-by-particle spectral modeling techniques. 
GalSyn is developed to facilitate an efficient particle-based spectral modeling with extensive flexibility, 
allowing users to systematically explore how different physical assumptions impact the emergent light from galaxies.

**GalSyn** is a powerful and flexible Python package designed to generate synthetic spectrophotometric galaxy observations 
from the outputs of hydrodynamical simulations. Using a highly customizable particle-based spectral modeling approach, 
GalSyn gives user full control over every step of the synthesis process. 
It allows user to experiment with different stellar population models and dust attenuation laws to produce a wide array 
of data products, from broad-band images and IFU cubes to detailed physical property maps.


Key Features
------------
GalSyn is designed with a modular and flexible structure, giving user comprehensive control over the physical ingredients 
used to create synthetic observations.

1. **Simulation Agnostic** -- GalSyn is designed to be independent of any specific hydrodynamical simulation. It can be applied to data from a wide range of simulations, such as IllustrisTNG and EAGLE, by first converting the native simulation output into a standardized HDF5 file that contains only information needed for the synthesis process.

2. **Flexible Stellar Population Synthesis Modeling** -- The code offers extensive control over the assignment of spectra to star particles.

    * Dual SPS Engine Support: GalSyn integrates two of the most widely-used SPS codes, `FSPS <https://dfm.io/python-fsps/current/>`_ and `Bagpipes <https://bagpipes.readthedocs.io/en/latest/>`_, allowing user to choose between different foundational stellar evolution models.
    * Extensive Customization: user can customize nearly every aspect of the stellar emission, including a wide array of choices for:
        - Stellar Isochrones (MIST, Padova, BaSTI, PARSEC, Geneva).
        - Stellar Spectral Libraries (MILES, BaSeL).
        - Initial Mass Functions (Chabrier, Salpeter, Kroupa, and more).
    * Binary Evolution: GalSyn can incorporate the effects of binary stellar evolution through the BPASS models within the FSPS framework.
    * Nebular Emission: Emission from ionized gas is self-consistently modeled using CLOUDY for young stellar populations (age < 10 Myr).

3. **Comprehensive Dust Attenuation Modeling**--GalSyn implements a detailed and highly flexible analytical dust attenuation model.

    * Two-component model: Attenuation is modeled from both the diffuse ISM and an extra component for the dense birth clouds that enshroud young stars (age < 10 Myr).
    * Spatially resolved attenuation: The V-band optical (:math:`A_{V}`) depth for each star particle is dynamically calculated from the properties of the cold gas in its line-of-sight.
    * Comprehensive Suite of Dust attenuation Laws: GalSyn provides extensive options for the dust attenuation curve, including:
        - Fixed Laws: A wide range of empirical laws, including Calzetti (2000), Salim et al. (2018), and extinction curves for the Milky Way, SMC, and LMC.
        - Dynamic Laws: A modified Calzetti law whose power-law slope, UV bump strength, and UV bump width can be set as free parameters or tied dynamically to the local V-band attenuation.

4. **Realistic Kinematic for IFU Data** -- To create realistic IFU data cubes, GalSyn implements a decoupled kinematics model. The stellar continuum and nebular emission lines are Doppler-shifted independently, using the line-of-sight velocities of the star particles and local gas particles, respectively. This allows for the recovery of distinct gas and stellar rotation curves.

5. **Comprehensive Data Products** -- GalSyn outputs a single FITS file containing a collection of data products.

    * Imaging and IFU cubes: The primary outputs are multi-band images and 3D IFU data cubes, with both dust-free and dust-attenuated versions for direct comparison.
    * Physical property maps: A comprehensive suite of 2D maps that provide insight into the galaxy's physical state. These include maps of stellar mass, gas mass, SFR, metallicity, velocity, and velocity dispersion, calculated using summation, mass-weighted, or light-weighted methods.
    * Spatially resolved SFH: A dedicated class, SFHReconstructor, reconstructs the SFH on a pixel-by-pixel basis, producing data cubes of mass formed, SFR, and metallicity as a function of lookback time.

6. **Simulation of Observational Effects** -- GalSyn has a post-processing module that can transform the idealized synthetic data into realistic mock observations. This includes functionalities to:
    
    * Convolve data with a Point Spread Function (PSF) to match a target instrument's resolution.
    * Perform spectral smoothing on IFU cubes to a desired spectral resolution (:math:`R=\lambda/\Delta\,\lambda`).
    * Inject realistic photon (shot) and background noise based on user-specified observational parameters like limiting magnitude and signal-to-noise ratio.

Dust Attenuation modeling
-------------------------

GalSyn provides a sophisticated treatment of dust attenuation, allowing users to choose how the dust content is calculated and which analytical law is applied to the synthesized spectra. 

The dust_method parameter in the GalaxySynthesizer class determines how the V-band attenuation (:math:`A_V`) is calculated for each part of the galaxy: 

    * Line-of-Sight ('los'): This is the default, most physically motivated method. It calculates the optical depth for each individual star particle by integrating the cold hydrogen gas column density and metallicity along the specific line-of-sight to the observer. 

    * SFR Surface Density ('sfr_AV'): An alternative framework that calculates an effective :math:`A_V` for a pixel based on its SFR surface density (:math:`M_{\odot}\,\text{yr}^{-1}\,\text{kpc}^{-2}`). This method utilizes the ``av_sfrden_relation`` to apply empirical relationships between star formation activity and dust obscuration. 

Once the :math:`A_V` is determined, GalSyn applies a specific attenuation curve (``dust_law``) to the spectrum.  The following options are available:

.. list-table:: Available Dust Attenuation Laws in GalSyn
   :widths: 10 30 60
   :header-rows: 1

   * - Option
     - Law Name
     - Description / Relevant Parameters
   * - **0**
     - **Modified Calzetti**
     - Variable slope and bump. Uses ``dust_index`` (:math:`\delta`), ``bump_amp``, and ``bump_dwave``
   * - **1**
     - **Salim et al. (2018)**
     - Custom polynomial curve. Uses ``salim_a0`` to ``salim_a3``, ``salim_B``, and ``salim_RV``.
   * - **2**
     - **Calzetti**
     - Standard starburst attenuation law from Calzetti et al. (2000).
   * - **3**
     - **SMC Gordon**
     - Small Magellanic Cloud extinction law from Gordon et al. (2003).
   * - **4**
     - **LMC Gordon**
     - Large Magellanic Cloud extinction law from Gordon et al. (2003).
   * - **5**
     - **MW Cardelli**
     - Milky Way extinction law from Cardelli, Clayton, & Mathis (1989).
   * - **6**
     - **MW Fitzpatrick**
     - Milky Way extinction law from Fitzpatrick (1999).


Realistic Kinematics for IFU Data
---------------------------------

To produce realistic synthetic IFU data cubes, GalSyn implements a sophisticated decoupled kinematics model. This approach ensures that the stellar and nebular components of a galaxy’s spectrum are Doppler-shifted independently, reflecting their distinct physical origins within the simulated environment.

Rather than assuming ionized gas moves in perfect lockstep with the stars that excite it, GalSyn treats these components separately:

    * Stellar Continuum: Represents the integrated light from stellar populations. It is Doppler-shifted based on the specific line-of-sight velocity of the parent star particle. This methodology directly ties stellar absorption features to the underlying stellar dynamics of the simulation.
    * Nebular Emission: Represents the light from the local ISM. It is shifted according to the light-weighted average velocity of nearby gas particles.

GalSyn identifies "local" gas for nebular shifting by applying physically motivated criteria for each star particle:

    * Spatial Proximity: The gas must be within a close 3D physical distance (typically < 300 pc) of the star particle.
    * Geometric Location: The gas must be located physically in front of the star along the line-of-sight.
    * Star Formation Activity: Only gas particles with a positive instantaneous SFR are considered, ensuring the kinematics trace the active ISM.

This decoupling is essential for creating high-fidelity synthetic observations. It allows researchers to independently measure and analyze gas and stellar rotation curves—a critical technique in observational studies of galaxy dynamics and evolution.

