Introduction and Capabilities
==============================

Background
----------
The current era of astronomy is defined by an unprecedented influx of observational data from state-of-the-art facilities. 
While this data provides a remarkably deep and complex view of galaxy evolution, it also presents a significant challenge: bridging the gap between raw observations 
and the theoretical physics encoded in numerical simulations. Synthetic observations are a critical tool for meeting this challenge. 
By generating mock observations from hydrodynamical simulations, we can directly compare theoretical predictions with real astronomical data, 
validate galaxy formation models, and interpret complex observational trends.

Two primary methods exist for this: physically rigorous but computationally expensive radiative transfer codes, 
and more efficient but less complex particle-by-particle spectral modeling techniques. 
GalSyn was developed to facilitate anefficient particle-based spectral modeling with extensive flexibility, 
allowing researchers to systematically explore how different physical assumptions impact the emergent light from galaxies.

What is GalSyn?
---------------
**GalSyn** is a powerful and flexible Python package designed to generate synthetic spectrophotometric galaxy observations from the outputs of hydrodynamical simulations. 
Using a highly customizable particle-based spectral modeling approach, GalSyn gives you full control over every step of the synthesis process. 
It allows you to experiment with different stellar population models, dust attenuation laws, and kinematic treatments to produce a wide array of data products, 
from multi-band images and IFU cubes to detailed physical property maps.


Key Features
------------
GalSyn is designed with a modular and flexible structure, giving you comprehensive control over the physical ingredients used to create synthetic observations.

1. **Simulation Agnostic**
GalSyn is designed to be independent of any specific hydrodynamical simulation. It can be applied to data from a wide range of simulations, 
such as IllustrisTNG and EAGLE, by first converting the native simulation output into a standardized HDF5 file format

2. **Flexible Stellar Population Synthesis Modeling**
The code offers extensive control over the assignment of spectra to star particles.

* Dual SPS Engine Support: GalSyn integrates two of the most widely-used SPS codes, FSPS and Bagpipes, allowing you to choose between different foundational stellar evolution models.

* Extensive Customization: You can customize nearly every aspect of the stellar emission, including a wide array of choices for:

    - Stellar Isochrones (MIST, Padova, BaSTI, PARSEC, Geneva).

    - Stellar Spectral Libraries (MILES, BaSeL).

    - Initial Mass Functions (Chabrier, Salpeter, Kroupa, and more).

* Binary Evolution: GalSyn can incorporate the effects of binary stellar evolution through the BPASS models within the FSPS framework.

* Nebular Emission: Emission from ionized gas is self-consistently modeled using CLOUDY for young stellar populations (age < 10 Myr).

3. **Comprehensive Dust Modeling**

GalSyn implements a detailed and highly flexible dust attenuation model.

* Two-Component Model: Attenuation is modeled from both the diffuse ISM and an extra component for the dense birth clouds that enshroud young stars (age < 10 Myr).

* Spatially Resolved Attenuation: The V-band optical depth for each star particle is dynamically calculated from the properties of the cold gas in its line-of-sight.

* Comprehensive Suite of Dust Laws: GalSyn provides extensive options for the dust attenuation curve, including:

    - Fixed Laws: A wide range of empirical laws, including Calzetti (2000), Salim et al. (2018), and extinction curves for the Milky Way, SMC, and LMC.

    - Dynamic Laws: A modified Calzetti law whose power-law slope and UV bump strength can be set as free parameters or tied dynamically to the local V-band attenuation.


4. **Realistic Kinematic for IFU Data**
To create realistic IFU data cubes, GalSyn implements a decoupled kinematics model. The stellar continuum and nebular emission lines are Doppler-shifted independently, 
using the line-of-sight velocities of the star particles and local gas particles, respectively. This allows for the accurate recovery of distinct gas and stellar rotation curves.

5. **Comprehensive Data Products**
GalSyn outputs a single FITS file containing a rich collection of data products.

* Imaging and IFU Cubes: The primary outputs are multi-band images and 3D IFU data cubes, with both dust-free and dust-attenuated versions for direct comparison.

* Physical Property Maps: A comprehensive suite of 2D maps that provide insight into the galaxy's physical state. These include maps of stellar mass, gas mass, SFR, metallicity, velocity, and velocity dispersion, calculated using summation, mass-weighted, or light-weighted methods.

* Spatially Resolved SFH: A dedicated class, SFHReconstructor, reconstructs the SFH on a pixel-by-pixel basis, producing data cubes of mass formed, SFR, and metallicity as a function of lookback time.

6. **Observational Effects Simulation**
GalSyn has a post-processing module that can transform the idealized synthetic data into realistic mock observations. This includes functionalities to:

* Convolve data with a Point Spread Function (PSF) to match a target instrument's resolution.

* Perform spectral smoothing on IFU cubes to a desired spectral resolution R.

* Inject realistic photon (shot) and background noise based on user-specified observational parameters like limiting magnitude and signal-to-noise ratio.

