# GalSyn

<div align="center">
  <img src="galsyn_logo.png" width="300">
</div>

<div align="center">
  <img src="docs/figures/idealized_images1.png" width="700">
</div>


**GalSyn** is a modular Python package designed for generating realistic synthetic spectrophotometric 
observations of galaxies from hydrodynamical simulation data. By employing particle-by-particle spectral modeling to 
3D data from hydrodynamical simulation such as IllustrisTNG and EAGLE, GalSyn enables the generation of realistic 
synthetic spectrophotometric data cubes, including broadband imaging and Integral Field Unit (IFU) spectroscopy. 
Beyond light synthesis, the tool produces comprehensive 2D physical property maps of the stellar populations, gas, 
and dust, as well as the decoupled kinematics of both stellar and gaseous components. 

A core philosophy of GalSyn is providing extensive flexibility over the physical ingredients involved in the 
synthesis procedure. This includes highly flexible control over the stellar population synthesis (SPS) modeling, 
and customize underlying components such as Initial Mass Functions (IMFs), stellar isochrones (e.g., MIST, Padova, BaSTI), 
stellar spectral libraries (e.g., MILES, BaSeL), and binary stellar evolution (BPASS). Furthermore, GalSyn implements 
highly flexible analytical dust attenuation models, allowing users to choose between fixed empirical laws or dynamic 
prescriptions with variable UV bump strengths and power-law slopes.

While traditional radiative transfer codes offer high physical rigor, they are often computationally intensive and 
offer limited flexibility regarding stellar population choices. GalSyn is built for computational efficiency and 
highly flexible user control, allowing for large-scale population studies and systematic exploration of how different 
physical assumptions (like IMF or dust laws) impact emergent galaxy light.

Check out the sections below to learn more about GalSyn's capabilities, how to use it, and its API. 
For more detailed information about the physical ingredients and algorithms, please see **Abdurro'uf et al. (2026, in prep.)**.