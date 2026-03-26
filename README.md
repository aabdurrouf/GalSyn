# GalSyn

<div align="center">
  <img src="galsyn_logo.png" width="300">
</div>

<div align="center">
  <img src="docs/figures/idealized_images1.png" width="800">
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

For more detailed information about the physical ingredients and algorithms, please see [Abdurro'uf et al. (2026)](https://ui.adsabs.harvard.edu/abs/2026arXiv260323986A/abstract).


## Installation

### Installing stable version

GalSyn is available as a package on PyPI and can be installed by executing the following command:

```
pip install galsyn
```

### Installing development version

If you want to install the most recent version of GalSyn, you can clone it from its GitHub repository and install:

```
git clone https://github.com/aabdurrouf/GalSyn.git
cd GalSyn
python -m pip install .
```

## Citation
If you use this code for your research, please reference [Abdurro'uf et al. (2026)](https://ui.adsabs.harvard.edu/abs/2026arXiv260323986A/abstract):

```
@ARTICLE{2026arXiv260323986A,
       author = {{Abdurro'uf} and {Ferguson}, Henry C. and {Salim}, Samir and {Iyer}, Kartheik G. and {Bradley}, Larry D. and {Coe}, Dan and {Saputra Haryana}, Novan and {Hassan}, Sultan and {Jung}, Intae and {Khullar}, Gourav and {Morishita}, Takahiro and {Mowla}, Lamiya},
        title = "{GalSyn I: A Forward-Modeling Framework for Synthetic Galaxy Observations from Hydrodynamical Simulations and First Data Release from IllustrisTNG}",
      journal = {arXiv e-prints},
     keywords = {Astrophysics of Galaxies},
         year = 2026,
        month = mar,
          eid = {arXiv:2603.23986},
        pages = {arXiv:2603.23986},
archivePrefix = {arXiv},
       eprint = {2603.23986},
 primaryClass = {astro-ph.GA},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2026arXiv260323986A},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
``` 
