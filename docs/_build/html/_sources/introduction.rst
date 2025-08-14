Introduction and Capabilities
==============================

What is GalSyn?
---------------
GalSyn (Galaxy Synthesis) is a Python package developed to facilitate generating astrophysical images of galaxies from hydrodynamical simulations. It provides a comprehensive suite of tools for [mention key functionalities, e.g., "generating synthetic galaxy catalogs," "modeling galaxy evolution," "analyzing observational data"].

Key Features
------------
* [cite_start]**Cosmology Handling**: Defines cosmology parameters like `COSMO` and `COSMO_LITTLE_H`[cite: 2].
* [cite_start]**IMF Setup**: Configures the Initial Mass Function (IMF) with parameters such as `IMF_TYPE`, `IMF_UPPER_LIMIT`, `IMF_LOWER_LIMIT`, `IMF1`, `IMF2`, `IMF3`, `VDMC`, and `MDAVE`[cite: 2].
* [cite_start]**Nebular Emission**: Allows control over nebular emission with `ADD_NEB_EMISSION` and `GAS_LOGU`[cite: 2].
* [cite_start]**IGM Absorption**: Includes options for Intergalactic Medium (IGM) absorption through `ADD_IGM_ABSORPTION` and `IGM_TYPE`[cite: 2].
* [cite_start]**Dust Attenuation**: Manages dust attenuation using parameters like `SCALE_DUST_REDSHIFT`, `DUST_LAW`, `DUST_INDEX`, `BUMP_AMP`, `DUST_INDEX_BC`, and `T_ESC`[cite: 2].
* [cite_start]**A_V vs Dust Index Relation**: Supports different relations for A_V vs dust_index, including the "Salim18" relation with parameters `SALIM_A0`, `SALIM_A1`, `SALIM_A2`, `SALIM_A3`, `SALIM_RV`, and `SALIM_B`[cite: 2].
* **Flexible SSP Code Integration**: Can utilize either FSPS or Bagpipes for Single Stellar Population (SSP) synthesis.
* **Precomputed SSP Grids**: Supports loading precomputed SSP grids for faster processing.
* **Filter Transmission Handling**: Loads and processes filter transmission data from external files.
* **Parallel Processing**: Uses `joblib` and `tqdm_joblib` for parallel execution to speed up image generation.

Background
----------
[Provide some background information on the scientific context or the problem that GalSyn aims to address. Why was it developed? What are its theoretical underpinnings?]

Getting Started
---------------
[Briefly explain how users can get started with GalSyn, perhaps linking to the first tutorial.]