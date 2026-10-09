Download Data from Illustris TNG and Make Standardized Input File
=================================================================
 
The first step in the GalSyn workflow is the preparation of input simulation data. 
While GalSyn is designed to be agnostic to the specific hydrodynamical simulation used, 
it requires the raw output to be converted into a standardized HDF5 file format. 
This ensures that the core routines operate on a consistent data structure regardless 
of whether the source is IllustrisTNG, EAGLE, or another simulation.  

The ``simutils_tng`` module provides a streamlined set of tools specifically for handling data from the IllustrisTNG simulation. 
The process involves two main steps: acquiring the data cutout and converting it to the physical units required by GalSyn.


Download HDF5 cutout data for a specific galaxy
-----------------------------------------------

To download a subhalo cutout of a specific galaxy (given snapshot number and subhalo ID) from the IllustrisTNG simulation, you can use the ``download_cutout_subhalo_hdf5`` function:

.. code-block:: python

    from galsyn.simutils_tng import download_cutout_subhalo_hdf5

    # Your personal API key from the IllustrisTNG website
    api_key = "your_api_key"

    # Specify simulation parameters
    sim = 'TNG50-1'         # The TNG simulation run
    snap_number = 62        # The snapshot index (e.g., z ~ 1.5 in IllustrisTNG)
    subhalo_id = 132810     # The subhalo ID

    # Downloads the particle data (gas and stars) for the specified subhalo
    cutout_name = f'cutout_shalo_{int(snap_number)}_{int(subhalo_id)}.hdf5'
    download_cutout_subhalo_hdf5(snap_number, subhalo_id, api_key=api_key, sim=sim, name=cutout_name)



Make standardized input file
----------------------------

The raw TNG cutout must be transformed into the standardized GalSyn input format. 
This step performs necessary unit conversions and identifies the required physical parameters for the next synthesis process.

.. code-block:: python

    from galsyn.simutils_tng import make_sim_file_from_tng_data, get_snap_z

    input_hdf5 = cutout_name

    # Retrieve the exact redshift for the given snapshot number using the TNG API
    z = get_snap_z(snap_number, api_key=api_key)
    print ('Redshift: %lf' % z)

    # Define the output path for the standardized file
    sim_file = f'sim_file_tng_{int(snap_number)}_{int(subhalo_id)}.hdf5'

    # Run the calculation
    make_sim_file_from_tng_data(input_hdf5, z, cosmo_h=0.6774, XH=0.76, output_hdf5=sim_file)