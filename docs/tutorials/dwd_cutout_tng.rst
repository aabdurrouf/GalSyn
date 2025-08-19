Download Data from Illustris TNG and Make Standardized Input File
=================================================================


Download HDF5 cutout data for a specific galaxy
-----------------------------------------------

To download a subhalo cutout of a specific galaxy (given snapshot number and subhalo ID) from the IllustrisTNG simulation, you can use the ``download_cutout_subhalo_hdf5`` function:

.. code-block:: python

    from galsyn.simutils_tng import download_cutout_subhalo_hdf5

    # NOTE: Replace with your actual IllustrisTNG API key
    api_key = "YOUR_API_KEY_HERE"

    snap_number = 39
    subhalo_id = 107965
    sim = 'TNG50-1'
    cutout_name = download_cutout_subhalo_hdf5(snap_number, subhalo_id,
                                                api_key=api_key, sim=sim)



Make standardized input file
----------------------------

After downloading the cutout, the next step is to convert it into the standardized HDF5 format that ``GalSyn`` uses as input:

.. code-block:: python

    from galsyn.simutils_tng import make_sim_file_from_tng_data, get_snap_z

    input_hdf5 = cutout_name
    z = get_snap_z(snap_number, api_key=api_key)
    print ('cutout file name: %s' % input_hdf5)
    print ('Redshift: %lf' % z)
    sim_file = "sim_file_tng_"+str(snap_number)+'_'+str(subhalo_id)+'.hdf5'  # output file name
    make_sim_file_from_tng_data(input_hdf5, z, cosmo_h=0.6774, XH=0.76, output_hdf5=sim_file)