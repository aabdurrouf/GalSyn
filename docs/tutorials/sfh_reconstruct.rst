Reconstructing Star Formation History
=====================================

To reconstruct the spatially resolved star formation history (SFH) of the galaxy, you can use the ``SFHReconstructor`` class:

.. code-block:: python

    from galsyn.simutils_tng import get_snap_z
    from galsyn import SFHReconstructor

    snap_number, subhalo_id = 39, 107965
    z = get_snap_z(snap_number, api_key=api_key)
    sim_file = "sim_file_tng_"+str(snap_number)+'_'+str(subhalo_id)+'.hdf5'
    sfh = SFHReconstructor(sim_file, z, Z_sun=0.019)

    sfh.dim_kpc = 80
    sfh.pix_arcsec = 0.03
    sfh.polar_angle_deg = 45
    sfh.azimuth_angle_deg = 60
    sfh.ncpu = 5
    sfh.name_out_sfh = "galsyn_sfh_"+str(snap_number)+"_"+str(subhalo_id)+".fits"
    sfh.sfh_del_t = 0.05         # SFH bin width in Gyr

    sfh.reconstruct_sfh()