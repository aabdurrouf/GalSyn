Setting up filter transmission curves
=====================================

To generate synthetic imaging data cubes using ``GalaxySynthesizer``, user needs to specify list of filters and their transmission curves. 
The ``filters`` input is a list containing filter names (arbitrary) in string, whereas ``filter_transmission_path`` input is a Dictionary of paths to text files containing the transmission function. 
Keys are filter names (as listed in ``filters``), values are file paths. Each text file has two columns: wavelength, transmission.

Alternatively, if you have `piXedfit <https://pixedfit.readthedocs.io/en/latest/>`_ installed, you can set up these inputs easily using ``make_filter_transmission_text_pixedfit`` function. 
It will create a dictionary and text files containing the filter transmission curves taken from the internal piXedfit package. For this to work, you need to use filter naming in piXedfit. 
Below is an example script for this:

.. code-block:: python

    from galsyn.utils import make_filter_transmission_text_pixedfit

    filters = ['hst_acs_f435w', 'hst_acs_f606w', 'hst_acs_f814w', 'hst_wfc3_ir_f110w', 'hst_wfc3_ir_f125w',
                'hst_wfc3_ir_f140w', 'hst_wfc3_ir_f160w', 'jwst_nircam_f090w', 'jwst_nircam_f115w',
                'jwst_nircam_f140m', 'jwst_nircam_f150w', 'jwst_nircam_f200w', 'jwst_nircam_f250m',
                'jwst_nircam_f277w', 'jwst_nircam_f300m', 'jwst_nircam_f356w', 'jwst_nircam_f410m',
                'jwst_nircam_f444w', 'jwst_nircam_f460m', 'jwst_nircam_f480m']

    filter_transmission_path = make_filter_transmission_text_pixedfit(filters, output_dir="filters")

