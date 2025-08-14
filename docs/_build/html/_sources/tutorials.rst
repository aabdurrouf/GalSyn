Tutorials
==========

This section provides practical guides and examples to help you get started with `GalSyn` and explore its features.

.. toctree::
   :maxdepth: 2
   :caption: Tutorials:

   tutorials/getting_started
   tutorials/basic_usage
   tutorials/customizing_ssp_engines

Getting Started
----------------
[A brief introduction to the tutorials section.]

.. _getting-started-tutorial:

Tutorial 1: Installing and Your First GalSyn Program
-----------------------------------------------------

This tutorial will guide you through installing GalSyn and running a simple example.

1.  **Installation:**
    ```bash
    pip install galsyn
    ```
    (Note: If your package is not yet on PyPI, you might need instructions for installing from source, e.g., `pip install -e .` from your project root.)

2.  **A Simple Example:**
    ```python
    import galsyn
    import numpy as np

    # Assuming you have a dummy simulation file or know its structure
    # For a real example, you'd replace 'path/to/your/simfile.hdf5' with actual data.
    # For demonstration, let's assume a placeholder.
    # Replace with actual usage of your GalaxySynthesizer class
    
    # Initialize the synthesizer with basic parameters
    synth = galsyn.GalaxySynthesizer(
        sim_file='path/to/your/simulation.hdf5', # Replace with your simulation file path
        z=0.1, # Redshift of the simulation snapshot
        filters=['sdss_g', 'sdss_r'], # Example filters
        filter_transmission_path={ # Provide paths to your filter transmission files
            'sdss_g': 'path/to/sdss_g.txt',
            'sdss_r': 'path/to/sdss_r.txt'
        }
    )

    # Set some properties (example)
    synth.dim_kpc = 50.0 # Image dimension in kpc
    synth.pix_arcsec = 0.05 # Pixel scale in arcsec/pixel
    synth.ncpu = 8 # Number of CPU cores to use

    # Run the synthesis (this will vary based on your main synthesis method)
    # Example placeholder for a synthesis call:
    # try:
    #     result_images = synth.synthesize_images()
    #     print("Images synthesized successfully!")
    #     # You might save or display the images here
    #     # for band, img in result_images.items():
    #     #     print(f"Image for {band} band has shape: {img.shape}")
    # except Exception as e:
    #     print(f"An error occurred during synthesis: {e}")

    # A more specific example if simulate_galaxy_image is the main method:
    # Assuming simulate_galaxy_image takes parameters like halo_id or coordinates
    # For a full working example, you'd need a valid sim_file and specific object IDs/coordinates
    print("Example setup for GalaxySynthesizer:")
    print(f"  Simulation File: {synth.sim_file}")
    print(f"  Redshift: {synth.z}")
    print(f"  Filters: {synth.filters}")
    print(f"  Image Dimension: {synth.dim_kpc} kpc")
    print(f"  Pixel Scale: {synth.pix_arcsec} arcsec/pixel")
    print(f"  Number of CPUs: {synth.ncpu}")
    ```

    [Explain what the code does and its output.]

.. _basic-usage-tutorial:

Tutorial 2: Basic Data Loading and Analysis
--------------------------------------------

[Content for the second tutorial, e.g., how to load data, perform a basic analysis.]

.. _customizing-ssp-engines:

Tutorial 3: Customizing SSP Engines (FSPS vs Bagpipes)
------------------------------------------------------

[Content for more advanced topics, e.g., how to switch between FSPS and Bagpipes, configure their specific parameters.]