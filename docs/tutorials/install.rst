Installation
============


Installing stable version
-------------------------
GalSyn is available as a package on PyPI and can be installed by executing the following command:

.. code-block:: bash
    
    pip install galsyn


Installing development version
------------------------------
If you want to install the most recent version of GalSyn, you can clone it from its GitHub repository and install:

.. code-block:: bash
    
    git clone https://github.com/aabdurrouf/GalSyn.git
    cd GalSyn
    python -m pip install .


Additional (optional) packages
------------------------------
GalSyn requires Python 3.9 or newer. The core dependencies (NumPy, SciPy, Astropy, h5py, joblib, tqdm, tqdm_joblib, and requests) are installed automatically.
The packages below are not installed by default, because they are only needed for specific tasks. Install the ones you need:

* **SPS engine** — needed to generate SSP grid files (see :doc:`ssp_grids`) or to run the synthesis with on-the-fly SPS calls (i.e., when no precomputed SSP grid file is provided). Install at least one of them:

  * `python-fsps <https://dfm.io/python-fsps/current/>`_ for the FSPS engine: ``pip install fsps``
  * `Bagpipes <https://bagpipes.readthedocs.io/en/latest/>`_ for the Bagpipes engine: ``pip install bagpipes``

  GalSyn does not need either of them when the synthesis is run with a precomputed SSP grid file.

* **Matplotlib** — needed to run the plotting parts of the example notebooks and tutorials: ``pip install matplotlib``
* **Pandeia engine** — needed only for estimating the JWST NIRSpec IFU sensitivity in the IFU observation tutorial (see :doc:`observe`).
* **illustris_python** — needed only for the tutorial on selecting a galaxy sample from the IllustrisTNG group catalogs (see :doc:`analyze_data`).
