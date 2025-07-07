import os
from setuptools import setup, find_packages

# Read the contents of your README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='galsyn',
    version='0.1.0',  # Start with a version, increment for updates
    author='Abdurrouf',  # Replace with your name
    author_email='abdurroufastro@gmail.com',  # Replace with your email
    description='A Python package for generating astrophysical images of galaxies from hydrodynamical simulations.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/aabdurrouf/GalSyn',  # Replace with your project's GitHub URL
    packages=find_packages(),  # Automatically finds packages in the current directory
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'License :: OSI Approved :: MIT License',  # Or choose another appropriate license
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Astronomy',
        'Topic :: Scientific/Engineering :: Physics',
        'Development Status :: 3 - Alpha', # Adjust as your project matures
    ],
    python_requires='>=3.8',  # Minimum Python version required
    install_requires=[
        'numpy',            # Fundamental for numerical operations
        'astropy',          # For astronomical calculations (cosmology, FITS)
        'scipy',            # For scientific computing (interpolation)
        'matplotlib',       # For plotting (even if not directly used in final output, good for dev/debug)
        'h5py',             # For reading HDF5 simulation data
        'joblib',           # For CPU-based parallel processing
        'tqdm',             # For progress bars (includes tqdm.joblib)
        'tqdm_joblib',
        'fsps',             # For Flexible Stellar Population Synthesis (requires setup, see notes below)
        'mpi4py',           # For MPI-based parallel processing (requires MPI installation)
        'importlib_metadata', # Explicitly added due to previous traceback, though often a transitive dep
        'psutil',           # If you use it for memory monitoring/debugging
        #'reproject',        # From previous tracebacks (dependency of piXedfit)
        #'dask',             # From previous tracebacks (dependency of reproject)
        #'piXedfit',         # If your code directly uses piXedfit (from tracebacks)
        # 'illustris_python', # NOTE: This package is typically installed directly from its GitHub repository
                              # as it's not available on PyPI. If your code strictly depends on it,
                              # users will need to install it separately, e.g.:
                              # pip install git+https://github.com/illustris/illustris_python.git
                              # If it's a local module you've included, ensure it's part of your
                              # 'galsyn' package structure.
    ],
    # If you have command-line scripts, you can define them here.
    # For example, if you want to run your main generation function from the command line:
    # entry_points={
    #     'console_scripts': [
    #         'galsyn-generate=galsyn.galsyn:main_generation_function',
    #     ],
    # },
)