=================================
Installation
=================================

Basic install
=================================

ATLAST is not on PyPI; install it from the repository in editable mode:

.. code-block:: bash

   git clone https://github.com/SterlingYM/ATLAST
   cd ATLAST
   pip install -e .

This installs the ``atlast`` package into the active Python environment, together
with its required dependencies: ``numpy>=1.26``, ``scipy>=1.11``, ``matplotlib>=3.8``,
``pandas>=2.1``, ``astropy>=6.0``, ``tqdm>=4.66``, and ``requests>=2.32``.

Verify the install:

.. code-block:: python

   from atlast import AtlasPhotometry, QueryATLAS, df_chip
   print(df_chip)

``df_chip`` is the bundled camera/chip table (see :doc:`reference_data`); printing it
confirms that both the code and its packaged data files were installed.

Optional dependencies
=================================

Several features import their dependencies lazily, so they only fail at the moment
you call them:

.. code-block:: bash

   pip install sncosmo        # light curve fitting
   pip install csaps          # spline smoothing / outlier clipping
   pip install scikit-learn   # automatic spline smoothing parameter

``sncosmo``
   Required by :meth:`~atlast.api_ATLAS.AtlasPhotometry.fit_lc`,
   :meth:`~atlast.api_ATLAS.AtlasPhotometry.plot_sncosmo_lc`,
   :meth:`~atlast.api_ATLAS.AtlasPhotometry.clip_outliers_sncosmo`, and
   :func:`~atlast.api_ATLAS.init_sncosmo_filters`. Note that SNCosmo will also need
   to download the model files (e.g. SALT2) the first time a model is used.

``csaps``
   Required by :meth:`~atlast.api_ATLAS.AtlasPhotometry.clip_outliers` and by
   ``plot_lc(plot_interp=True)``, both of which fit a smoothing spline through the
   light curve.

``scikit-learn``
   Only used when ``clip_outliers`` is called without an explicit ``csaps_smooth``,
   in which case the smoothing parameter is chosen by K-fold cross-validation.

Building the documentation
=================================

.. code-block:: bash

   pip install sphinx sphinx_rtd_theme nbsphinx
   conda install -c conda-forge pandoc      # or: brew install pandoc
   cd docs
   make html

The rendered HTML is written to ``docs/build/html/index.html``.

``pandoc`` is a non-Python dependency of ``nbsphinx``, which uses it to convert the
markdown cells of the tutorial notebooks. Without it the build fails with
``PandocMissing``; it cannot be installed with ``pip`` alone.

Notebooks are *not* executed during the build (``nbsphinx_execute = 'never'`` in
``conf.py``), since a notebook may depend on an API token or on local data files.
Their stored outputs are rendered as they are, so re-run and re-save a notebook when
its output should change:

.. code-block:: bash

   jupyter nbconvert --to notebook --execute --inplace \
       source/notebooks/quick_look.ipynb

Data files shipped with the package
===================================

Three data files are installed alongside the code and are located at import time via
:mod:`importlib.resources`:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - File
     - Contents
   * - ``atlast/chip_info.dat``
     - Camera (``sitecam``) and chip nickname, with the MJD range over which each
       chip was in use. Loaded into :data:`atlast.api_ATLAS.df_chip`.
   * - ``atlast/ATLAS_cyan.dat``
     - Cyan (``c``) bandpass transmission curve, from the SVO Filter Profile Service.
   * - ``atlast/ATLAS_orange.dat``
     - Orange (``o``) bandpass transmission curve.

The two filter curves are registered with SNCosmo by
:func:`~atlast.api_ATLAS.init_sncosmo_filters`; you do not need to download them
separately.
