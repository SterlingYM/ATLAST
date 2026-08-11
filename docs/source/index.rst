.. ATLAST documentation master file, created by
   sphinx-quickstart on Mon Mar 10 00:42:41 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to ATLAST!
=================================

.. .. image:: images/welcome.jpg
..    :alt: Example image
..    :width: 100 %
..    :align: center


=================================
What is ATLAST?
=================================
ATLAST (ATLAS Transient Lightcurve Analysis and Smoothing Tools) is an all-in-one
platform to download, analyze, and visualize light curve data from ATLAS. It can do:

* query ATLAS forced photometry
* visualize light curves
* fit light curves with a variety of models
* and many more...

In practice the package is organised around three entry points:

:class:`~atlast.api_ATLAS.QueryATLAS`
   Talks to the ATLAS forced-photometry server: queues a request for a position on
   the sky, then downloads the resulting photometry text file and (optionally) the
   difference images.

:class:`~atlast.api_ATLAS.AtlasPhotometry`
   Wraps a downloaded photometry file. It converts ATLAS microJansky fluxes into the
   SNANA flux scale, tags every epoch with the camera/chip it came from, and gives you
   quality cuts, nightly binning, outlier clipping, zero-level and zero-point
   corrections, SNCosmo fitting, and SNANA-format export.

:class:`~atlast.api_SNANA.SNANAPhotometry` and friends
   Read data back in from SNANA text files and SNANA FITS simulations, so that real
   and simulated light curves can be pushed through the same analysis code.

Each of these is covered in the :doc:`user guide <rst_files/querying>`, and an
end-to-end walkthrough on a real supernova lives in the
:doc:`quick look notebook <notebooks/quick_look>`.

=================================
When to use ATLAST?
=================================
ATLAST is aimed at the step between "I have an ATLAS forced-photometry file" and
"I have a light curve I trust". It is useful when you want to:

* **Pull forced photometry in bulk.** The query layer handles queueing, server-side
  throttling, and saving results, so a list of targets can be looped over.
* **Clean up ATLAS light curves.** Forced photometry on ATLAS difference images
  carries a number of well-known systematics -- template changes partway through a
  light curve, position-dependent zero-point errors across the chip, non-zero
  baseline flux from ill-subtraction, and underestimated flux errors. ATLAST has
  a dedicated correction for each of these; see :doc:`rst_files/corrections`.
* **Bin and clip.** Nightly quads are combined into single points with an inverse-
  variance weighted mean, with sigma-clipping either in flux or in *pull* space.
* **Fit supernova models.** :meth:`~atlast.api_ATLAS.AtlasPhotometry.fit_lc` is a
  thin wrapper over SNCosmo that registers the ATLAS ``o`` and ``c`` bandpasses,
  handles Milky Way dust, and supports iterative fitting with outlier rejection.
* **Export to SNANA.** :meth:`~atlast.api_ATLAS.AtlasPhotometry.to_SNANA` writes a
  standard SNANA-format data file, optionally splitting filters per chip so that
  chip-dependent calibration can be fit downstream.

It is *not* a general-purpose photometry pipeline: it assumes ATLAS forced
photometry as input, and the ``o``/``c`` filter pair throughout.

=================================
What is required to use ATLAST?
=================================

**An ATLAS forced photometry account.**
Queries go through https://fallingstar-data.com/forcedphot, which requires an
account and an API token. Requesting photometry without a token will fail; see
:doc:`rst_files/querying` for how to obtain and pass one. You do not need an
account if you already have photometry files on disk -- everything downstream of
:class:`~atlast.api_ATLAS.AtlasPhotometry` works offline.

**Python 3 and the core scientific stack.**
Installing the package pulls in ``numpy``, ``scipy``, ``matplotlib``, ``pandas``,
``astropy``, ``tqdm``, and ``requests``.

**Optional extras**, imported lazily and only needed for specific features:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Package
     - Needed for
   * - ``sncosmo``
     - :meth:`~atlast.api_ATLAS.AtlasPhotometry.fit_lc`,
       :meth:`~atlast.api_ATLAS.AtlasPhotometry.clip_outliers_sncosmo`,
       :func:`~atlast.api_ATLAS.init_sncosmo_filters`
   * - ``csaps``
     - spline-based outlier clipping
       (:meth:`~atlast.api_ATLAS.AtlasPhotometry.clip_outliers`) and
       ``plot_lc(plot_interp=True)``
   * - ``scikit-learn``
     - automatic selection of the smoothing parameter for the spline above

See :doc:`rst_files/installation` for the install commands.

**TITAN calibration data (optional, access-controlled).**
Two corrections need calibration products that are *not* distributed with this
package: the position-dependent zero-point maps used by
:meth:`~atlast.api_ATLAS.AtlasPhotometry.apply_flux_corr` and the error fudge
function used by :meth:`~atlast.api_ATLAS.AtlasPhotometry.apply_fudge_func`. See
`Marlin et al. <https://arxiv.org/abs/2512.21903>`_ (arXiv:2512.21903) for how they
are derived. The data is reserved for the TITAN team until the science analyses are
complete; contact **Y. S. Murakami** or **Eli Marlin** to request access. Everything
else in ATLAST works without it.


.. toctree::
   :maxdepth: 3
   :hidden:
   :caption: Welcome!

   Home <self>

.. include:: toc.rst


.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`
