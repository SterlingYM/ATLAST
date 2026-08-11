=================================
Reference data and conventions
=================================

This page collects the fixed numbers, tables and conventions that the rest of the
package assumes.

Filters
=================================

ATLAS observes in two broad filters, referred to throughout by their single-character
codes -- the same codes that appear in the ``F`` column of a photometry file and in
the ``filters`` argument of :meth:`~atlast.api_ATLAS.AtlasPhotometry.cut`:

.. list-table::
   :header-rows: 1
   :widths: 15 20 65

   * - Code
     - Name
     - Approximate coverage
   * - ``o``
     - orange
     - ~5600–8200 Å, roughly *r* + *i*
   * - ``c``
     - cyan
     - ~4200–6500 Å, roughly *g* + *r*

Transmission curves are shipped as ``atlast/ATLAS_orange.dat`` and
``atlast/ATLAS_cyan.dat`` (two columns: wavelength in Å, transmission), sourced from
the SVO Filter Profile Service. :func:`~atlast.api_ATLAS.init_sncosmo_filters`
registers them with SNCosmo under the names ``'o'`` and ``'c'``.

Flux scale
=================================

ATLAS reports difference flux in microJansky, an AB scale with zero point 23.9.
ATLAST converts to the zero point given by the ``zp`` argument, default ``27.5``:

.. math::

   f = f_{\mu\mathrm{Jy}} \times 10^{-0.4\,(23.9 - \mathrm{zp})}

With ``zp=27.5`` this is the SNANA ``FLUXCAL`` scale, which is why
:meth:`~atlast.api_ATLAS.AtlasPhotometry.to_SNANA` writes fluxes unchanged. Fluxes are
signed; negative values at low signal-to-noise are expected and must not be discarded.

Template changes
=================================

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Attribute
     - MJD
     - Date
   * - ``t12``
     - 58417
     - Template generation 1 → 2
   * - ``t23``
     - 58882
     - Template generation 2 → 3

See :doc:`corrections` for what to do about a light curve that crosses one.

Cameras and chips
=================================

ATLAS is a multi-telescope survey, and individual cameras have had their detectors
replaced. Both facts matter for calibration, so ATLAST tags every epoch with the chip
that took it.

The ``Obs`` column of a photometry file starts with a three-character *sitecam* code
identifying the telescope and camera (``01a`` = Haleakalā, ``02a`` = Mauna Loa,
``03a`` = Chile, ``04a`` = South Africa). Combining that with the epoch's MJD gives a
unique chip, since each chip was in use over a known date range.

That mapping is the packaged table ``atlast/chip_info.dat``, loaded at import time as
:data:`atlast.api_ATLAS.df_chip`:

.. code-block:: python

   from atlast import df_chip
   print(df_chip)

.. list-table::
   :header-rows: 1
   :widths: 12 14 28 23 23

   * - Chip ID
     - sitecam
     - chip_nickname
     - mjd_min
     - mjd_max
   * - 0
     - 01a
     - acam2
     - 57800.12362165
     - 58715.64170035
   * - 1
     - 01a
     - gold
     - 58719.35390715
     - 59465.63979805
   * - 2
     - 01a
     - green+fuzzy+4x
     - 59466.01857699
     - 59830.64425335
   * - 3
     - 01a
     - magenta+wormy+5x
     - 59830.64425335
     - current
   * - 4
     - 02a
     - acam1
     - 57800.12388325
     - 58717.64234705
   * - 5
     - 02a
     - red
     - 58718.16986815
     - 59519.65651365
   * - 6
     - 02a
     - red+alien+1
     - 59522.12579875
     - current
   * - 7
     - 03a
     - blue+cruddy+2
     - 59561.86449218
     - current
   * - 8
     - 04a
     - gold+freckles+3x
     - 59605.03066507
     - current

**The chip ID is the row index of this table.** That index is what appears in
``phot._chipid`` and ``phot.chipid``, what indexes the correction maps used by
:meth:`~atlast.api_ATLAS.AtlasPhotometry.apply_flux_corr`, what the ``chipid``
argument of :meth:`~atlast.api_ATLAS.AtlasPhotometry.cut` selects on, and what is
appended to the filter name on SNANA export (``ATLAS-c4``). Adding a row anywhere but
the end will therefore invalidate existing correction maps and exported files.

Two details of the loaded table:

* ``'current'`` in the ``mjd_max`` column is replaced with ``999999`` so that
  comparisons work, which leaves that column with ``object`` dtype rather than float.
  Cast before doing arithmetic on it.
* Epochs whose sitecam and MJD match no row get ``_chipid = nan``. These are dropped
  by every call to ``cut``, and become filter name ``ATLAS-c9999`` on export if the
  cut is bypassed. A light curve losing an unexpected number of epochs at load time is
  usually a sign that ``chip_info.dat`` needs a new row appended.

Helper functions
=================================

Two statistics helpers in :mod:`atlast.api_ATLAS` are useful on their own. Neither is
re-exported at package level; import them from the module.

:func:`~atlast.api_ATLAS.chauvenet_sigma_level`
   ``chauvenet_sigma_level(N)`` returns the sigma level at which, for ``N`` samples
   drawn from a Gaussian, fewer than half an event is expected -- i.e.
   :math:`\Phi^{-1}(1 - 1/4N)`. This is what makes
   :meth:`~atlast.api_ATLAS.AtlasPhotometry.clip_outliers_sncosmo` scale its threshold
   with sample size instead of clipping everything at a fixed 3 sigma.

:func:`~atlast.api_ATLAS.sigma_clip_in_pull`
   ``sigma_clip_in_pull(flux, flux_err, sigma_scale=3, maxiter=10)`` returns
   ``(pull, weighted_mean, mask)``. It clips in *pull* space -- residual divided by
   each point's own error -- rather than in flux, which is the right choice when
   errors vary a lot within a group, as they do across a night of ATLAS exposures.

   The reference value is the inverse-variance weighted mean of the kept points, which
   on its own would be vulnerable to a single bad exposure: a catastrophic subtraction
   often carries a *small* formal error and therefore a large weight. Two choices guard
   against that. The iteration is seeded with the **median** flux rather than the
   weighted mean, and the clip width comes from the **normalized MAD** of the pulls
   rather than their standard deviation -- neither is moved much by one extreme point.
   The width is floored at 1, so a group whose scatter already matches its errors is
   clipped at ``sigma_scale`` in the usual sense. Iteration stops once the kept set
   stops changing, or after ``maxiter`` rounds.

:class:`~atlast.api_ATLAS.GridLookup`
   ``GridLookup(grid_data)`` wraps a ``(3, N, N)`` array of ``[xx, yy, zz]`` in a
   :class:`scipy.spatial.cKDTree` and returns the nearest ``zz`` for queried
   coordinates. Used for the chip zero-point maps, but generic:

   .. code-block:: python

      from atlast import GridLookup
      lookup = GridLookup(grid_data)
      zp_offset = lookup(x, y)     # scalar in, scalar out; array in, array out
