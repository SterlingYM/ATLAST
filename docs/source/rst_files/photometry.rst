=================================
Working with light curves
=================================

:class:`~atlast.api_ATLAS.AtlasPhotometry` is the workhorse of the package. It reads a
downloaded photometry file, converts it to a common flux scale, and exposes the data
as plain numpy arrays that every other method operates on.

Loading a file
=================================

.. code-block:: python

   from atlast import AtlasPhotometry

   phot = AtlasPhotometry('phot_output/job12345.txt', objname='2021abc')

Constructor arguments:

``phot_file``
   Path to the ATLAS forced photometry text file.

``objname``
   Label used in plot titles and carried through to binned copies. Optional.

``zp``
   Zero point of the output flux scale, default ``27.5`` (the SNANA convention).

``warn_template_changes``
   If ``True`` (default), print a warning when the light curve spans one of the
   known template changes. See :doc:`corrections`.

Loading does a fair amount of work up front: it parses the file, looks up the chip
for every epoch, converts fluxes, and applies the default cuts.

The flux scale
=================================

ATLAS reports difference flux in microJansky (``uJy``), which is an AB scale with a
zero point of 23.9. ATLAST rescales to the zero point you asked for:

.. math::

   f = f_{\mu\mathrm{Jy}} \times 10^{-0.4\,(23.9 - \mathrm{zp})}

With the default ``zp=27.5`` this is the SNANA ``FLUXCAL`` scale, which is why plots
are labelled "SNANA flux" and why :doc:`snana` export needs no further conversion.
Errors are scaled by the same factor. Fluxes are signed -- a non-detection is a small
number that may well be negative, not a missing value -- so avoid working in
magnitudes when the source is faint.

Two sets of attributes
=================================

Every quantity exists twice:

* **Underscore-prefixed arrays** (``_mjd``, ``_flux``, ``_flux_err``, ``_filters``,
  ``_mag``, ``_magerr``, ``_chi2dof``, ``_chipid``, ``_sitecam``) hold *every* row in
  the file, in file order. They are never subset.
* **Bare arrays** (``mjd``, ``flux``, ``flux_err``, ``filters``, ``mag``, ``magerr``,
  ``chi2dof``, ``chipid``) hold only the rows that pass the current cuts.

``phot.cuts`` is the boolean mask connecting the two, and ``phot.df_phot`` is the raw
:class:`pandas.DataFrame` the file was read into.

The consequence is that cuts are **not** cumulative by default: each call to
:meth:`~atlast.api_ATLAS.AtlasPhotometry.cut` recomputes the mask from the full
dataset. To narrow an existing selection, pass the current mask in via ``s``:

.. code-block:: python

   phot.cut(fluxerr_max=3000)            # starts from all finite data
   phot.cut(chi2dof_max=5)               # ALSO starts from all finite data -- the
                                         # fluxerr cut above is gone
   phot.cut(s=phot.cuts, chi2dof_max=5)  # narrows the existing selection

Because corrections such as
:meth:`~atlast.api_ATLAS.AtlasPhotometry.subtract_flux_offset` modify the underscore
arrays in place and then re-apply ``phot.cuts``, the ordering of cuts and corrections
does not otherwise matter.

Applying cuts
=================================

.. code-block:: python

   phot.cut(fluxerr_max=3000, chi2dof_max=5)

Any combination of the following may be given; each is applied only if not ``None``:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Argument
     - Cut
   * - ``mag_min`` / ``mag_max``
     - Magnitude range.
   * - ``magerr_min`` / ``magerr_max``
     - Magnitude uncertainty range.
   * - ``mjd_min`` / ``mjd_max``
     - Time range.
   * - ``flux_min`` / ``flux_max``
     - Flux range, on the rescaled (SNANA) scale.
   * - ``fluxerr_min`` / ``fluxerr_max``
     - Flux uncertainty range. A cut at ``fluxerr_max=3000`` is a sensible floor for
       almost any ATLAS light curve; ``2000`` is common.
   * - ``chi2dof_max``
     - Maximum reduced chi-square of the PSF fit (the ``chi/N`` column).
   * - ``chipid``
     - Keep only epochs from one chip (an index into :data:`~atlast.api_ATLAS.df_chip`).
   * - ``filters``
     - List of filters to keep, default ``['o', 'c']``.
   * - ``sigma_max`` / ``res_max``
     - Clip on the residual from a smoothing spline; see below.
   * - ``sncosmo_res_max``
     - Clip on the residual from a fitted SNCosmo model.
   * - ``flux_pull_max``
     - Clip on the nightly pull; requires
       :meth:`~atlast.api_ATLAS.AtlasPhotometry.calc_nightly_pull` to have been run.
   * - ``s``
     - Starting boolean mask, as described above.

Epochs with non-finite flux or flux error, or with no matching chip, are always
dropped.

Plotting
=================================

.. code-block:: python

   fig, ax = phot.plot_lc(title='SN 2021abc')

Orange and cyan points are drawn in their respective colours, with a dotted line at
zero flux. If a sigma-clipping step has run, the clipped points are drawn as well and
marked with a red ``x``, which makes it easy to check that clipping did something
sensible. ``plot_lc`` returns the ``(fig, ax)`` pair, and accepts an existing ``ax``
so light curves can be tiled into a grid.

``plot_lc(plot_interp=True)`` overlays the smoothing spline (requires ``csaps``).

Nightly binning
=================================

ATLAS typically takes four exposures of a field in quick succession. Combining them
raises the signal-to-noise per point substantially:

.. code-block:: python

   phot_binned = phot.group_by_mjd(delta_mjd=0.5, sigma_clip_level=False)
   phot_binned.plot_lc()

Consecutive epochs in the same filter separated by less than ``delta_mjd`` days are
combined with an inverse-variance weighted mean; the binned MJD is the weighted mean
of the epoch MJDs. Groups containing a non-positive error are dropped.

``sigma_clip_level`` optionally applies :func:`astropy.stats.sigma_clip` within each
group before averaging, but only for groups of at least ``min_Ndata_for_clip`` points
(default 4). Clipping three points at 3 sigma is not meaningful, hence the floor.

The return value is an :class:`~atlast.api_ATLAS.AtlasBinnedPhotometry`, a subclass
that carries ``objname`` and ``zp`` over and supports the same cuts, plotting, fitting
and export. It does *not* carry magnitudes, chip IDs, or chi-square across, so cuts
and methods that depend on those (``chi2dof_max``, ``chipid``, ``clip_outliers``,
``apply_flux_corr``) apply to the *unbinned* object only. Do your chip-level
corrections first, then bin.

Nightly pull
=================================

An alternative to clipping inside the binning step is to compute, for every epoch,
how far it sits from its own night's weighted mean, in units of its own error:

.. math::

   \mathrm{pull} = \frac{f - \bar{f}_{\mathrm{night}}}{\sigma_f}

.. code-block:: python

   phot.calc_nightly_pull(delta_mjd=0.5, pull_clip=3)
   phot.cut(s=phot.cuts, flux_pull_max=3)

The nightly weighted mean is itself computed with an iterative clip performed in pull
space (:func:`~atlast.api_ATLAS.sigma_clip_in_pull`), so that a single bad exposure
does not drag the reference value it is being compared against. ``calc_nightly_pull``
adds ``MJD_group``, ``flux_group_wmean``, ``flux_pull`` and ``clip_mask`` columns to
``phot.df_phot`` and populates ``phot._flux_pull``.

For the underlying per-group statistics -- median, mean, weighted mean, standard
deviation and normalised MAD for each night -- call
:meth:`~atlast.api_ATLAS.AtlasPhotometry.stats_grouped_by_mjd` directly, which returns
a DataFrame with one row per epoch and the group statistics attached.

Spline-based outlier clipping
=================================

For well-sampled light curves, outliers can be found relative to a smoothing spline
through the data (requires ``csaps``):

.. code-block:: python

   phot.clip_outliers(sigma=15, res_max=2)

This iterates: fit a spline, cut points whose residual exceeds ``res_max`` mag or
``sigma`` times their own error, refit. If ``csaps_smooth`` is not given, the
smoothing parameter is selected by 5-fold cross-validation, which needs
``scikit-learn`` and is noticeably slower.

.. warning::

   ``clip_outliers`` works in *magnitude* space and therefore requires the ``m`` and
   ``dm`` columns. It is not available on binned light curves, and it is unreliable
   for faint sources where many epochs have no valid magnitude. For those cases prefer
   the nightly pull, or the model-based clipping described in :doc:`fitting`.
