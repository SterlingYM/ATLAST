=================================
Light curve fitting
=================================

Fitting is a wrapper around `SNCosmo <https://sncosmo.readthedocs.io>`_, which must be
installed separately.

Registering the ATLAS bandpasses
=================================

SNCosmo does not know about the ATLAS filters, so register them once per session
before fitting anything:

.. code-block:: python

   from atlast import init_sncosmo_filters
   init_sncosmo_filters()

This reads the transmission curves shipped with the package and registers them under
the names ``'o'`` (orange) and ``'c'`` (cyan) -- the same single-character codes used
in the ``F`` column of the photometry file, so no filter name mapping is needed.
``force=True`` is the default, meaning re-running it overwrites the existing
registration rather than raising.

Fitting
=================================

.. code-block:: python

   phot.fit_lc(preset_vals={'z': 0.05678})

:meth:`~atlast.api_ATLAS.AtlasPhotometry.fit_lc` builds an
:class:`astropy.table.Table` from the currently-cut data, prepares a model if one has
not been set up yet, and calls ``sncosmo.fit_lc``. Key arguments:

``preset_vals``
   Parameters to *fix* rather than fit, as a dict. Anything listed here is removed
   from the free parameter list. Fixing the redshift is the usual case; ``t0`` and
   ``mwebv`` are also common.

``model_source``
   SNCosmo source name, default ``'salt2'``.

``bounds``
   Parameter bounds. Defaults to ``{'x1': (-5, 5), 'c': (-1, 1)}``, which are SALT2
   parameters -- **if you change the source, change the bounds**. The redshift bounds
   are merged in separately from
   :meth:`~atlast.api_ATLAS.AtlasPhotometry.set_sncosmo_redshift_bounds`, which
   defaults to ``(0.01, 0.1)``. Widen that for anything outside the local universe:

   .. code-block:: python

      phot.set_sncosmo_redshift_bounds((0.01, 0.3))

``correct_mwdust``, ``dust_model``
   Attach a Milky Way dust effect to the model. ``dust_model`` is one of ``'F99'``
   (default), ``'CCM89'`` or ``'OD94'``; the latter two fix :math:`R_V = 3.1`.

``iterate_fit``, ``iterative_phase_cut``
   Fit once, use the preliminary ``t0`` and ``z`` to convert to rest-frame phase, cut
   to ``iterative_phase_cut`` (default ``[-10, 40]`` days), and refit starting from
   the first solution. This keeps late-time and pre-explosion epochs from pulling the
   fit around.

``modelcov``
   Pass model covariance through to SNCosmo. Off by default.

``plot``, ``ax``, ``param_fontsize``
   Plot the fitted model over the data, annotated with the fitted parameters.

``raise_error``
   By default ``fit_lc`` catches any exception, prints ``lc fit failed: ...`` and
   returns ``False``; on success it returns ``True``. Set ``raise_error=True`` while
   debugging a single object, and leave it ``False`` when looping over a sample.

.. warning::

   ``mwebv`` is currently forced to ``0`` when the model is constructed. To apply a
   real Milky Way extinction, pass it explicitly as a preset:
   ``preset_vals={'mwebv': ebv}``. Compute ``ebv`` from the object's coordinates with
   e.g. ``sfdmap``; see the `SNCosmo dust documentation
   <https://sncosmo.readthedocs.io/en/stable/models.html>`_.

After a successful fit:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Attribute
     - Contents
   * - ``sncosmo_result``
     - The SNCosmo ``Result``: ``parameters``, ``errors``, ``chisq``, ``ndof``, ...
       ``None`` if every parameter was preset.
   * - ``sncosmo_fitted_model``
     - The fitted ``sncosmo.Model``.
   * - ``sncosmo_data``
     - The table that was fitted.
   * - ``sncosmo_residual_flux``
     - Data minus model, for the cut epochs.
   * - ``_sncosmo_residual_flux``
     - The same for all epochs.

``phot.plot_sncosmo_lc()`` renders SNCosmo's own multi-panel plot instead of the
single-panel overlay.

Using a non-default model
=================================

Prepare the model explicitly before fitting to control the source and the free
parameter list:

.. code-block:: python

   phot._prep_sncosmo_model(source='hsiao', params_to_fit=['z', 't0', 'amplitude'])
   phot.fit_lc(bounds={'amplitude': (0, 1e-10)}, preset_vals={'z': 0.05678})

Note that ``fit_lc`` determines the free parameters from the model itself minus
whatever is in ``preset_vals``, so ``params_to_fit`` mainly matters as a record of
intent; the bounds are what you need to get right for a new source.

Model-based outlier clipping
=================================

The most robust clipper in the package rejects points against the fitted model rather
than against a spline, and so works on faint and sparsely-sampled light curves:

.. code-block:: python

   Nsigma = phot.clip_outliers_sncosmo(sigma_minimum=3, plot=False)

Each iteration fits, computes the standard deviation of the flux residuals, and sets
the clip threshold from `Chauvenet's criterion
<https://en.wikipedia.org/wiki/Chauvenet%27s_criterion>`_ via
:func:`~atlast.api_ATLAS.chauvenet_sigma_level` -- the sigma level at which you would
expect fewer than half an event across the sample. This scales with the number of
points, so a 500-epoch light curve is clipped at a higher threshold than a 20-epoch
one, rather than at a fixed 3 sigma.

``sigma_minimum`` (default 3) is a floor: if the criterion suggests a threshold below
it, no clipping is done and the method returns the suggested level. Iteration stops
when no further points are removed or after ``maxiter`` (default 10) rounds. The
method returns ``None`` if any fit in the loop failed, so check the return value
before using the result.

``sigma_factor`` scales the computed threshold; leave it at 1.0 unless you have a
specific reason.

A full pass
=================================

.. code-block:: python

   from atlast import AtlasPhotometry, init_sncosmo_filters

   init_sncosmo_filters()

   phot = AtlasPhotometry('phot.txt', objname=name)
   phot.cut(fluxerr_max=3000, chi2dof_max=5)
   phot_binned = phot.group_by_mjd(delta_mjd=0.5)

   # first pass: get a peak time
   phot_binned.set_sncosmo_redshift_bounds((0.01, 0.15))
   ok = phot_binned.fit_lc(preset_vals={'z': redshift}, plot=False)
   pkmjd = phot_binned.sncosmo_result.parameters[1]

   # baseline correction now that the transient window is known
   phot_binned.subtract_flux_offset(zf_t3=phot_binned.get_zero_levels(pkmjd=pkmjd))

   # second pass, with model-based clipping
   phot_binned.clip_outliers_sncosmo(
       sigma_minimum=3,
       preset_vals={'z': redshift},
       plot=False,
   )

Keyword arguments given to ``clip_outliers_sncosmo`` are forwarded to both ``fit_lc``
and ``cut``, which is how the redshift stays fixed across every refit.
