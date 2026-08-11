=================================
Systematics and corrections
=================================

ATLAS difference-image forced photometry carries several systematics that matter for
transient science. ATLAST provides a correction for each. They are largely
independent, and a typical pipeline applies them in the order below.

.. important::

   **Obtaining the calibration data.**

   Two of the corrections on this page — the position-dependent zero-point maps used
   by :meth:`~atlast.api_ATLAS.AtlasPhotometry.apply_flux_corr` and the error fudge
   function used by :meth:`~atlast.api_ATLAS.AtlasPhotometry.apply_fudge_func` —
   require calibration products that are **not distributed with this package**.

   For how these products are derived and validated, see `Marlin et al.
   <https://arxiv.org/abs/2512.21903>`_ (arXiv:2512.21903).

   The calibration data itself is reserved for the TITAN team until the science
   analyses are complete. To request access, contact **Y. S. Murakami** or
   **Eli Marlin**.

   Everything else in ATLAST — cuts, binning, nightly pull, zero levels, fitting and
   SNANA export — works without these files.

.. contents:: On this page
   :local:
   :depth: 1


Template changes
=================================

ATLAS reference images have been rebuilt twice. Because the reference is subtracted
from every science frame, a light curve that spans one of these dates can show a step
in its baseline flux -- an apparent jump that is entirely instrumental.

The two dates are hard-coded on the object as ``t12`` and ``t23``:

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Attribute
     - MJD
     - Meaning
   * - ``t12``
     - 58417
     - Template generation 1 → 2
   * - ``t23``
     - 58882
     - Template generation 2 → 3

At load time :meth:`~atlast.api_ATLAS.AtlasPhotometry.check_template_change` compares
these to the data's MJD range and sets ``template_change_12`` / ``template_change_23``.
By default it also prints a warning; pass ``warn_template_changes=False`` to the
constructor to silence it when looping over many objects.

There are two ways to handle a light curve that crosses a template change.

**Drop the data on the far side.** If the change falls well away from the transient,
the simplest fix is to keep only the side that matters:

.. code-block:: python

   phot.cut_data_at_template_change(pkmjd=pkmjd, retain_phase=[-20, 50])

This requires the peak MJD, so run it after a first fit (see :doc:`fitting`) or with
an externally known ``t0``.

.. note::

   As implemented, a template change earlier than ``pkmjd - retain_phase[0]`` causes
   all data *before* the change to be cut, and a change later than
   ``pkmjd + retain_phase[1]`` causes all data *after* it to be cut. With the default
   ``retain_phase=[-20, 50]`` the first threshold evaluates to ``pkmjd + 20``.

**Correct each era separately.** If you need the full baseline -- for example to
measure a pre-explosion limit -- fit and subtract the zero level in each template era
independently, as described next.


Zero levels (ill-subtraction)
=================================

If the reference image contains flux from the transient itself, or the host
subtraction is imperfect, the whole light curve sits on a non-zero baseline. Since we
know the transient contributes nothing long before and long after peak, that baseline
can be measured directly from the data.

.. code-block:: python

   zero_levels = phot.get_zero_levels(
       pkmjd=pkmjd,
       filters=['c', 'o'],
       phase_before=-40,   # epochs earlier than pkmjd-40 are assumed source-free
       phase_after=200,    # as are epochs later than pkmjd+200
   )
   # {'c': [flux, flux_err], 'o': [flux, flux_err]}

The epochs outside the transient window are sigma-clipped and combined with an
inverse-variance weighted mean. If fewer than ``min_Ndata`` (default 5) usable epochs
survive in a filter, that filter's zero level is returned as ``[0, 0]`` -- i.e. no
correction rather than a correction from noise.

.. warning::

   The weighting is ``1 / err**2``, so any surviving epoch with an error of exactly
   zero makes the weighted mean ``nan`` for that filter. This happens on real files.
   Check the result before applying it, and cut on ``fluxerr_min`` first if needed:

   .. code-block:: python

      phot.cut(s=phot.cuts, fluxerr_min=1e-6, fluxerr_max=3000)
      zero_levels = phot.get_zero_levels(pkmjd=pkmjd)
      assert all(np.isfinite(v[0]) for v in zero_levels.values())

Subtract the measured offsets with
:meth:`~atlast.api_ATLAS.AtlasPhotometry.subtract_flux_offset`, which takes one
dictionary per template era and adds the offset uncertainty to the flux errors in
quadrature:

.. code-block:: python

   phot.subtract_flux_offset(zf_t1=zl_early, zf_t2=zl_mid, zf_t3=zl_late)

The three arguments correspond to MJD < ``t12``, ``t12`` ≤ MJD < ``t23``, and
MJD ≥ ``t23``. Pass ``None`` for eras you do not want to touch. For a light curve
that does not cross a template change, a single dictionary in the right slot is
enough:

.. code-block:: python

   phot.subtract_flux_offset(zf_t3=phot.get_zero_levels(pkmjd=pkmjd))

To measure a separate zero level per era, cut to that era first, call
``get_zero_levels``, then restore the cuts:

.. code-block:: python

   cuts = phot.cuts.copy()

   phot.cut(s=cuts, mjd_max=phot.t12)
   zl_early = phot.get_zero_levels(pkmjd=pkmjd)

   phot.cut(s=cuts, mjd_min=phot.t12, mjd_max=phot.t23)
   zl_mid = phot.get_zero_levels(pkmjd=pkmjd)

   phot.cut(s=cuts)
   phot.subtract_flux_offset(zf_t1=zl_early, zf_t2=zl_mid)

``subtract_flux_offset`` caches the uncorrected arrays in ``_flux_raw`` /
``_flux_err_raw`` and restores from them on every call, so it is safe to re-run with
different offsets without compounding the correction.

.. warning::

   :meth:`~atlast.api_ATLAS.AtlasBinnedPhotometry.correct_zero_levels` is a
   convenience wrapper on binned light curves that is currently broken: it unpacks
   the single dictionary returned by ``get_zero_levels`` into two variables and then
   raises ``TypeError``. Use ``get_zero_levels`` together with
   ``subtract_flux_offset`` as shown above.


Position-dependent zero points
=================================

The photometric zero point varies across an ATLAS chip. Given a correction map you
can look up an offset for each epoch from its ``(x, y)`` pixel position and its chip:

.. code-block:: python

   phot.apply_flux_corr('zp_correction_maps.npy')

.. note::

   The correction maps are TITAN-team calibration data and are not shipped with
   ATLAST. See `Marlin et al. <https://arxiv.org/abs/2512.21903>`_ for how they are
   built, and contact Y. S. Murakami or Eli Marlin to request access.

The file is a numpy array that unpacks into two elements, ``(corr_cyan, corr_orange)``.
Each is indexed by chip ID -- the row index into :data:`~atlast.api_ATLAS.df_chip`,
see :doc:`reference_data` -- and each entry is a ``(3, N, N)`` array of
``[xx, yy, zz]`` giving the zero-point offset ``zz`` on a grid of chip positions.
Lookup is nearest-neighbour via :class:`~atlast.api_ATLAS.GridLookup`, which builds a
:class:`scipy.spatial.cKDTree` over the grid points.

The sign convention is the one place this is easy to get wrong:

.. warning::

   The map must contain the zero-point offset **to be applied** to the data, not the
   one to be removed. A stored value of ``+0.1`` *reduces* the flux by
   :math:`10^{-0.4 \times 0.1} \approx 0.91`. The method prints this reminder every
   time; pass ``print_message=False`` to suppress it.

Both flux and flux error are multiplied by the same factor, so signal-to-noise is
unchanged. The applied values are kept for inspection in ``_zp_offset`` (magnitudes)
and ``_flux_factor`` (multiplicative). Epochs whose chip or filter has no entry in the
map are left untouched.

Because it modifies ``_flux`` in place with no cached original, calling
``apply_flux_corr`` twice applies the correction twice. Re-load the file if you need
to start over.


Error inflation
=================================

ATLAS flux errors are known to be underestimated, and the size of the underestimate
depends on the flux level. Two tools address this.

**A flat fractional term**, added in quadrature:

.. code-block:: python

   phot.add_error_quadrature(percent_flux=5.0)

This adds 5% of each epoch's flux to its error in quadrature.

**A flux-dependent fudge function**, a multiplicative correction calibrated
externally:

.. code-block:: python

   phot.apply_fudge_func('fudge_function.npy')

.. note::

   Like the zero-point maps, the fudge function is TITAN-team calibration data and is
   not shipped with ATLAST. Its derivation is described in `Marlin et al.
   <https://arxiv.org/abs/2512.21903>`_; contact Y. S. Murakami or Eli Marlin to
   request access. ``add_error_quadrature`` above needs no external file, but a flat
   percentage is a crude stand-in — it does not capture the flux dependence, so do not
   treat it as a substitute in published work.

The file unpacks into ``(flux_axis, fudge_vals)``: a flux grid and the factor by which
the error at that flux should be multiplied. Values are linearly interpolated, and
clamped to the endpoint values outside the tabulated range rather than extrapolated.
The factor actually applied to each epoch is stored in ``_fudge_factor``.

.. warning::

   Neither method caches the original errors, so both compound if called more than
   once. Apply each exactly once per loaded light curve.


Recommended order
=================================

.. code-block:: python

   phot = AtlasPhotometry('phot.txt', objname=name)

   # 1. quality cuts, on the unbinned data
   phot.cut(fluxerr_max=3000, chi2dof_max=5)

   # 2. chip-level corrections, which need x/y and chip ID
   phot.apply_flux_corr('zp_correction_maps.npy')

   # 3. error inflation
   phot.apply_fudge_func('fudge_function.npy')

   # 4. bin to one point per night per filter
   phot_binned = phot.group_by_mjd(delta_mjd=0.5)

   # 5. baseline correction, once the peak time is known
   phot_binned.subtract_flux_offset(zf_t3=phot_binned.get_zero_levels(pkmjd=pkmjd))

Steps 2 and 3 must precede binning, since the binned object no longer carries chip
IDs or pixel positions. Step 5 needs a peak time, so in practice it follows a first
pass through :doc:`fitting`.
