=================================
SNANA input and output
=================================

ATLAST works natively in the SNANA ``FLUXCAL`` scale (zero point 27.5), so exporting
to SNANA format is a formatting step rather than a conversion. The same module also
reads SNANA data back in, which lets real and simulated light curves be pushed through
identical analysis code.

Exporting a light curve
=================================

.. code-block:: python

   headers = {
       'SURVEY': 'ATLAS',
       'SNID': objname,
       'RA': f'{ra:.5f}',
       'DECL': f'{dec:.5f}',
       'RA_HOST': f'{ra_host:.5f}',
       'DEC_HOST': f'{dec_host:.5f}',
       'HOSTGAL_LOGMASS': f'{logmass:.2f} +- {logmass_err:.2f}',
       'REDSHIFT_HELIO': f'{redshift:.6f} +- 0.0001',
       'FILTERS': 'co',
   }
   phot.to_SNANA('output_file.txt', header=headers)

Only epochs passing the current cuts are written. The output looks like:

.. code-block:: text

   SURVEY: ATLAS
   SNID: 2021abc

   # ========================================
   NOBS: 2855
   NVAR: 5
   VARLIST: MJD FLT FIELD FLUXCAL FLUXCALERR
   OBS:   57804.222886 ATLAS-c4 -1 -550.85 963.98
   OBS:   57804.229474 ATLAS-c4 -1 330.51 605.93

Every key in ``header`` is written verbatim as ``KEY: VALUE``, so any metadata SNANA
expects can be passed through. ``FIELD`` is always ``-1``, and fluxes are rounded to
two decimals.

Per-chip filter names
=================================

Note the ``FLT`` values above: ``ATLAS-c4``, not ``c``. By default the exported filter
name is ``'ATLAS-' + filter + chip_id``, where the chip ID is the row index into
:data:`~atlast.api_ATLAS.df_chip` (see :doc:`reference_data`). Epochs whose chip could
not be identified get ``9999``.

This is deliberate: it lets a downstream SNANA fit solve for a separate zero point per
camera-and-chip combination, which is the natural way to absorb the chip-to-chip
calibration differences that :doc:`corrections` addresses at the map level. If you do
not want that, collapse the names with ``filtername_map``:

.. code-block:: python

   # merge every chip back into two bands
   fmap = {f'ATLAS-{f}{i}': f for f in 'co' for i in range(len(phot.df_chip))}
   phot.to_SNANA('output_file.txt', header=headers, filtername_map=fmap)

``filtername_map`` is applied after the per-chip names are generated, so its keys must
be the generated names. Names not present in the map are left unchanged.

To write a different set of columns, pass ``columns_to_output``; the default is
``['MJD', 'FLT', 'FIELD', 'FLUXCAL', 'FLUXCALERR']``. The columns must already exist
in ``phot.df_phot``.

.. warning::

   ``to_SNANA`` raises ``AttributeError: 'AtlasBinnedPhotometry' object has no
   attribute '_chipid'`` when called on a binned light curve, because binning does not
   carry chip IDs across. Export from the unbinned object, or attach the attribute
   yourself before exporting:

   .. code-block:: python

      import numpy as np
      phot_binned._chipid = np.full(len(phot_binned._mjd), np.nan)   # -> FLT 'ATLAS-c9999'

Reading SNANA data
=================================

:class:`~atlast.api_SNANA.SNANAPhotometry` subclasses
:class:`~atlast.api_ATLAS.AtlasPhotometry`, so everything in :doc:`photometry` and
:doc:`fitting` -- cuts, plotting, binning, SNCosmo fitting -- applies to it too:

.. code-block:: python

   from atlast import SNANAPhotometry

   sn = SNANAPhotometry('output_file.txt', objname='2021abc')
   sn.plot_lc()

   print(sn.metadata['REDSHIFT_HELIO'])

Flux is taken from ``FLUXCAL`` / ``FLUXCALERR`` directly, with no rescaling, so the
file's zero point must match the ``zp`` argument (default 27.5). Header lines are
parsed into ``sn.metadata``.

Filters come from the ``FLT`` column, or ``BAND`` if ``FLT`` is absent; anything before
a ``/`` is stripped. Since SNCosmo needs the bare ``o``/``c`` names, map per-chip
filter names back when reading a file written with the default naming:

.. code-block:: python

   fmap = {f'ATLAS-{f}{i}': f for f in 'co' for i in range(9)}
   sn = SNANAPhotometry('output_file.txt', filtername_map=fmap)

:func:`~atlast.api_SNANA.read_SNANA_data` is available on its own if you only want the
DataFrame; the metadata lands in ``df.attrs``.

.. note::

   SNANA files carry no magnitudes or PSF chi-square, so ``mag``/``magerr`` are absent
   and ``_chi2dof`` is filled with zeros. Cuts on those quantities are therefore not
   meaningful on ``SNANAPhotometry``, and ``chi2dof_max`` will remove nothing.

Simulations and multi-object files
==================================

SNANA distributes simulations and compiled datasets as a HEAD/PHOT FITS pair. Both
classes open the pair and slice out individual objects using the ``PTROBS_MIN`` /
``PTROBS_MAX`` pointers in the header table:

.. code-block:: python

   from atlast import SNANASim, SNANAData

   sim = SNANASim('SIM_HEAD.FITS', 'SIM_PHOT.FITS')
   lc = sim.get(1234)             # SNID as an integer
   lc.plot_lc()

   data = SNANAData('DATA_HEAD.FITS', 'DATA_PHOT.FITS')
   print(data.unique_ids)         # available SNIDs, as strings
   lc = data.get(data.unique_ids[0])

The two classes are the same except for the SNID type: ``SNANASim.get`` casts SNIDs to
``int`` (simulations number their objects), while ``SNANAData.get`` matches on the
stripped string. Both return a :class:`~atlast.api_SNANA.SNANAPhotometry`, take the
band from the last character of ``BAND``, and drop duplicated MJDs keeping the last.
``objname`` defaults to ``'SIM <simid>'``.

The header table is available as ``.df_header`` and the full photometry table as
``.df_photometry``, for anything the accessor does not cover -- e.g. selecting objects
by simulated parameters before pulling their light curves.
