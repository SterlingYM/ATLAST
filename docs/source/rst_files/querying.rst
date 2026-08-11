=================================
Querying ATLAS forced photometry
=================================

:class:`~atlast.api_ATLAS.QueryATLAS` is a thin client for the ATLAS forced
photometry server at https://fallingstar-data.com/forcedphot. Requests are queued
server-side: you POST a position and MJD range, get back a *task URL*, and then poll
that URL until the result is ready.

Getting an API token
=================================

Register for an account on the `ATLAS forced photometry site
<https://fallingstar-data.com/forcedphot>`_, then retrieve your token. The token is
passed straight into the ``Authorization`` header:

.. code-block:: python

   from atlast import QueryATLAS

   api = QueryATLAS(api_token='your_api_token_here')

Do not hard-code the token in a script that will be committed. Read it from the
environment instead:

.. code-block:: python

   import os
   api = QueryATLAS(api_token=os.environ['ATLAS_TOKEN'])

The server URL can be overridden with the ``url`` argument, but the default is
correct for the public service.

Requesting photometry
=================================

:meth:`~atlast.api_ATLAS.QueryATLAS.request_phot` passes all of its keyword arguments
straight through to the server as form data, so the parameter names are the ones the
ATLAS API itself defines:

.. code-block:: python

   task_url = api.request_phot(
       ra=radeg,
       dec=decdeg,
       mjd_min=58000,
       mjd_max=59000,
       use_reduced=False,   # difference-image photometry (default); True = reduced
       send_email=False,
   )

It returns the task URL on success (HTTP 201). If the server responds with anything
other than 201 or a throttle, a ``ValueError`` carrying the status code and response
body is raised.

.. note::

   The server rate-limits requests. ``request_phot`` retries up to ``maxiter=10``
   times when it sees an HTTP 429; if it exhausts those attempts it raises
   ``ValueError('Maxiter reached...')``. When looping over many targets, add your own
   pause between calls rather than relying on the retry loop alone.

Checking status and downloading
=================================

The task does not complete immediately. Poll it:

.. code-block:: python

   status = api.get_task_status(task_url)   # full JSON from the server
   urls   = api.get_data_urls(task_url)     # {'result_url': ..., 'result_imagezip_url': ...}

``result_url`` is ``None`` while the job is still running. Once it is populated:

.. code-block:: python

   # fetch the photometry as text, without writing to disk
   text, phot_url = api.get_phot_text(task_url)

   # or save it into a folder, using the server-side filename
   filename = api.save_phot(task_url, folder='phot_output')

:meth:`~atlast.api_ATLAS.QueryATLAS.save_phot` returns the path it wrote. If a file
of that name already exists it returns immediately without re-downloading, which
makes it safe to re-run over a target list. It also strips the leading ``###`` from
the header line so that the file parses cleanly downstream.

Calling ``get_phot_text`` before the job finishes raises
``ValueError('No photometry available yet')`` -- catch that and retry, or poll
``get_data_urls`` first.

A typical loop over a target list:

.. code-block:: python

   import time

   task_urls = {}
   for name, (ra, dec) in targets.items():
       task_urls[name] = api.request_phot(ra=ra, dec=dec, mjd_min=57000, mjd_max=60000)
       time.sleep(1)

   files = {}
   for name, task_url in task_urls.items():
       while api.get_data_urls(task_url)['result_url'] is None:
           time.sleep(30)
       files[name] = api.save_phot(task_url, folder='phot_output')

Difference images
=================================

Difference image cutouts have to be requested explicitly for an existing task, and
are then served as a zip archive:

.. code-block:: python

   zip_url = api.request_images(task_url)     # None if not ready yet
   folder  = api.save_images(task_url, path='./diff_images/')

:meth:`~atlast.api_ATLAS.QueryATLAS.save_images` streams the archive with a progress
bar, unzips it into ``path``, deletes the zip, and returns the directory it extracted
into. It raises ``ValueError('Images are not ready yet')`` if the archive URL is not
yet available -- call ``request_images`` first and give the server time to work.

What the photometry file contains
=================================

The saved file is whitespace-separated with a one-line header. The columns ATLAST
relies on are:

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Column
     - Meaning
   * - ``MJD``
     - Modified Julian Date of the exposure.
   * - ``m``, ``dm``
     - Magnitude and its uncertainty.
   * - ``uJy``, ``duJy``
     - Difference flux and uncertainty in microJansky. This is what ATLAST converts
       into the SNANA flux scale; it is signed, and negative values are meaningful.
   * - ``F``
     - Filter, ``o`` (orange) or ``c`` (cyan).
   * - ``chi/N``
     - Reduced chi-square of the PSF fit. Useful as a quality cut.
   * - ``Obs``
     - Observation identifier. Its first three characters give the ``sitecam``
       (e.g. ``01a``), which ATLAST uses to look up which chip took the exposure.
   * - ``x``, ``y``
     - Pixel position on the chip, needed by
       :meth:`~atlast.api_ATLAS.AtlasPhotometry.apply_flux_corr` for
       position-dependent zero-point corrections.

Once you have the file, continue with :doc:`photometry`.
