=================================
Querying TNS
=================================

:class:`~atlast.api_TNS.QueryTNS` is a small client for the `Transient Name Server
<https://www.wis-tns.org>`_ API, used to look up coordinates, discovery dates,
classifications and spectra for named transients -- typically to build the target list
that is then fed to :doc:`querying`.

.. note::

   Unlike the other classes, ``QueryTNS`` is not re-exported at package level. Import
   it from its module:

   .. code-block:: python

      from atlast.api_TNS import QueryTNS

Credentials
=================================

TNS requires a *bot* account. Create one under your TNS user, then supply its numeric
ID, name and API key:

.. code-block:: python

   tns = QueryTNS(
       tns_id='12345',
       bot_name='my_bot',
       api_key='your_api_key',
   )

All three go into the request: ``tns_id`` and ``bot_name`` build the ``User-Agent``
marker TNS uses to identify the bot, and ``api_key`` is sent with each query. Requests
without a correctly-formed marker are rejected.

Searching and retrieving
=================================

Both search and object endpoints accept whatever parameters the TNS API defines; they
are forwarded as-is:

.. code-block:: python

   # cone search
   results = tns.search_objects(ra='10.684', dec='41.269', radius='5', units='arcsec')

   # search by internal survey name
   results = tns.search_objects(internal_name='ATLAS21abc')

   # full record for one object, including spectra
   obj = tns.get_object(objname='2021abc', spectra=1)
   print(obj['reply']['radeg'], obj['reply']['decdeg'], obj['reply']['redshift'])

Both return the ``data`` field of the JSON response. If TNS reports a problem, the
message is printed and the raw response returned instead, so check the shape of what
you get back before indexing into it. HTTP errors are printed from a lookup table and
the method returns ``None``.

Bulk lookups
=================================

:meth:`~atlast.api_TNS.QueryTNS.get_objects` wraps ``get_object`` for a list of names
and checkpoints to a JSON file as it goes:

.. code-block:: python

   tns.get_objects(['2021abc', '2021def', ...], jsonfile='tns_cache.json')
   replies = tns.get_objects_replies      # {objname: reply}

The cache is what makes this usable over a few thousand objects: on startup it loads
``jsonfile`` and skips any object already present, so an interrupted run can simply be
restarted. Results are flushed every ``save_every`` (default 100) objects and once at
the end.

Objects that raise are recorded as the string ``'failed'`` rather than aborting the
loop. Those *are* retried on a later run, since the skip condition excludes them.
Filter them out before use:

.. code-block:: python

   good = {k: v for k, v in tns.get_objects_replies.items() if v != 'failed'}

Downloading files
=================================

Spectra and photometry files referenced in an object record are fetched by full URL:

.. code-block:: python

   tns.get_file('https://www.wis-tns.org/system/files/.../spectrum.txt',
                savefile='spectrum.txt')

The response is written as text, so this is suitable for ASCII spectra and photometry
files rather than FITS.

Building a target list
=================================

Putting it together -- resolve names to coordinates on TNS, then queue ATLAS forced
photometry for each:

.. code-block:: python

   from atlast import QueryATLAS
   from atlast.api_TNS import QueryTNS

   tns = QueryTNS(tns_id=tns_id, bot_name=bot_name, api_key=tns_key)
   atlas = QueryATLAS(api_token=atlas_token)

   tns.get_objects(names, jsonfile='tns_cache.json')

   task_urls = {}
   for name, reply in tns.get_objects_replies.items():
       if reply == 'failed':
           continue
       task_urls[name] = atlas.request_phot(
           ra=reply['radeg'],
           dec=reply['decdeg'],
           mjd_min=float(reply['discoverydate_mjd']) - 100,
           mjd_max=float(reply['discoverydate_mjd']) + 300,
       )
