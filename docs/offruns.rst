Catalog-separated off runs
==========================

Set the following in the pipeline config and ``init-lstana`` stage 1 keeps
only runs pointed away from known TeV sources:

.. code-block:: toml

   [data_filter]
   away_source = true
   min_distance = 3.0      # deg from point-like sources
   extension_factor = 2.5  # extended sources add this factor times their size

The catalog-separation cut uses the HESS, LHAASO, HAWC and Gamma-Cat
catalogs, and the ``[data_filter.cuts]`` target-angle range is disabled:
an off run cannot point at the target and away from it at once.

The standalone ``prepare-offruns`` command remains available for a separate
off-run workspace:

.. code-block:: bash

   prepare-offruns config/config.toml \
     --output /path/to/offruns \
     --with-dl1 \
     --with-dl2 \
     --min-separation 3 \
     --extension-factor 2.5

It applies the configured basic and advanced quality cuts with the
target-angle range disabled, then rejects pointings near the same four
catalogs. It always writes ``data_check/DL1_datacheck_offruns.h5``;
``--with-dl1`` (``--with-dl`` is an alias) and ``--with-dl2`` additionally
create idempotent data links under ``dl1/`` and ``dl2/``. When neither
option is given, no data links are created.
