Catalog-separated off runs: ``prepare-offruns``
===============================================

.. code-block:: bash

   prepare-offruns config/config.toml \
     --output /path/to/offruns \
     --with-dl1 \
     --with-dl2 \
     --min-separation 3 \
     --extension-factor 2.5

Applies the configured basic and advanced quality cuts with the target-angle
range disabled, then rejects pointings near HESS, LHAASO, and HAWC catalog
sources. Always writes ``data_check/DL1_datacheck_offruns.h5``;
``--with-dl1`` (``--with-dl`` is an alias) and ``--with-dl2`` additionally
create idempotent data links under ``dl1/`` and ``dl2/``. When neither
option is given, no data links are created.
