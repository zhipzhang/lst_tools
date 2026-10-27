The analysis pipeline: ``init-lstana``
======================================

Edit ``config/config.toml`` (the source, zenith binning, quality cuts, and
one section per data level), then run:

.. code-block:: bash

   init-lstana config/config.toml --output /path/to/analysis

Every data level has two possible origins, selected by a ``source`` key:

- ``source = "link"`` — place the files the standard pipeline already
  produced on La Palma into the workspace zenith-bin tree.
- ``source = "build"`` — produce the level ourselves.

Stages
------

1. **Select runs** — nightly DataCheck statistics filtered by the
   ``[data_filter]`` cuts (and the ``[source]`` position), grouped into the
   ``[zenith_binning]`` bins. The selected tables and advanced-cut
   diagnostics are saved under ``data_check/``.
2. **DL1** (``[dl1]``) — the external input; linking is the only supported
   mode today (``dl1/<zd_bin>/``).
3. **DL2** (``[dl2]``) — either link the official DL2, or build our own by
   reconstructing the linked DL1 with the RF models of
   ``[dl2.build] rf_directory``. The RF directory is a *template*:
   ``$NSB_LEVEL`` is substituted per run at build time with the run's NSB
   level, read from its DL1 file; the declination line stays literal.
4. **DL3** (``[dl3]``) — either link existing DL3 products (newest version,
   optionally narrowed by ``[dl3.link] irf_type``), or build them: IRFs are
   generated from the MC test dataset matching each DL2 file's provenance
   (``[dl3.build.irf]``), then every run is reduced to a DL3 FITS
   observation, interpolating the IRFs to the run's pointing.

The MC campaign is selected by ``[dl3.build.irf] mc_dl2_directory``. This is
a directory template: ``$NSB_LEVEL`` is replaced independently for every DL2
file using its provenance. Relative paths resolve below
``/fefs/aswg/data/mc/DL2/AllSky``; absolute paths are also accepted. The
finder recursively reads the campaign's ``TestingDataset`` directories and
selects the matching declination line. With ``point_like = true`` it uses
``Gamma`` and writes point-like IRFs; otherwise it uses ``GammaDiffuse`` and
writes full-enclosure IRFs.

The event selection for built DL3 comes from ``[dl3.build.irf]``: the
quality filters (``intensity_cut``, ``leakage_cut``) are passed to the
reduction and the gammaness cut (``gh_cut`` or ``gh_efficiency``) is carried
inside the IRFs, so data and IRFs always agree.

Workspace layout
----------------

.. code-block:: text

   analysis/
   ├── data_check/
   ├── dl1/<zd_bin>/     # linked input (stage 2)
   ├── dl2/<zd_bin>/     # linked or built (stage 3)
   ├── irfs/             # built during the DL3 build
   │   └── nsb_<nsb>/dec_<line>/<irf_type>/
   │       └── theta_<zenith>_az_<azimuth>.irf.fits.gz
   └── dl3/<zd_bin>/     # linked or built (stage 4)

Here ``<irf_type>`` is explicitly ``point`` or ``full-enclosure``. The DL3
reduction reads the same type-specific directory, so the two kinds of IRFs
cannot overwrite each other or be mixed in one interpolation set.

Zenith bins include their lower edge and exclude their upper edge; the last
bin also includes its upper edge. Reruns are incremental: existing links are
kept, existing files are reused, and conflicting files are never replaced.
Each level's build mode has an ``overwrite`` option to force rebuilding.

Configuration
-------------

See the comments in ``config/config.toml`` for every option. The essential
shape is one section per level, with the mode-specific options in
``[dlX.link]`` / ``[dlX.build]``:

.. code-block:: toml

   [dl2]
   enabled = true
   source = "build"       # "build" | "link"

   [dl2.build]
   rf_directory = "/fefs/aswg/data/models/AllSky/nsb_tuning_$NSB_LEVEL/dec_347"
   overwrite = false

   [dl3]
   enabled = true
   source = "build"

   [dl3.build]
   interp_method = "linear"
   use_nearest_irf_node = false
   overwrite = false

   [dl3.build.irf]
   mc_dl2_directory = "20250212_v0.10.17_allsky_interp_dl2_irfs_nsb_$NSB_LEVEL"
   intensity_cut = 80
   leakage_cut = 1.0
   gh_efficiency = 0.7    # or gh_cut = 0.7 for a fixed threshold
   point_like = false     # true: point/Gamma; false: full-enclosure/GammaDiffuse

Options of the inactive mode are ignored with a warning, so both blocks may
be kept while switching ``source`` during testing. A level with
``enabled = false`` (or no section at all) is skipped.
