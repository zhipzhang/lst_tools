The analysis pipeline: ``init-lstana``
======================================

Edit ``config/config.toml`` (the source, quality cuts, and one section per
data level), then run:

.. code-block:: bash

   init-lstana config/config.toml --output /path/to/analysis

Every data level has two possible origins, selected by a ``source`` key:

- ``source = "link"`` — place files the standard pipeline already produced
  on La Palma directly into the corresponding data-level directory.
- ``source = "build"`` — produce the level ourselves.

Stages
------

1. **Select runs** — nightly DataCheck statistics filtered by the
   ``[data_filter]`` cuts (and the ``[source]`` position). The selected runs
   are saved under ``data_check/`` and reused: when that file exists, it is
   loaded and the nightly files are not read again. ``--refresh-selection``
   recomputes the selection after changing the cuts or source. With
   ``[data_filter] away_source = true`` the target-angle cut is replaced by
   a catalog-separation cut (``min_distance``, ``extension_factor``), so the
   same pass selects off runs.
2. **DL1** (``[dl1]``) — the external input; linking is the only supported
   mode today (``dl1/``).
3. **DL2** (``[dl2]``) — either link the official DL2, or build our own by
   reconstructing the linked DL1 with the RF models of
   ``[dl2.build] rf_directory``. The RF directory is a *template*:
   ``$NSB_LEVEL`` is substituted per run at build time with the run's NSB
   level, read from its DL1 file; the declination line stays literal.
4. **DL3** (``[dl3]``) — either link existing DL3 products (newest version,
   optionally narrowed by ``[dl3.link] irf_type``), or build them: IRFs are
   generated from the MC test dataset matching each DL2 file's provenance
   (``[dl3.build]``), then every run is reduced to a DL3 FITS
   observation, interpolating the IRFs to the run's pointing.

The MC campaign is selected by ``[dl3.build] mc_dl2_directory``. This is
a directory template: ``$NSB_LEVEL`` is replaced independently for every DL2
file using its provenance. Relative paths resolve below
``/fefs/aswg/data/mc/DL2/AllSky``; absolute paths are also accepted. The
finder recursively reads the campaign's ``TestingDataset`` directories and
selects the matching declination line. With ``point_like = true`` it uses
``Gamma`` and writes point-like IRFs; otherwise it uses ``GammaDiffuse`` and
writes full-enclosure IRFs.

Each ``[[dl3.build.reduction]]`` table defines one named DL3 selection. The
quality filters (``intensity_cut``, ``leakage_cut``) are passed to the
reduction and the gammaness cut (``gh_cut`` or ``gh_efficiency``) is carried
inside the IRFs, so data and IRFs always agree. Shared interpolation and MC
options stay under ``[dl3.build]``.

DL3 builds follow ``RunSelection``. Each selected run must have exactly one
DL2 file directly below ``dl2/``. Missing inputs stop the build with a list
of runs; ambiguous matches are rejected. Unselected files are ignored, and
an empty selection performs no work and submits no job. The selected file
paths and reduction definitions are saved with a SLURM build, and the worker
checks that the files still exist before starting.

The build prepares IRF interpolation sets in parallel, then runs named
reductions in parallel after all IRFs are ready. Workers are bounded by the
local CPU count, or the job's CPU allocation under SLURM. With SLURM enabled,
the same build runs inside one job; ``[slurm.dl3]`` sets its CPU, memory, and
walltime allocation.

Workspace layout
----------------

.. code-block:: text

   analysis/
   ├── data_check/
   ├── dl1/              # linked input (stage 2)
   ├── dl2/              # linked or built (stage 3)
   ├── irfs/             # built during the DL3 build
   │   └── <reduction_name>/nsb_<nsb>/dec_<line>/<irf_type>/<cut_label>/
   │       └── theta_<zenith>_az_<azimuth>.irf.fits.gz
   └── dl3/<reduction_name>/  # built reductions (stage 4)

The unique ``<reduction_name>`` keeps DL3 products and IRFs separate, even
when reductions differ only in their quality cuts. Here ``<irf_type>`` is
explicitly ``point`` or ``full-enclosure``, and ``<cut_label>`` names the
gammaness selection, ``gheff0.7`` for an energy-dependent efficiency or
``ghcut0.7`` for a fixed cut. Each reduction reads the IRF directory matching
its own configuration, so incompatible IRFs never share an interpolation set.

Reruns are incremental: existing links are kept, existing files are reused,
and conflicting files are never replaced. Each level's build mode has an
``overwrite`` option to force rebuilding. A reduction name identifies its
configuration: after changing its cuts, use a new name or enable
``overwrite`` to avoid reusing the previous DL3 files. DL4 generation should
select one named reduction at a time.

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
   mc_dl2_directory = "20250212_v0.10.17_allsky_interp_dl2_irfs_nsb_$NSB_LEVEL"

   [[dl3.build.reduction]]
   name = "standard"
   intensity_cut = 80
   leakage_cut = 1.0
   gh_efficiency = 0.7    # or gh_cut = 0.7 for a fixed threshold
   point_like = false     # true: point/Gamma; false: full-enclosure/GammaDiffuse

   [[dl3.build.reduction]]
   name = "tight"
   intensity_cut = 150
   leakage_cut = 0.8
   gh_efficiency = 0.9
   point_like = false

   [slurm.dl3]
   cpus_per_task = 8
   mem = "40GB"
   time = "04:00:00"

Options of the inactive mode are ignored with a warning, so both blocks may
be kept while switching ``source`` during testing. A level with
``enabled = false`` (or no section at all) is skipped.
