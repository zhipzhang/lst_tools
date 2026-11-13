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
   recomputes the selection after changing the cuts or source.
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

Catalog-separated off runs (``away_source``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To select off runs directly in stage 1 without a separate workflow, enable
``away_source`` under ``[data_filter]``:

.. code-block:: toml

   [data_filter]
   away_source = true
   min_distance = 3.0      # deg; minimum pointing distance from point-like sources
   extension_factor = 2.5  # extended sources add this factor times their size

When ``away_source = true``, the ``[data_filter.cuts]`` target-angle range
(``min_angle_to_source``, ``max_angle_to_source``) is disabled and pointings
near known TeV sources in the HESS, LHAASO, HAWC, and Gamma-Cat catalogs are
rejected instead. All subsequent stages (DL1 linking, DL2 reconstruction, and
DL3 reduction) then operate on the selected off runs in the same way.

DL3 build and SLURM execution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The MC campaign is selected by ``[dl3.build] mc_dl2_directory``. This is
a directory template: ``$NSB_LEVEL`` is replaced independently for every DL2
file using its provenance. Relative paths resolve below
``/fefs/aswg/data/mc/DL2/AllSky``; absolute paths are also accepted. The
finder recursively reads the campaign's ``TestingDataset`` directories and
selects the matching declination line. With ``point_like = true`` it uses
``Gamma`` and writes point-like IRFs; otherwise it uses ``GammaDiffuse`` and
writes full-enclosure IRFs. Point-like reductions require exactly one of
``theta_cut`` or ``theta_containment``.

Each ``[[dl3.build.reduction]]`` table defines one named DL3 selection. The
quality filters (``intensity_cut``, ``leakage_cut``) are passed to the
reduction and the gammaness cut (``gh_cut`` or ``gh_efficiency``) is carried
inside the IRFs, so data and IRFs always agree. Shared interpolation and MC
options stay under ``[dl3.build]``.

DL2 and DL3 builds follow ``RunSelection``. Selected runs without exactly one
input file directly below ``dl1/`` or ``dl2/`` are skipped with a warning.
Unselected files are ignored, and an empty selection performs no work and
submits no job.

For each named reduction, the DL3 build prepares IRF interpolation grids in
parallel, then reduces DL2 runs in batches across workers after all IRFs are
ready. Workers are bounded by the local CPU count, or the job's CPU allocation
under SLURM. With ``[slurm] enabled = true``:

- **DL2** submits batched jobs (``[slurm.dl2] runs_per_job``) running
  ``lst-dl1-to-dl2``.
- **DL3** writes ``irfs/<reduction_name>/irf_config.json`` and submits one job
  per named reduction running ``lst-dl2-to-dl3``; ``[slurm.dl3]`` sets its
  worker batch size (``runs_per_worker``), CPU (``cpus_per_task``), memory,
  and walltime allocation.

Workspace layout
----------------

.. code-block:: text

   analysis/
   ├── data_check/
   ├── dl1/              # linked input (stage 2)
   ├── dl2/              # linked or built (stage 3)
   ├── irfs/             # built during the DL3 build
   │   └── <reduction_name>/dec_<line>/nsb_<nsb>/<irf_type>/<cut_label>/
   │       └── theta_<zenith>_az_<azimuth>.irf.fits.gz
   └── dl3/<reduction_name>/  # built reductions (stage 4)

The unique ``<reduction_name>`` keeps DL3 products and IRFs separate, even
when reductions differ only in their quality cuts. Here ``<irf_type>`` is
explicitly ``point`` or ``full-enclosure``, and ``<cut_label>`` names the
gammaness selection, ``gheff0.7`` for an energy-dependent efficiency or
``ghcut0.7`` for a fixed cut. Point-like IRFs also include the theta selection,
for example ``ghcut0.7_thetacut0.2`` or ``gheff0.7_thetacont0.68``.
Each reduction reads the IRF directory matching
its own configuration, so incompatible IRFs never share an interpolation grid.

Reruns are incremental: existing links are kept, existing files are reused,
and conflicting files are never replaced. Each level's build mode has an
``overwrite`` option to force rebuilding. A reduction name identifies its
configuration: after changing its cuts, use a new name or enable
``overwrite`` to avoid reusing the previous DL3 files. DL4 generation should
select one named reduction at a time.

Summarizing selected runs: ``summarize-lstana``
-----------------------------------------------

After DL3 files are available, summarize the selected runs into an ECSV or CSV
table:

.. code-block:: bash

   summarize-lstana config/config.toml --analysis-dir /path/to/analysis -o run_summary.ecsv

It records ``run_number``, ``date``, ``RA_PNT``, ``DEC_PNT``, ``AZ_PNT``,
``ZE_PNT``, ``B_DELTA``, ``LIVETIME``, ``N_EVENTS``, and the ``NSB_LEVEL``
read from DL2 provenance. When multiple DL3 reductions are configured, select
a specific reduction with ``-r`` / ``--reduction``.

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
   runs_per_worker = 5
   cpus_per_task = 8
   mem = "40GB"
   time = "04:00:00"

Options of the inactive mode are ignored with a warning, so both blocks may
be kept while switching ``source`` during testing. A level with
``enabled = false`` (or no section at all) is skipped.
