# lst_tools

Utilities built on top of [lstchain](https://github.com/cta-observatory/cta-lstchain)
and [ctapipe](https://github.com/cta-observatory/ctapipe) for LST data analysis.

## Installation

Create a conda environment (recommended, since lstchain is distributed via conda):

```bash
conda create -n lst_tools -c conda-forge python=3.11 lstchain
conda activate lst_tools
pip install -e .[dev]
```

## Loading DataCheck statistics

The nightly LST-1 DataCheck HDF5 files can be loaded and reduced to
run-wise statistics directly:

```python
from lst_tools.datacheck import DataCheckTables

tables = DataCheckTables.from_files(["/path/to/DL1_datacheck_20240101.h5"])
statistics = tables.statistics          # one row per run

# keep a subset of runs and write a new datacheck file
tables.select_runs([14083, 14084]).save_to_h5file("selected.h5", overwrite=True)
```

Quality cuts are applied with `DataFilter`:

```python
from lst_tools.datacheck import DataFilter

data_filter = DataFilter(
    source_ra=83.632,
    source_dec=22.0145,
    min_angle_to_source=0.3,
    max_angle_to_source=0.5,
    max_pointing_dec_std=0.01,
)
after_basic = data_filter.apply_basic_cuts(statistics)
selected = data_filter.apply_advanced_cuts(after_basic)
```

## Running the analysis pipeline: `init-lstana`

Edit `config/config.toml` (the source, zenith binning, quality cuts, and
one section per data level), then run:

```bash
init-lstana config/config.toml --output /path/to/analysis
```

Every data level has two possible origins, selected by a `source` key:

- `source = "link"` — place the files the standard pipeline already
  produced on La Palma into the workspace zenith-bin tree.
- `source = "build"` — produce the level ourselves.

The stages:

1. **Select runs** — nightly DataCheck statistics filtered by the
   `[data_filter]` cuts (and the `[source]` position), grouped into the
   `[zenith_binning]` bins. The selected tables and advanced-cut
   diagnostics are saved under `data_check/`.
2. **DL1** (`[dl1]`) — the external input; linking is the only supported
   mode today (`dl1/<zd_bin>/`).
3. **DL2** (`[dl2]`) — either link the official DL2, or build our own by
   reconstructing the linked DL1 with the RF models of
   `[dl2.build] rf_directory` (a template: `$NSB_LEVEL` is substituted
   per run from its DL1 file).
4. **DL3** (`[dl3]`) — either link existing DL3 products (newest
   version, optionally narrowed by `[dl3.link] irf_type`), or build them:
   IRFs are generated from the MC test dataset matching each DL2 file's
   provenance (`[dl3.build.irf]`), then every run is reduced to a DL3
   FITS observation, interpolating the IRFs to the run's pointing.

The MC campaign comes from `[dl3.build.irf] mc_dl2_directory`, a template
whose `$NSB_LEVEL` is replaced from each DL2 file's provenance. Relative
paths resolve below `/fefs/aswg/data/mc/DL2/AllSky`; absolute paths are also
accepted. `point_like = true` selects `Gamma` MC and point-like IRFs, while
the default `false` selects `GammaDiffuse` and full-enclosure IRFs.

The event selection for built DL3 comes from `[dl3.build.irf]`: the
quality filters (`intensity_cut`, `leakage_cut`) are passed to the
reduction and the gammaness cut (`gh_cut` or `gh_efficiency`) is carried
inside the IRFs, so data and IRFs always agree. Generated IRFs are stored
under `irfs/nsb_<nsb>/dec_<line>/<irf_type>/`, with `<irf_type>` explicitly
set to `point` or `full-enclosure`.

Zenith bins include their lower edge and exclude their upper edge; the
last bin also includes its upper edge. Reruns are incremental: existing
links are kept, existing files are reused, and conflicting files are
never replaced (each level has an `overwrite` option for its build mode).
See the comments in `config/config.toml` for every option.

## Preparing catalog-separated off runs: `prepare-offruns`

```bash
prepare-offruns config/config.toml \
  --output /path/to/offruns \
  --with-dl1 \
  --with-dl2 \
  --min-separation 3 \
  --extension-factor 2.5
```

Applies the configured basic and advanced quality cuts with the
target-angle range disabled, then rejects pointings near HESS, LHAASO,
and HAWC catalog sources. Always writes
`data_check/DL1_datacheck_offruns.h5`; `--with-dl1` (`--with-dl` is an
alias) and `--with-dl2` additionally create idempotent data links under
`dl1/` and `dl2/`.

## Development

```bash
pytest        # run tests
ruff check .  # lint
```

## Layout

```
src/lst_tools/     # package source
config/config.toml # annotated example pipeline configuration
tests/             # test suite
```
