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

## Development

```bash
pytest        # run tests
ruff check .  # lint
```

## Run the analysis pipeline

Edit `config/config.toml` to set the source, quality cuts, zenith-angle
edges, and the RF models. Then run:

```bash
init-lstana config/config.toml --output /path/to/analysis
```

The command runs a six-stage pipeline; everything from DL2 onward is
produced by the pipeline itself:

1. **Select runs** — nightly DataCheck statistics filtered by the
   `[data_filter]` cuts, grouped into the `[zenith_binning]` bins.
2. **Data check** — the selected tables and cut diagnostics in
   `data_check/`.
3. **Link DL1** — idempotent symlinks of the selected runs into
   `dl1/<zd_bin>/`.
4. **Build DL2** (`[dl2]` section) — our own reconstruction of the linked
   DL1 with the configured `rf_directory` RF models into `dl2/<zd_bin>/`.
5. **Build IRFs** (`[irf]` section) — one IRF per MC node of each run's MC
   group in `irfs/nsb_<nsb>/dec_<line>/`, located from the DL2 provenance.
6. **Reduce to DL3** (`[dl3] enabled = true`) — one DL3 FITS observation
   per run in `dl3/<zd_bin>/`.

```text
analysis/
├── data_check/
├── dl1/<zd_bin>/          # linked inputs
├── dl2/<zd_bin>/          # built by stage 4
├── irfs/nsb_<nsb>/dec_<line>/   # built by stage 5
└── dl3/<zd_bin>/          # written by stage 6
```

Bins include their lower edge and exclude their upper edge; the last bin
also includes its upper edge. Reruns are incremental: existing links are
kept, existing DL2/IRF/DL3 files are reused (an IRF is regenerated when the
`[irf]` config changed), and conflicting files are never replaced. Each
stage has an `overwrite` option to force rebuilding.

The DL3 reduction uses lstchain's `lstchain_create_dl3_file` machinery,
interpolating the IRFs of each run's MC group to the run's pointing. The
event selection comes from the `[irf]` section: the quality filters
(`intensity_cut`, `leakage_cut`) are passed to the reduction, and the
gammaness cut (`gh_cut` or `gh_efficiency`) is carried inside the IRFs
themselves, so data and IRFs always agree. For DL3, IRFs are looked up
under `<output>/irfs` unless `[dl3] irf_directory` overrides it — useful to
reduce against IRFs generated outside this workspace (stage 5 skipped).

## DataCheck statistics

Load DataCheck tables and calculate run-wise statistics directly:

```python
from lst_tools.datacheck import DataCheckTables

tables = DataCheckTables.from_files(["/path/to/data_check.h5"])
statistics = tables.statistics
```

## Prepare catalog-separated off runs

Apply the configured basic and advanced quality cuts, ignore the configured
target-angle range, and reject pointings near HESS, LHAASO, and HAWC catalog
sources:

```bash
prepare-offruns config/config.toml \
  --output /path/to/offruns \
  --with-dl1 \
  --with-dl2 \
  --min-separation 3 \
  --extension-factor 2.5
```

This always creates `data_check/DL1_datacheck_offruns.h5`. Pass `--with-dl1`
(`--with-dl` is an alias) and/or `--with-dl2` to create idempotent links under
`dl1/` and/or `dl2/`. When neither option is given, no data links are created.

## Layout

```
src/lst_tools/   # package source
tests/             # test suite
```
