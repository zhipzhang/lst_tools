# lst_ulities

Utilities built on top of [lstchain](https://github.com/cta-observatory/cta-lstchain)
and [ctapipe](https://github.com/cta-observatory/ctapipe) for LST data analysis.

## Installation

Create a conda environment (recommended, since lstchain is distributed via conda):

```bash
conda create -n lst_ulities -c conda-forge python=3.11 lstchain
conda activate lst_ulities
pip install -e .[dev]
```

## Development

```bash
pytest        # run tests
ruff check .  # lint
```

## Initialize a zenith-binned analysis

Edit `config/config.toml` to set the source, quality cuts, and zenith-angle
edges. Then run:

```bash
init-lstana config/config.toml --output /path/to/analysis
```

The command filters runs once and creates idempotent links grouped by data
level and zenith-angle bin:

```text
analysis/
├── data_check/
├── dl1/
│   ├── zd_0_20/
│   ├── zd_20_30/
│   └── ...
└── dl2/
    ├── zd_0_20/
    ├── zd_20_30/
    └── ...
```

Bins include their lower edge and exclude their upper edge; the last bin also
includes its upper edge. Running the command again keeps correct links and
does not replace conflicting files. The selected-run HDF5 table contains both
`mean_zenith_angle` and `zenith_bin` columns.

DL3 linking is controlled explicitly by `[dl3]` in the TOML configuration. The
configured products are crossed with `cut_configs`, and only exact matches are
linked. Other discovered DL3 files are ignored. Each combination is kept
separate to avoid mixing analysis products or colliding filenames. When the
same product exists under multiple processing-version directories immediately
before `std`, the newest semantic version is selected automatically:

```text
dl3/
├── point/
│   ├── gheff0.7_thetacont0.7/zd_0_20/
│   └── gheff0.9_thetacont0.7/zd_0_20/
├── full_ring/
└── full_diffuse/
```

## Shared DataCheck tables

Point the global store at an analysis's `data_check/` directory once. The
tables and run statistics are stored in separate files and loaded lazily when
each property is first accessed:

```python
from lst_ulities.datacheck import initialize_data_check, run_data_check

initialize_data_check("/path/to/analysis/data_check")

# Available from any other module after initialization.
tables = run_data_check.data_check_tables
run_statistics = run_data_check.run_statistics
```

`init-lstana` points the store at its output directory after writing both
products. Access before initialization raises a clear `RuntimeError`; neither
file is read merely by importing or initializing the store.

## Layout

```
src/lst_ulities/   # package source
tests/             # test suite
```
