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

## Layout

```
src/lst_ulities/   # package source
tests/             # test suite
```
