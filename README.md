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
does not replace conflicting files. The `data_check/` directory contains the
selected DataCheck tables and advanced-cut diagnostic plots.

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

## Build a single-run workspace

Build the target DL2 link, matching off-run DL2 files, IRFs, and a reproducible
`workrun.toml` configuration for one run:

```bash
build-workrun --run 12345 \
  --dl2-path /path/to/dl2 \
  --data-check-path /path/to/data_check \
  --offrun-dl1-path /path/to/offruns/dl1 \
  --mc-dl2-path /path/to/mc/DL2/AllSky \
  --irf-output-dir /path/to/irfs \
  --output-dir /path/to/workdir
```

Candidate DL1 links are resolved to their source paths before their tailcuts
are compared with the target DL2 provenance. Mismatches and broken links are
reported and skipped. Existing generated DL2 and IRF files are reused safely.

## Layout

```
src/lst_tools/   # package source
tests/             # test suite
```
