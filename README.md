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

Build the target DL2 link, IRFs, target DL3, matching off-run DL2 files, and a
reproducible `workrun.toml` configuration for one run:

```bash
build-workrun --run 12345 \
  --dl2-path /path/to/dl2 \
  --data-check-path /path/to/data_check \
  --offrun-data-check-path /path/to/offrun/datacheck \
  --offrun-dl1-path /path/to/offruns/dl1 \
  --mc-dl2-path /path/to/mc/DL2/AllSky \
  --irf-output-dir /path/to/irfs \
  --source-name "Crab Nebula" \
  --source-ra 83.632 \
  --source-dec 22.0145 \
  --background-energy-edges='[0.03,0.1,0.3,1.0,3.0,10.0,30.0]' \
  --background-theta-edges='[0.0,0.25,0.5,0.75,1.0,1.5,2.0]' \
  --output-dir /path/to/workdir
```

The same inputs can be supplied in TOML:

```toml
[BuildWorkRun]
run_number = 12345
dl2_path = "/path/to/dl2"
data_check_path = "/path/to/data_check"
offrun_data_check_path = "/path/to/offrun/datacheck"
offrun_dl1_path = "/path/to/offruns/dl1"
mc_dl2_path = "/path/to/mc/DL2/AllSky"
irf_output_dir = "/path/to/irfs"
output_dir = "/path/to/workdir"
source_name = "Crab Nebula"
source_ra = 83.632
source_dec = 22.0145
zenith_tolerance_deg = 2.0
nsb_relative_tolerance = 0.1
date_tolerance_days = 90
require_matching_tailcuts = true
gh_efficiency = 0.7
background_energy_edges = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]
background_theta_edges = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
background_spectral_index = -2.0
# Optional validation runs to leave out of background training:
background_exclude_run_numbers = [12340]
```

Run it as `build-workrun --config=config.toml`. Command-line options supplied
at the same time override TOML values, for example
`build-workrun --config=config.toml --run 12346`. The generated
`Run12345/workrun.toml` contains this `[BuildWorkRun]` table too, so it can be
passed back to the command directly.

Each run is built in this order: link the target DL2, generate and link its
IRFs, generate the target DL3 beside the target DL2 in `RunXXXXX/`, reconstruct
the selected off-run DL2 files, then generate `background2d.fits`. DL3
generation searches the run-local IRF directory with the precise pattern
`azimuth_*_zenith_*/irf.fits.gz`. Source coordinates are specified as numeric
degrees in TOML or on the command line.

The background stage reads the intensity threshold from the target DL2 and
the energy-dependent cuts from the target DL3 `GH_CUTS` table. It applies both
cuts to every training file in `offdl2/`, sums their effective livetimes, and
bins the surviving events using the configured energy and radial-offset edges.

The background stage is also available independently. For example,
`background.toml` can contain:

```toml
[BuildDL2Background]
target_dl2_file = "/path/to/Run12345/dl2_LST-1.Run12345.h5"
target_dl3_file = "/path/to/Run12345/dl3_LST-1.Run12345.fits"
offrun_dl2_path = "/path/to/Run12345/offdl2"
output_file = "/path/to/Run12345/background2d.fits"
energy_edges = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]
theta_edges = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
spectral_index = -2.0
exclude_run_numbers = [12340]
```

Run it with `build-dl2-background --config=background.toml`.

Candidate DL1 links are resolved to their source paths before their tailcuts
are compared with the target DL2 provenance. Mismatches and broken links are
reported and skipped. Existing generated DL2 and IRF files are reused safely.
Use `--allow-mismatched-tailcuts` to include mismatched files when needed;
matching tailcuts remain required by default. The generated `workrun.toml`
records this choice in its `[offrun_tailcuts]` section.
Target-run statistics come from `--data-check-path`; candidate statistics come
from the separate `--offrun-data-check-path`.
Off runs must fall within a true ±90-day seasonal window by default, including
correct month lengths and wrapping across New Year.
Run-local IRFs are exposed directly as `irf/azimuth_<az>_zenith_<zd>` links;
the shared IRF store retains the full declination, intensity, and efficiency
hierarchy.

## Layout

```
src/lst_tools/   # package source
tests/             # test suite
```
