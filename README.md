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

## Layout

```
src/lst_ulities/   # package source
tests/             # test suite
```
