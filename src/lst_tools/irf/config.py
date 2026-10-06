"""The complete configuration of one IRF production run."""

import json
from dataclasses import dataclass, field
from pathlib import Path

from lstchain.io.event_selection import DataBinning

from lst_tools.event_filter import EventFilter

DEFAULT_CONFIG_PATH = Path(__file__).parent / "default_config.json"


def _default_data_binning() -> DataBinning:
    """DataBinning with the values from the JSON default config."""
    with open(DEFAULT_CONFIG_PATH) as file_handle:
        binning_config = json.load(file_handle)["DataBinning"]
    return DataBinning(**binning_config)


def _trait_config(component) -> dict:
    return {name: getattr(component, name) for name in component.traits(config=True)}


@dataclass(eq=False)
class IRFConfig:
    """Everything ``lstchain_create_irf_files`` is configured with.

    Composes the event selection (``EventFilter``: EventSelector + DL3Cuts)
    and the binning (``DataBinning``, defaulting to the ``DataBinning``
    section of ``default_config.json``); the gammaness mode flag is taken
    from the filter. Use the same ``event_filter`` that is applied to the
    DL2/DL3 data, so the IRFs and the data selection always agree.

    With ``EventFilter.from_cuts(50, 1.0, gh_cut=0.7)`` and the default
    binning, the serialized config reproduces ``default_config.json``
    exactly (pinned by a test); that file is the human-readable statement
    of the defaults.
    """

    event_filter: EventFilter
    data_binning: DataBinning = field(default_factory=_default_data_binning)

    def to_lstchain_config(self) -> dict:
        """Serialize to the full config dict of ``lstchain_create_irf_files``."""
        return {
            **self.event_filter.to_lstchain_config(),
            "IRFFITSWriter": {"energy_dependent_gh": self.event_filter.energy_dependent_gh},
            "DataBinning": _trait_config(self.data_binning),
        }
