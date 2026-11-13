"""Configuration generation for lstchain IRF creation and DL2-to-DL3 reduction."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Self

import numpy as np
from lstchain.io.event_selection import DataBinning, DL3Cuts, EventSelector

from .layout import gammaness_label

DEFAULT_CONFIG_PATH = Path(__file__).parent / "default_config.json"


def _load_default_config() -> dict:
    with open(DEFAULT_CONFIG_PATH) as file_handle:
        return json.load(file_handle)


def _trait_config(component) -> dict:
    return {name: getattr(component, name) for name in component.traits(config=True)}


def _default_event_selector() -> EventSelector:
    return EventSelector(**_load_default_config()["EventSelector"])


def _default_dl3_cuts() -> DL3Cuts:
    return DL3Cuts(**_load_default_config()["DL3Cuts"])


def _default_data_binning() -> DataBinning:
    return DataBinning(**_load_default_config()["DataBinning"])


@dataclass(eq=False)
class IRFConfig:
    """Configuration for ``lstchain_create_irf_files`` and ``DataReductionFITSWriter``.

    Composes lstchain's ``EventSelector``, ``DL3Cuts``, and ``DataBinning``
    alongside the ``IRFFITSWriter`` mode flags (``energy_dependent_gh``,
    ``point_like``, ``energy_dependent_theta``).

    All defaults are loaded live from ``irf/default_config.json``:

    - ``EventSelector.filters``:
        - ``intensity``: ``[50, inf]``
        - ``width``: ``[0, inf]``
        - ``length``: ``[0, inf]``
        - ``r``: ``[0, 1]``
        - ``wl``: ``[0.01, 1]``
        - ``leakage_intensity_width_2``: ``[0, 1]``
        - ``event_type``: ``[32, 32]``
    - ``EventSelector.finite_params``: ``["intensity", "length", "width"]``
    - ``DL3Cuts``:
        - ``min_event_p_en_bin = 100``
        - ``global_gh_cut = 0.7``, ``gh_efficiency = 0.7``,
          ``min_gh_cut = 0.1``, ``max_gh_cut = 0.98``
        - ``global_theta_cut = 0.2``, ``theta_containment = 0.7``,
          ``min_theta_cut = 0.1``, ``max_theta_cut = 0.32``, ``fill_theta_cut = 0.32``
        - ``global_alpha_cut = 10``, ``alpha_containment = 0.7``,
          ``min_alpha_cut = 1``, ``max_alpha_cut = 20``, ``fill_alpha_cut = 20``
        - ``allowed_tels = [1]``
    - ``DataBinning``:
        - ``true_energy``: ``0.005`` to ``500`` TeV, ``25`` bins (``scale_true_energy = 1.0``)
        - ``reco_energy``: ``0.005`` to ``500`` TeV, ``25`` bins
        - ``energy_migration``: ``0.2`` to ``5``, ``30`` bins
        - ``fov_offset``: ``0`` to ``2.5`` deg, ``6`` edges
        - ``bkg_fov_offset``: ``0`` to ``10`` deg, ``21`` edges
        - ``source_offset``: ``0`` to ``1`` deg, ``101`` edges
    """

    event_selector: EventSelector = field(default_factory=_default_event_selector)
    dl3_cuts: DL3Cuts = field(default_factory=_default_dl3_cuts)
    data_binning: DataBinning = field(default_factory=_default_data_binning)
    energy_dependent_gh: bool = False
    point_like: bool = False
    energy_dependent_theta: bool = False

    @classmethod
    def from_cuts(
        cls,
        intensity_cut: float | None = None,
        leakage_cut: float | None = None,
        *,
        gh_cut: float | None = None,
        gh_efficiency: float | None = None,
        point_like: bool = False,
        theta_cut: float | None = None,
        theta_containment: float | None = None,
        data_binning: DataBinning | None = None,
    ) -> Self:
        """Build an ``IRFConfig`` from analysis cuts on top of ``default_config.json``.

        Parameters
        ----------
        intensity_cut
            Minimum image intensity in p.e. (defaults to ``50`` from ``default_config.json``).
        leakage_cut
            Maximum ``leakage_intensity_width_2`` (defaults to ``1.0`` from ``default_config.json``).
        gh_cut, gh_efficiency
            Gammaness selection: pass exactly one of ``gh_cut`` (fixed global cut)
            or ``gh_efficiency`` (energy-dependent quantile efficiency in ``(0, 1]``).
        point_like
            If ``False`` (default), configure full-enclosure IRFs; theta cuts are forbidden.
            If ``True``, configure point-like IRFs; exactly one of ``theta_cut`` or
            ``theta_containment`` must be provided.
        theta_cut, theta_containment
            Angular cut selection for ``point_like=True``: pass exactly one of
            ``theta_cut`` (fixed global cut in deg) or ``theta_containment``
            (energy-dependent containment fraction in ``(0, 1]``).
        data_binning
            Optional ``DataBinning`` override (defaults to ``default_config.json``).
        """
        if (gh_cut is None) == (gh_efficiency is None):
            raise ValueError("exactly one of gh_cut and gh_efficiency must be set")
        if gh_efficiency is not None and not 0 < gh_efficiency <= 1:
            raise ValueError("gh_efficiency must be in (0, 1]")

        if not point_like:
            if theta_cut is not None or theta_containment is not None:
                raise ValueError("theta_cut and theta_containment are only valid when point_like=True")
        else:
            if (theta_cut is None) == (theta_containment is None):
                raise ValueError("exactly one of theta_cut and theta_containment must be set when point_like=True")
            if theta_containment is not None and not 0 < theta_containment <= 1:
                raise ValueError("theta_containment must be in (0, 1]")
            if theta_cut is not None and theta_cut <= 0:
                raise ValueError("theta_cut must be positive")

        defaults = _load_default_config()
        selector_cfg = defaults["EventSelector"]
        filters = dict(selector_cfg["filters"])
        if intensity_cut is not None:
            filters["intensity"] = [float(intensity_cut), np.inf]
        if leakage_cut is not None:
            filters["leakage_intensity_width_2"] = [0, float(leakage_cut)]
        event_selector = EventSelector(filters=filters, finite_params=list(selector_cfg["finite_params"]))

        cuts_cfg = dict(defaults["DL3Cuts"])
        if gh_cut is not None:
            cuts_cfg["global_gh_cut"] = float(gh_cut)
        else:
            cuts_cfg["gh_efficiency"] = float(gh_efficiency)
        if theta_cut is not None:
            cuts_cfg["global_theta_cut"] = float(theta_cut)
        elif theta_containment is not None:
            cuts_cfg["theta_containment"] = float(theta_containment)
        dl3_cuts = DL3Cuts(**cuts_cfg)

        return cls(
            event_selector=event_selector,
            dl3_cuts=dl3_cuts,
            data_binning=data_binning if data_binning is not None else DataBinning(**defaults["DataBinning"]),
            energy_dependent_gh=gh_efficiency is not None,
            point_like=point_like,
            energy_dependent_theta=point_like and theta_containment is not None,
        )

    @classmethod
    def from_json(cls, path: str | Path) -> Self:
        """Load an ``IRFConfig`` from a lstchain IRF JSON config file."""
        with open(path) as file_handle:
            raw = json.load(file_handle)
        defaults = _load_default_config()
        writer = raw.get("IRFFITSWriter", {})
        point_like = bool(writer.get("point_like", False))
        return cls(
            event_selector=EventSelector(**raw.get("EventSelector", defaults["EventSelector"])),
            dl3_cuts=DL3Cuts(**raw.get("DL3Cuts", defaults["DL3Cuts"])),
            data_binning=DataBinning(**raw.get("DataBinning", defaults["DataBinning"])),
            energy_dependent_gh=bool(writer.get("energy_dependent_gh", False)),
            point_like=point_like,
            energy_dependent_theta=point_like and bool(writer.get("energy_dependent_theta", False)),
        )

    @property
    def irf_type(self) -> Literal["full-enclosure", "point"]:
        return "point" if self.point_like else "full-enclosure"

    @property
    def cut_label(self) -> str:
        """Path-safe name of the configured gammaness and angular selections."""
        if self.energy_dependent_gh:
            label = gammaness_label("efficiency", float(self.dl3_cuts.gh_efficiency))
        else:
            label = gammaness_label("cut", float(self.dl3_cuts.global_gh_cut))
        if not self.point_like:
            return label
        if self.energy_dependent_theta:
            return f"{label}_thetacont{float(self.dl3_cuts.theta_containment):g}"
        return f"{label}_thetacut{float(self.dl3_cuts.global_theta_cut):g}"

    def to_dl3_config(self) -> dict:
        """Export the ``EventSelector`` and ``DL3Cuts`` sections for ``DataReductionFITSWriter``."""
        return {
            "EventSelector": _trait_config(self.event_selector),
            "DL3Cuts": _trait_config(self.dl3_cuts),
        }

    def to_irf_config(self) -> dict:
        """Export the full config dict for ``lstchain_create_irf_files``."""
        writer_config = {
            "energy_dependent_gh": self.energy_dependent_gh,
            "point_like": self.point_like,
        }
        if self.point_like:
            writer_config["energy_dependent_theta"] = self.energy_dependent_theta
        return {
            **self.to_dl3_config(),
            "IRFFITSWriter": writer_config,
            "DataBinning": _trait_config(self.data_binning),
        }

    def write_irf_config(self, path: str | Path) -> Path:
        """Write the ``lstchain_create_irf_files`` JSON config to ``path``."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as file_handle:
            json.dump(self.to_irf_config(), file_handle, indent=4)
        return output_path
