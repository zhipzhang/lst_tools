import astropy.units as u
import numpy as np
import pandas as pd
import pytest
from astropy.table import QTable, Table
from lstchain.io.event_selection import EventSelector

from lst_tools.dl2 import DL2EventFilter, EnergyDependentGHCut


class TestEnergyDependentGHCut:
    def test_from_edges_applies_half_open_bins_and_inclusive_cut(self):
        gh_cut = EnergyDependentGHCut.from_edges([0.1, 0.3, 1.0], [0.6, 0.8])
        data = pd.DataFrame(
            {
                "event_id": [0, 1, 2, 3, 4, 5, 6, 7],
                "reco_energy": [0.09, 0.1, 0.29, 0.3, 0.3, 0.9, 1.0, np.nan],
                "gh_score": [0.99, 0.6, 0.59, 0.7, 0.8, 0.81, 0.99, 0.99],
            }
        )

        result = gh_cut.apply(data)

        assert result["event_id"].tolist() == [1, 4, 5]

    def test_from_table_converts_energy_units_and_uses_gammaness_fallback(self):
        cut_table = QTable(
            {
                "low": [100.0, 300.0] * u.GeV,
                "high": [300.0, 1000.0] * u.GeV,
                "cut": [0.6, 0.8],
            }
        )
        gh_cut = EnergyDependentGHCut.from_table(cut_table)
        data = pd.DataFrame(
            {
                "event_id": [0, 1, 2],
                "reco_energy": [0.2, 0.5, 0.5],
                "gammaness": [0.65, 0.75, 0.85],
            }
        )

        result = gh_cut.apply(data)

        assert result["event_id"].tolist() == [0, 2]

    def test_empty_dataframe_preserves_columns(self):
        gh_cut = EnergyDependentGHCut.from_edges([0.1, 1.0], [0.5])
        data = pd.DataFrame(columns=["reco_energy", "gh_score"])

        result = gh_cut.apply(data)

        assert result.empty
        assert result.columns.tolist() == ["reco_energy", "gh_score"]

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"energy_low": np.array([]), "energy_high": np.array([]), "cuts": np.array([])},
            {"energy_low": np.array([0.1, 0.3]), "energy_high": np.array([0.3]), "cuts": np.array([0.5, 0.6])},
        ],
    )
    def test_rejects_empty_or_mismatched_bins(self, kwargs):
        with pytest.raises(ValueError):
            EnergyDependentGHCut(**kwargs)

    @pytest.mark.parametrize(
        ("edges", "cuts"),
        [
            ([0.1], []),
            ([0.3, 0.1], [0.5]),
            ([0.1, 0.3, 0.3], [0.5, 0.6]),
        ],
    )
    def test_from_edges_validates_edges(self, edges, cuts):
        with pytest.raises(ValueError):
            EnergyDependentGHCut.from_edges(edges, cuts)

    def test_from_table_requires_expected_columns(self):
        with pytest.raises(ValueError, match="missing required columns"):
            EnergyDependentGHCut.from_table(Table({"low": [0.1], "high": [1.0]}))


class TestDL2EventFilter:
    def test_from_cuts_configures_event_selector_filters(self):
        event_filter = DL2EventFilter.from_cuts(intensity_cut=50, leakage_cut=0.2, gh_cut=0.7)

        assert event_filter.event_selector.filters["intensity"] == [50.0, np.inf]
        assert event_filter.event_selector.filters["leakage_intensity_width_2"] == [0, 0.2]
        assert event_filter.gh_cut == pytest.approx(0.7)

    def test_delegates_to_event_selector_and_does_not_mutate_input(self, monkeypatch):
        selector = EventSelector()
        calls = []

        def fake_filter_cut(df: pd.DataFrame) -> pd.DataFrame:
            calls.append(df)
            df["mutated"] = True
            return df.iloc[[0, 2]]

        monkeypatch.setattr(selector, "filter_cut", fake_filter_cut)
        data = pd.DataFrame({"event_id": [0, 1, 2], "gh_score": [0.9, 0.9, 0.5]})
        original = data.copy(deep=True)

        quality_only = DL2EventFilter(selector)
        assert quality_only(data)["event_id"].tolist() == [0, 2]
        assert len(calls) == 1
        assert calls[0] is not data
        pd.testing.assert_frame_equal(data, original)

    def test_applies_scalar_gh_cut_after_event_selector(self, monkeypatch):
        selector = EventSelector()
        monkeypatch.setattr(selector, "filter_cut", lambda df: df.iloc[1:])
        data = pd.DataFrame({"event_id": [0, 1, 2], "gh_score": [0.9, 0.7, 0.8]})

        result = DL2EventFilter(selector, gh_cut=0.7)(data)

        assert result["event_id"].tolist() == [2]

    def test_applies_energy_dependent_gh_cut_after_event_selector(self, monkeypatch):
        selector = EventSelector()
        monkeypatch.setattr(selector, "filter_cut", lambda df: df.iloc[1:])
        gh_cut = EnergyDependentGHCut.from_edges([0.1, 0.3, 1.0], [0.6, 0.8])
        data = pd.DataFrame(
            {
                "event_id": [0, 1, 2],
                "reco_energy": [0.2, 0.2, 0.5],
                "gh_score": [0.9, 0.6, 0.75],
            }
        )

        result = DL2EventFilter(selector, gh_cut=gh_cut)(data)

        assert result["event_id"].tolist() == [1]

    @pytest.mark.parametrize("bad_gh_cut", [True, "0.7", object()])
    def test_rejects_invalid_gh_cut_type(self, bad_gh_cut):
        with pytest.raises(TypeError):
            DL2EventFilter(gh_cut=bad_gh_cut)
