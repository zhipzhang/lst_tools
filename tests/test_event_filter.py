import numpy as np
import pandas as pd
import pytest
from lstchain.io.event_selection import DataBinning

from lst_tools.event_filter import EventFilter


def quality_filter_columns(n):
    """Columns required by the default EventSelector filters."""
    return {
        "r": np.full(n, 0.5),
        "wl": np.full(n, 0.5),
        "event_type": np.full(n, 32),
        "length": np.full(n, 10.0),
        "width": np.full(n, 10.0),
    }


@pytest.fixture
def events():
    n = 9
    return pd.DataFrame(
        {
            "event_id": range(n),
            "reco_energy": [0.09, 0.1, 0.29, 0.3, 0.3, 0.9, 1.0, 0.2, None],
            "gh_score": [0.99, 0.6, 0.59, 0.7, 0.8, 0.81, 0.99, None, 0.99],
            "intensity": [100, 100, 100, 100, 40, 60, 100, 100, 100],
            "leakage_intensity_width_2": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.5, 0.1, 0.1],
            **quality_filter_columns(n),
        }
    )


def test_from_cuts_requires_exactly_one_gammaness_mode():
    with pytest.raises(ValueError, match="exactly one of gh_cut and gh_efficiency"):
        EventFilter.from_cuts(50, 0.2)
    with pytest.raises(ValueError, match="exactly one of gh_cut and gh_efficiency"):
        EventFilter.from_cuts(50, 0.2, gh_cut=0.5, gh_efficiency=0.7)


@pytest.mark.parametrize("gh_efficiency", [0.0, -0.5, 1.5])
def test_from_cuts_validates_gh_efficiency(gh_efficiency):
    with pytest.raises(ValueError, match="gh_efficiency must be in"):
        EventFilter.from_cuts(50, 0.2, gh_efficiency=gh_efficiency)


def test_from_cuts_sets_the_filters_and_the_gammaness_mode():
    fixed = EventFilter.from_cuts(80, 0.2, gh_cut=0.5)
    assert fixed.event_selector.filters["intensity"] == [80.0, np.inf]
    assert fixed.event_selector.filters["leakage_intensity_width_2"] == [0, 0.2]
    assert fixed.dl3_cuts.global_gh_cut == pytest.approx(0.5)
    assert not fixed.energy_dependent_gh

    quantile = EventFilter.from_cuts(80, 0.2, gh_efficiency=0.7)
    assert quantile.dl3_cuts.gh_efficiency == pytest.approx(0.7)
    assert quantile.energy_dependent_gh
    np.testing.assert_allclose(
        quantile.energy_bins,
        DataBinning().reco_energy_bins().to_value("TeV"),
    )


def test_fixed_mode_applies_quality_gammaness_intensity_and_leakage_cuts(events):
    event_filter = EventFilter.from_cuts(50, 0.2, gh_cut=0.8)

    result = event_filter(events)

    # gh_score >= 0.8 keeps 0, 4, 5, 6, 8; intensity >= 50 drops 4;
    # leakage <= 0.2 drops 6
    assert result["event_id"].tolist() == [0, 5, 8]


def test_quantile_mode_keeps_about_the_efficiency_fraction():
    rng = np.random.default_rng(42)
    n = 50000
    events = pd.DataFrame(
        {
            "reco_energy": rng.uniform(0.05, 5.0, n),
            "gh_score": rng.uniform(0, 1, n),
            "intensity": np.full(n, 100.0),
            "leakage_intensity_width_2": np.full(n, 0.1),
            **quality_filter_columns(n),
        }
    )
    event_filter = EventFilter.from_cuts(50, 0.2, gh_efficiency=0.7)

    result = event_filter(events)

    assert len(result) / len(events) == pytest.approx(0.7, abs=0.05)


def test_quantile_mode_does_not_mutate_the_callers_frame():
    rng = np.random.default_rng(7)
    n = 50000
    events = pd.DataFrame(
        {
            "reco_energy": rng.uniform(0.05, 5.0, n),
            "gh_score": rng.uniform(0, 1, n),
            "intensity": np.full(n, 100.0),
            "leakage_intensity_width_2": np.full(n, 0.1),
            **quality_filter_columns(n),
        }
    )
    columns_before = events.columns.tolist()

    EventFilter.from_cuts(50, 0.2, gh_efficiency=0.7)(events)

    assert events.columns.tolist() == columns_before


def test_to_lstchain_config_serializes_the_wrapped_components():
    event_filter = EventFilter.from_cuts(80, 0.2, gh_cut=0.5)

    config = event_filter.to_lstchain_config()

    assert config["EventSelector"]["filters"]["intensity"] == [80.0, np.inf]
    assert config["EventSelector"]["filters"]["leakage_intensity_width_2"] == [0, 0.2]
    assert config["DL3Cuts"]["global_gh_cut"] == pytest.approx(0.5)
    assert config["DL3Cuts"]["min_gh_cut"] == pytest.approx(0.1)
    assert config["DL3Cuts"]["max_gh_cut"] == pytest.approx(0.98)


def test_to_lstchain_config_quantile_mode():
    event_filter = EventFilter.from_cuts(80, 0.2, gh_efficiency=0.7)

    assert event_filter.to_lstchain_config()["DL3Cuts"]["gh_efficiency"] == pytest.approx(0.7)
