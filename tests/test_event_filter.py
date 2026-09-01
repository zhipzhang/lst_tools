import pandas as pd
import pytest

from lst_tools.event_filter import apply_energy_dependent_gammaness_cuts


@pytest.fixture
def events():
    return pd.DataFrame(
        {
            "event_id": range(9),
            "reco_energy": [0.09, 0.1, 0.29, 0.3, 0.3, 0.9, 1.0, 0.2, None],
            "gammaness": [0.99, 0.6, 0.59, 0.7, 0.8, 0.81, 0.99, None, 0.99],
        },
        index=[8, 3, 7, 1, 9, 2, 6, 5, 4],
    )


@pytest.mark.parametrize(
    "data",
    [
        pd.DataFrame(),
        pd.DataFrame({"reco_energy": [0.2]}),
        pd.DataFrame({"gammaness": [0.8]}),
    ],
)
def test_necessary_columns(data):
    message = "DataFrame must contain 'reco_energy' column and 'gammaness' column"

    with pytest.raises(ValueError, match=message):
        apply_energy_dependent_gammaness_cuts(data, [], [], [])


def test_applies_cut_for_each_energy_bin(events):
    result = apply_energy_dependent_gammaness_cuts(
        events,
        energy_low=[0.1, 0.3],
        energy_high=[0.3, 1.0],
        gh_cuts=[0.6, 0.8],
    )

    assert result["event_id"].tolist() == [1, 4, 5]
    assert result.index.tolist() == [3, 9, 2]
    assert result.columns.tolist() == events.columns.tolist()


def test_does_not_modify_input(events):
    original = events.copy(deep=True)

    apply_energy_dependent_gammaness_cuts(events, [0.1], [1.0], [0.7])

    pd.testing.assert_frame_equal(events, original)


def test_empty_configuration_returns_empty_frame(events):
    result = apply_energy_dependent_gammaness_cuts(events, [], [], [])

    pd.testing.assert_frame_equal(result, events.iloc[0:0])


@pytest.mark.parametrize(
    ("energy_low", "energy_high", "gh_cuts", "message"),
    [
        ([0.1], [0.3, 1.0], [0.8], "must have the same length"),
        ([0.3], [0.3], [0.8], "energy_low must be less than energy_high"),
        ([0.1], [float("nan")], [0.8], "must not contain NaN"),
        ([0.1, 0.2], [0.4, 0.5], [0.6, 0.8], "must not overlap"),
        ([0.1], [0.3], ["not-a-number"], "must contain numeric values"),
    ],
)
def test_invalid_cut_configuration(events, energy_low, energy_high, gh_cuts, message):
    with pytest.raises(ValueError, match=message):
        apply_energy_dependent_gammaness_cuts(events, energy_low, energy_high, gh_cuts)
