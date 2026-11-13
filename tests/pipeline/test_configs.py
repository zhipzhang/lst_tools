from pathlib import Path

import pytest

from lst_tools.irf import IRFConfig
from lst_tools.pipeline import (
    DL1ProducerConfig,
    DL2ProducerConfig,
    DL3ProducerConfig,
    PipelineConfig,
)
from lst_tools.pipeline.configs import ConfigError

RF_DIRECTORY = "/fefs/aswg/data/models/AllSky/nsb_tuning_0.24/dec_347"


class TestDL1Config:
    def test_missing_section_defaults_to_enabled_link(self):
        config = DL1ProducerConfig.parse(None)
        assert config.enabled is True
        assert config.source == "link"
        assert config.link is None

    def test_link_output_directory(self):
        config = DL1ProducerConfig.parse({"link": {"output_directory": "/data/my/dl1"}})
        assert config.link.output_directory == Path("/data/my/dl1")

    def test_build_source_rejected(self):
        with pytest.raises(ConfigError, match="not supported yet"):
            DL1ProducerConfig.parse({"source": "build"})

    def test_unknown_key_rejected(self):
        with pytest.raises(ConfigError, match="Unknown keys"):
            DL1ProducerConfig.parse({"whatever": True})


class TestDL2Config:
    def test_missing_section_means_disabled(self):
        config = DL2ProducerConfig.parse(None)
        assert config.enabled is False

    def test_build_mode(self):
        config = DL2ProducerConfig.parse(
            {
                "source": "build",
                "build": {
                    "output_directory": "/data/my/dl2",
                    "rf_directory": RF_DIRECTORY,
                    "overwrite": True,
                },
            }
        )
        assert config.enabled is True
        assert config.source == "build"
        assert config.build.output_directory == Path("/data/my/dl2")
        assert config.build.rf_directory == RF_DIRECTORY
        assert config.build.overwrite is True

    def test_rf_directory_template_with_nsb_placeholder_is_accepted(self):
        config = DL2ProducerConfig.parse(
            {"source": "build", "build": {"rf_directory": "/models/AllSky/nsb_tuning_$NSB_LEVEL/dec_347"}}
        )
        assert config.build.rf_directory == "/models/AllSky/nsb_tuning_$NSB_LEVEL/dec_347"

    def test_link_output_directory(self):
        config = DL2ProducerConfig.parse({"source": "link", "link": {"output_directory": "/data/my/dl2"}})
        assert config.link.output_directory == Path("/data/my/dl2")

    def test_link_mode(self):
        config = DL2ProducerConfig.parse({"source": "link"})
        assert config.source == "link"
        assert config.build is None

    def test_source_is_required(self):
        with pytest.raises(ConfigError, match='requires source = "build" or "link"'):
            DL2ProducerConfig.parse({"build": {"rf_directory": RF_DIRECTORY}})

    def test_build_mode_requires_build_section(self):
        with pytest.raises(ConfigError, match=r"\[dl2.build\] is required"):
            DL2ProducerConfig.parse({"source": "build"})

    def test_build_mode_requires_rf_directory(self):
        with pytest.raises(ConfigError, match="rf_directory"):
            DL2ProducerConfig.parse({"source": "build", "build": {}})

    def test_build_options_warned_and_skipped_in_link_mode(self):
        with pytest.warns(UserWarning, match=r'\[dl2.build\] options are ignored because source = "link"'):
            config = DL2ProducerConfig.parse({"source": "link", "build": {"rf_directory": RF_DIRECTORY}})
        assert config.build is None

    def test_unknown_link_option_rejected(self):
        with pytest.raises(ConfigError, match=r"Unknown keys in \[dl2.link\]"):
            DL2ProducerConfig.parse({"source": "link", "link": {"rf_directory": RF_DIRECTORY}})

    def test_disabled_skips_mode_validation(self):
        config = DL2ProducerConfig.parse({"enabled": False, "source": "build"})
        assert config.enabled is False
        assert config.build is None


class TestDL3Config:
    def test_missing_section_means_disabled(self):
        config = DL3ProducerConfig.parse(None)
        assert config.enabled is False

    def test_build_mode_requires_build_section(self):
        with pytest.raises(ConfigError, match=r"\[dl3.build\] is required"):
            DL3ProducerConfig.parse({"source": "build"})

    @staticmethod
    def _reduction(**overrides):
        options = {
            "name": "standard",
            "intensity_cut": 80,
            "leakage_cut": 1.0,
            "gh_efficiency": 0.7,
        }
        options.update(overrides)
        return options

    def test_build_mode_requires_a_reduction(self):
        with pytest.raises(ConfigError, match=r"at least one \[\[dl3.build.reduction\]\]"):
            DL3ProducerConfig.parse({"source": "build", "build": {"overwrite": True}})

    def test_build_mode_with_multiple_reductions(self):
        config = DL3ProducerConfig.parse(
            {
                "source": "build",
                "build": {
                    "interp_method": "nearest",
                    "use_nearest_irf_node": True,
                    "overwrite": True,
                    "mc_dl2_directory": "custom_nsb_$NSB_LEVEL",
                    "reduction": [
                        self._reduction(point_like=True, theta_cut=0.2),
                        self._reduction(
                            name="tight",
                            intensity_cut=150,
                            leakage_cut=0.8,
                            gh_efficiency=None,
                            gh_cut=0.9,
                        ),
                    ],
                },
            }
        )
        assert config.source == "build"
        assert config.build.interp_method == "nearest"
        assert config.build.use_nearest_irf_node is True
        assert config.build.overwrite is True
        assert config.build.irf_directory is None
        assert config.build.mc_dl2_directory == "custom_nsb_$NSB_LEVEL"
        assert [reduction.name for reduction in config.build.reductions] == ["standard", "tight"]
        assert config.build.reductions[0].irf_config.point_like is True
        assert config.build.reductions[0].irf_config.energy_dependent_gh is True
        assert config.build.reductions[1].irf_config.energy_dependent_gh is False
        assert config.build.reductions[1].irf_config.dl3_cuts.global_gh_cut == pytest.approx(0.9)
        assert all(isinstance(reduction.irf_config, IRFConfig) for reduction in config.build.reductions)

    def test_build_mode_rejects_flat_cuts(self):
        with pytest.raises(ConfigError, match=r"cuts moved under \[\[dl3.build.reduction\]\]"):
            DL3ProducerConfig.parse({"source": "build", "build": {"intensity_cut": 80}})

    def test_build_mode_rejects_deprecated_irf_section(self):
        with pytest.raises(ConfigError, match=r"\[dl3.build.irf\] was removed"):
            DL3ProducerConfig.parse({"source": "build", "build": {"irf": {}}})

    def test_build_defaults_to_full_enclosure(self):
        config = DL3ProducerConfig.parse(
            {"source": "build", "build": {"reduction": [self._reduction()]}}
        )
        assert config.build.reductions[0].irf_config.point_like is False
        assert config.build.mc_dl2_directory == "20250212_v0.10.17_allsky_interp_dl2_irfs_nsb_$NSB_LEVEL"

    def test_build_requires_intensity_and_leakage(self):
        reduction = {"name": "standard", "gh_efficiency": 0.7}
        with pytest.raises(ConfigError, match="missing required key"):
            DL3ProducerConfig.parse({"source": "build", "build": {"reduction": [reduction]}})

    def test_build_requires_exactly_one_gh_mode(self):
        reduction = self._reduction(gh_cut=0.5)
        with pytest.raises(ConfigError, match="exactly one"):
            DL3ProducerConfig.parse({"source": "build", "build": {"reduction": [reduction]}})

    def test_build_point_like_requires_exactly_one_theta_mode(self):
        reduction = self._reduction(point_like=True)
        with pytest.raises(ConfigError, match="exactly one of theta_cut and theta_containment"):
            DL3ProducerConfig.parse({"source": "build", "build": {"reduction": [reduction]}})

    @pytest.mark.parametrize("name", [None, "", ".standard", "../standard", "two/words"])
    def test_build_rejects_unsafe_reduction_names(self, name):
        with pytest.raises(ConfigError, match="path-safe name"):
            DL3ProducerConfig.parse(
                {"source": "build", "build": {"reduction": [self._reduction(name=name)]}}
            )

    def test_build_rejects_duplicate_reduction_names(self):
        with pytest.raises(ConfigError, match="names must be unique"):
            DL3ProducerConfig.parse(
                {
                    "source": "build",
                    "build": {"reduction": [self._reduction(), self._reduction(gh_efficiency=0.9)]},
                }
            )

    def test_build_rejects_unknown_reduction_option(self):
        with pytest.raises(ConfigError, match=r"Unknown keys in \[dl3.build.reduction\[0\]\]"):
            DL3ProducerConfig.parse(
                {"source": "build", "build": {"reduction": [self._reduction(unknown=True)]}}
            )

    def test_build_options_warned_and_skipped_in_link_mode(self):
        with pytest.warns(UserWarning, match=r'\[dl3.build\] options are ignored because source = "link"'):
            config = DL3ProducerConfig.parse({"source": "link", "build": {"overwrite": True}})
        assert config.build is None

    def test_link_options_warned_and_skipped_in_build_mode(self):
        with pytest.warns(UserWarning, match=r'\[dl3.link\] options are ignored because source = "build"'):
            config = DL3ProducerConfig.parse(
                {
                    "source": "build",
                    "link": {"irf_type": "point"},
                    "build": {"reduction": [self._reduction()]},
                }
            )
        assert config.link is None
        assert config.build.reductions[0].irf_config is not None

    def test_link_options_default_to_both_irf_types(self):
        config = DL3ProducerConfig.parse({"source": "link", "link": {}})
        assert config.link.irf_type is None

    @pytest.mark.parametrize("irf_type", ["full-enclosure", "point"])
    def test_link_options_accept_valid_irf_type(self, irf_type):
        config = DL3ProducerConfig.parse({"source": "link", "link": {"irf_type": irf_type}})
        assert config.link.irf_type == irf_type

    def test_link_options_reject_invalid_irf_type(self):
        with pytest.raises(ConfigError, match="irf_type must be"):
            DL3ProducerConfig.parse({"source": "link", "link": {"irf_type": "foo"}})

    def test_disabled_skips_mode_validation(self):
        config = DL3ProducerConfig.parse({"enabled": False, "source": "build"})
        assert config.enabled is False


class TestParsePipelineConfig:
    def test_full_config(self):
        pipeline = PipelineConfig.parse(
            {
                "dl1": {},
                "dl2": {"source": "build", "build": {"rf_directory": RF_DIRECTORY}},
                "dl3": {"source": "link", "link": {"irf_type": "point"}},
            }
        )
        assert pipeline.dl1.enabled is True
        assert pipeline.dl2.source == "build"
        assert pipeline.dl3.link.irf_type == "point"

    def test_empty_config_disables_dl2_and_dl3(self):
        pipeline = PipelineConfig.parse({})
        assert pipeline.dl1.enabled is True
        assert pipeline.dl2.enabled is False
        assert pipeline.dl3.enabled is False

    def test_top_level_irf_section_raises_moved_error(self):
        with pytest.raises(ConfigError, match=r"moved to \[\[dl3.build.reduction\]\]"):
            PipelineConfig.parse({"irf": {"intensity_cut": 80}})
