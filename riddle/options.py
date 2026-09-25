"""Serializable, explicit switches for the production RIDDLE algorithm."""
from copy import deepcopy

FEATURES = ("rosenblatt", "guided_fit", "hard_bg", "contrastive_fit", "score_flow", "coherent_mixture")
DEFAULT_ENHANCEMENTS = {
    **{name: True for name in FEATURES},
    "guide_epochs": 20, "guide_warmup_epochs": 10,
    "hard_start_epoch": 5, "hard_pool_fraction": 0.25, "hard_sampling_fraction": 0.5,
    "contrastive_strength": 1.0, "score_flow_epochs": 60,
    "rosenblatt_bins": 12, "rosenblatt_hidden": 64, "rosenblatt_bound": 8.0,

    "guide_folds": 5,
    "guide_mass_conditioning": False,
    "guide_reference_multiplier": 4,
    "guide_refresh_reference": True,
    "guide_ratio_calibration": True,
    "tail_rank": True,
    "tail_candidate_multiplier": 8,
    "tail_hard_fraction": 0.01,
    "tail_strength": 0.05,
    "tail_margin": 0.0,
    "tail_temperature": 1.0,
    "tail_mass_bins": 8,
    "responsibility_temperature": 0.7,
    "qphi_epochs": 40,
    "qphi_mass_bins": 4,
    "checkpoint_weighting": "uniform",
}


def feature_options(settings):
    return {**deepcopy(DEFAULT_ENHANCEMENTS), **settings.get("enhancements", {})}


def effective_features(settings):
    options = feature_options(settings)
    result = {name: options[name] for name in FEATURES}

    result["hard_bg"] = result["hard_bg"] and result["guided_fit"]
    return result


def add_feature_arguments(parser):
    import argparse
    for name in FEATURES:
        parser.add_argument("--" + name.replace("_", "-"), dest=name,
                            action=argparse.BooleanOptionalAction, default=None,
                            help="RIDDLE: enable/disable " + name.replace("_", " ") +
                            " (default: enabled; YAML can override)")

