from copy import deepcopy
import math
from pathlib import Path

import yaml

def default_config_path(name):
    """Locate bundled configuration files."""
    import sysconfig
    if name not in ("settings.yaml", "datasets.yaml", "populations.yaml"):
        raise ValueError("Unknown bundled configuration")
    candidates = (Path(__file__).resolve().parents[1] / "config" / name,
                  Path(sysconfig.get_path("data")) / "config" / name)
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"Bundled {name} missing; reinstall riddle-lhco or use its source tree")


DEFAULT_PATH = default_config_path("settings.yaml")


def keys(mapping, expected, label):
    if not isinstance(mapping, dict) or set(mapping) != set(expected.split()):
        raise ValueError(f"Invalid {label} settings keys")


def integer(value, label, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")


def number(value, label, minimum=0, maximum=math.inf, *, strict=True):
    if (
        type(value) not in (int, float)
        or not math.isfinite(value)
        or not (value > minimum if strict else value >= minimum)
        or value >= maximum
    ):
        raise ValueError(f"Invalid {label}")


def sr_closure_mode(value):
    if type(value) is bool:
        value = "on" if value else "off"
    if value not in ("auto", "on", "off"):
        raise ValueError("riddle.sr_closure must be auto, on, or off")
    return value


def validate_residual(value):
    value = deepcopy(value)
    value.setdefault("core", "residual")
    if value["core"] not in ("residual", "stein_witness"):
        raise ValueError("riddle.core must be residual or stein_witness")
    stein_defaults = {
        "hidden_features": 128,
        "hidden_layers": 3,
        "activation": "silu",
        "witness_regularization": 1.0,
        "potential_center_strength": 0.0,
        "data_tail_fraction": 0.1,
        "closure_samples": 4096,
        "closure_sigma": 5.0,
        "closure_mass_bins": 8,
        "ensemble_fit_selection": "all-valid",
        "scoring": {
            "mode": "tail_focus",
            "reference_samples": 524288,
            "reference_split": 0.5,
            "mass_bins": 8,
            "energy_weight": 1.5,
            "operator_weight": 0.6,
            "operator_gate_z": 1.96,
            "operator_temperature": 0.25,
            "beta": 0.5,
            "local_gate_z": 1.28,
            "local_temperature": 1.0,
            "final_mass_bins": 2,
            "final_transform": "background_cdf_power",
            "final_power": 10.0,
            "qscore_batch_size": 32768,
            "inference_batch_size": 16384,
            # Preserve legacy scoring when auto-switch settings are absent.
            "auto_switch": {
                "enabled": {"riddle": False, "iad": False, "supervised": False},
                "fallback": {"riddle": "tail_focus", "iad": "tail_focus", "supervised": "potential_qnorm"},
                "reference_c_samples": 131072,
                "efficiencies": [0.004, 0.01, 0.05, 0.1],
                "weights": [0.5, 0.3, 0.15, 0.05],
                "confidence": 0.99,
                "bootstrap_replicas": 4000,
                "min_tail_events": 50,
                "seed": 75501,
            },
            "support_guard": {
                "enabled": True,
                "statistic": "radius",
                "mass_bins": 8,
                "gate_quantile": 0.999,
                "weight": 2.0,
                "temperature": 0.25,
            },
        },
    }
    supplied_stein = value.get("stein", {})
    if not isinstance(supplied_stein, dict) or set(supplied_stein) - stein_defaults.keys():
        raise ValueError("Invalid Stein residual settings")
    supplied_scoring = supplied_stein.get("scoring", {})
    if not isinstance(supplied_scoring, dict) or set(supplied_scoring) - stein_defaults["scoring"].keys():
        raise ValueError("Invalid Stein scoring settings")
    supplied_support_guard = supplied_scoring.get("support_guard", {})
    if (not isinstance(supplied_support_guard, dict)
            or set(supplied_support_guard) - stein_defaults["scoring"]["support_guard"].keys()):
        raise ValueError("Invalid Stein support guard settings")
    merged_scoring = {**stein_defaults["scoring"], **supplied_scoring}
    merged_scoring["support_guard"] = {
        **stein_defaults["scoring"]["support_guard"], **supplied_support_guard
    }
    supplied_auto = supplied_scoring.get("auto_switch", {})
    auto_defaults = stein_defaults["scoring"]["auto_switch"]
    if not isinstance(supplied_auto, dict) or set(supplied_auto) - auto_defaults.keys():
        raise ValueError("Invalid auto-switch settings")
    merged_scoring["auto_switch"] = {**auto_defaults, **supplied_auto}
    value["stein"] = {**stein_defaults, **supplied_stein,
                      "scoring": merged_scoring}
    from .options import DEFAULT_ENHANCEMENTS, FEATURES, feature_options
    supplied = value.get("enhancements", {})
    if not isinstance(supplied, dict) or set(supplied) - DEFAULT_ENHANCEMENTS.keys():
        raise ValueError("Invalid RIDDLE enhancement settings")
    value["enhancements"] = e = feature_options(value)
    for name in FEATURES:
        if type(e[name]) is not bool:
            raise ValueError(f"{name} must be boolean")
    for name in ("guide_epochs", "score_flow_epochs", "rosenblatt_bins", "rosenblatt_hidden"):
        integer(e[name], name, 2)
    for name in ("guide_warmup_epochs", "hard_start_epoch"):
        integer(e[name], name, 0)
    for name in ("guide_folds",):
        integer(e[name], name, 2)
    for name in ("guide_reference_multiplier", "tail_candidate_multiplier", "tail_mass_bins", "qphi_mass_bins", "qphi_hidden_features", "pseudo_sr_parallel_probes"):
        integer(e[name], name, 1)
    if e["pseudo_sr_parallel_probes"] > 6:
        raise ValueError("pseudo_sr_parallel_probes must be at most 6")
    integer(e["qphi_epochs"], "qphi_epochs", 10)
    for name in ("guide_mass_conditioning", "guide_refresh_reference", "guide_ratio_calibration", "tail_rank",
                 "contrastive_positive_weighted", "contrastive_negative_weighted"):
        if type(e[name]) is not bool:
            raise ValueError(f"{name} must be boolean")
    for name in ("hard_pool_fraction", "hard_sampling_fraction", "tail_hard_fraction"):
        number(e[name], name, maximum=1)
    for name in ("contrastive_strength", "rosenblatt_bound", "tail_temperature", "responsibility_temperature"):
        number(e[name], name)
    for name in ("tail_strength", "tail_margin"):
        number(e[name], name, strict=False)
    if e["checkpoint_weighting"] not in ("uniform", "validation_likelihood"):
        raise ValueError("checkpoint_weighting must be uniform or validation_likelihood")
    if e["guided_fit"] and e["hard_bg"] and e["hard_start_epoch"] >= e["guide_epochs"]:
        raise ValueError("Hard-background mining must start before the last guide epoch")
    if isinstance(value, dict) and "fits" in value:
        if "runs" in value:
            raise ValueError("Use fits, or the legacy runs key, not both")
        value["runs"] = value.pop("fits")

    value.setdefault("fit_recovery", {"max_retries": 2, "validation_sigma": 2.0})


    from .roles import DEFAULT_POLICY
    value.setdefault("data_policy", DEFAULT_POLICY)

    value.setdefault("ensemble_completion", "strict")
    value.setdefault("ensemble_fit_selection", "all")
    value.setdefault("ensemble_fit_count", 20)


    value.setdefault("mass_fraction", {
        "enabled": True,
        "control_points": 5,
        "smoothness": 0.0001,
        "variation": 0.0002,
        "damping": 0.5,
        "min_fraction": 1e-5,
        "max_fraction": 0.5,
        "max_iterations": 100,
    })

    extra = "".join(" " + k for k in ("mass_conditioning", "input_space", "optimization", "data_policy", "ensemble_completion", "ensemble_fit_selection", "ensemble_fit_count", "background_correction", "sr_closure") if k in value)
    keys(value, "core runs epochs fractions initialization flow training fit_recovery enhancements stein mass_fraction" + extra, "RIDDLE")
    if "data_policy" in value:
        from .roles import policy_parts, DIAGNOSTIC_POLICIES
        policy_parts(value["data_policy"])
        if value["data_policy"] in DIAGNOSTIC_POLICIES and (value["runs"] != 1 or not all(e[k] for k in FEATURES)):
            raise ValueError("Diagnostic role replay requires one fit and all six features")
    if value["ensemble_completion"] not in ("strict", "partial"):
        raise ValueError("ensemble_completion must be strict or partial")
    if value["ensemble_fit_selection"] not in ("validation-best", "all"):
        raise ValueError("ensemble_fit_selection must be validation-best or all")
    integer(value["ensemble_fit_count"], "ensemble_fit_count")
    if "mass_conditioning" in value and type(value["mass_conditioning"]) is not bool:
        raise ValueError("mass_conditioning must be boolean")
    if value["core"] == "stein_witness" and not value.get("mass_conditioning", False):
        raise ValueError("stein_witness requires mass_conditioning=true")
    stein = value["stein"]
    keys(stein, "hidden_features hidden_layers activation witness_regularization potential_center_strength data_tail_fraction closure_samples closure_sigma closure_mass_bins ensemble_fit_selection scoring", "Stein residual")
    integer(stein["hidden_features"], "stein.hidden_features", 4)
    integer(stein["hidden_layers"], "stein.hidden_layers", 1)
    if stein["activation"] != "silu":
        raise ValueError("stein.activation must be silu")
    number(stein["witness_regularization"], "stein.witness_regularization")
    number(stein["potential_center_strength"], "stein.potential_center_strength", strict=False)
    if value["core"] == "stein_witness" and stein["potential_center_strength"] != 0:
        raise ValueError("stein_witness requires potential_center_strength=0")
    number(stein["data_tail_fraction"], "stein.data_tail_fraction", maximum=1.00000001)
    if stein["data_tail_fraction"] > 1:
        raise ValueError("stein.data_tail_fraction must be at most 1")
    integer(stein["closure_samples"], "stein.closure_samples", 256)
    number(stein["closure_sigma"], "stein.closure_sigma")
    integer(stein["closure_mass_bins"], "stein.closure_mass_bins", 1)
    if stein["ensemble_fit_selection"] not in ("validation-best", "all-valid"):
        raise ValueError("stein.ensemble_fit_selection must be validation-best or all-valid")
    scoring = stein["scoring"]
    scoring_defaults = stein_defaults["scoring"]
    if not isinstance(scoring, dict) or set(scoring) != set(scoring_defaults):
        raise ValueError("Invalid Stein scoring settings")
    if scoring["mode"] not in ("potential_raw", "potential_qnorm", "local_qnorm", "hybrid", "hybrid_gated", "sic_preserving", "tail_focus"):
        raise ValueError("Invalid stein.scoring.mode")
    integer(scoring["reference_samples"], "stein.scoring.reference_samples", 1024)
    number(scoring["reference_split"], "stein.scoring.reference_split", maximum=1)
    if not 0 < scoring["reference_split"] < 1:
        raise ValueError("stein.scoring.reference_split must lie strictly between zero and one")
    integer(scoring["mass_bins"], "stein.scoring.mass_bins", 1)
    number(scoring["energy_weight"], "stein.scoring.energy_weight", strict=False)
    number(scoring["operator_weight"], "stein.scoring.operator_weight", strict=False)
    if type(scoring["operator_gate_z"]) not in (int, float) or not math.isfinite(scoring["operator_gate_z"]):
        raise ValueError("Invalid stein.scoring.operator_gate_z")
    number(scoring["operator_temperature"], "stein.scoring.operator_temperature")
    number(scoring["beta"], "stein.scoring.beta", strict=False)
    number(scoring["local_temperature"], "stein.scoring.local_temperature")
    if type(scoring["local_gate_z"]) not in (int, float) or not math.isfinite(scoring["local_gate_z"]):
        raise ValueError("Invalid stein.scoring.local_gate_z")
    integer(scoring["final_mass_bins"], "stein.scoring.final_mass_bins", 1)
    if scoring["final_transform"] not in ("identity", "background_cdf", "background_cdf_power"):
        raise ValueError("Invalid stein.scoring.final_transform")
    number(scoring["final_power"], "stein.scoring.final_power")
    integer(scoring["qscore_batch_size"], "stein.scoring.qscore_batch_size", 1)
    integer(scoring["inference_batch_size"], "stein.scoring.inference_batch_size", 1)
    support_guard = scoring["support_guard"]
    support_defaults = scoring_defaults["support_guard"]
    if not isinstance(support_guard, dict) or set(support_guard) != set(support_defaults):
        raise ValueError("Invalid Stein support guard settings")
    if type(support_guard["enabled"]) is not bool:
        raise ValueError("stein.scoring.support_guard.enabled must be boolean")
    if support_guard["statistic"] != "radius":
        raise ValueError("stein.scoring.support_guard.statistic must be radius")
    integer(support_guard["mass_bins"], "stein.scoring.support_guard.mass_bins", 1)
    number(support_guard["gate_quantile"], "stein.scoring.support_guard.gate_quantile", maximum=1)
    number(support_guard["weight"], "stein.scoring.support_guard.weight", strict=False)
    number(support_guard["temperature"], "stein.scoring.support_guard.temperature")
    if support_guard["enabled"] and scoring["mode"] not in ("tail_focus", "potential_qnorm"):
        raise ValueError("stein.scoring.support_guard.enabled requires tail_focus or potential_qnorm")
    auto = scoring["auto_switch"]
    if (not isinstance(auto["enabled"], dict) or set(auto["enabled"]) != {"riddle", "iad", "supervised"}
            or any(type(v) is not bool for v in auto["enabled"].values())):
        raise ValueError("Auto-switch requires independent boolean riddle/iad/supervised switches")
    if (not isinstance(auto["fallback"], dict) or set(auto["fallback"]) != {"riddle", "iad", "supervised"}
            or any(v not in ("tail_focus", "potential_qnorm") for v in auto["fallback"].values())):
        raise ValueError("Auto-switch requires a PEW or potential-qnorm fallback for each method")
    integer(auto["reference_c_samples"], "auto-switch reference C samples", 512)
    efficiencies, weights = auto["efficiencies"], auto["weights"]
    if (not isinstance(efficiencies, list) or not isinstance(weights, list) or not efficiencies
            or len(efficiencies) != len(weights)):
        raise ValueError("Auto-switch efficiencies and weights must be aligned nonempty lists")
    for v in efficiencies:
        number(v, "auto-switch efficiency", maximum=1)
    if efficiencies != sorted(set(efficiencies)):
        raise ValueError("Auto-switch efficiencies must be unique and increasing")
    for v in weights:
        number(v, "auto-switch weight")
    if not math.isclose(sum(weights), 1, abs_tol=1e-12):
        raise ValueError("Auto-switch weights must sum to one")
    number(auto["confidence"], "auto-switch confidence", minimum=0.5, maximum=1)
    integer(auto["bootstrap_replicas"], "auto-switch bootstrap replicas", 100)
    integer(auto["min_tail_events"], "auto-switch minimum tail events", 1)
    integer(auto["seed"], "auto-switch seed", 0)
    if auto["seed"] >= 2**32:
        raise ValueError("Auto-switch seed exceeds NumPy seed range")
    if any(auto["enabled"].values()) and (value["core"] != "stein_witness" or scoring["mode"] not in ("tail_focus", "potential_qnorm")):
        raise ValueError("Auto-switch requires the Stein PEW or potential-qnorm score")
    if value["core"] == "stein_witness" and value.get("input_space") == "physical":
        raise ValueError("stein_witness is defined for mapped latent-space RIDDLE")

    mf = value["mass_fraction"]
    mf.setdefault("update_mode", "mean_damped")
    keys(mf, "enabled control_points smoothness variation damping min_fraction max_fraction max_iterations update_mode", "mass fraction")
    if mf["update_mode"] not in ("mean_matched", "mean_damped"):
        raise ValueError("mass_fraction.update_mode must be mean_matched or mean_damped")
    if type(mf["enabled"]) is not bool:
        raise ValueError("mass_fraction.enabled must be boolean")
    if value["core"] == "stein_witness":
        mf["enabled"] = False
    integer(mf["control_points"], "mass_fraction.control_points", 3)
    if mf["control_points"] % 2 == 0:
        raise ValueError("mass_fraction.control_points must be odd so one control point is at the SR center")
    number(mf["smoothness"], "mass_fraction.smoothness", strict=False)
    number(mf["variation"], "mass_fraction.variation", strict=False)
    number(mf["damping"], "mass_fraction.damping", maximum=1.00000001)
    if mf["damping"] > 1:
        raise ValueError("mass_fraction.damping must be at most 1")
    number(mf["min_fraction"], "mass_fraction.min_fraction", maximum=1)
    number(mf["max_fraction"], "mass_fraction.max_fraction", maximum=1)
    if mf["min_fraction"] >= mf["max_fraction"]:
        raise ValueError("mass_fraction min_fraction must be smaller than max_fraction")
    integer(mf["max_iterations"], "mass_fraction.max_iterations", 10)
    if mf["enabled"]:
        if not value.get("mass_conditioning"):
            raise ValueError("The v5.3 f(m) method requires mass_conditioning=true")
        if not e["guided_fit"]:
            raise ValueError("The v5.3 f(m) method requires guided_fit=true")
        if value.get("input_space") == "physical":
            raise ValueError("The v5.3 f(m) method is defined for mapped latent-space RIDDLE")
    correction = value.get("background_correction", "none")
    if correction not in ("none", "bgcorr_40_reguide"):
        raise ValueError("background_correction must be none or bgcorr_40_reguide")
    value["background_correction"] = correction
    value["sr_closure"] = sr_closure_mode(value.get("sr_closure", "auto"))
    if correction == "bgcorr_40_reguide":
        if not value.get("mass_conditioning"):
            raise ValueError("bgcorr_40_reguide requires mass_conditioning=true")
        if value["core"] == "residual" and (not e["guided_fit"] or not e["contrastive_fit"]):
            raise ValueError("bgcorr_40_reguide requires guided_fit and contrastive_fit for residual")
        if e["score_flow"]:
            raise ValueError("bgcorr_40_reguide uses the q_phi density ratio directly; disable score_flow")
        if value.get("input_space") == "physical":
            raise ValueError("bgcorr_40_reguide is defined in mapped latent space, not physical-input mode")
    if "input_space" in value and (value["input_space"] != "physical" or not value.get("mass_conditioning")):
        raise ValueError("input_space is only supported for the physical, mass-conditioned pilot")
    recovery = value["fit_recovery"]
    keys(recovery, "max_retries validation_sigma", "fit recovery")
    integer(recovery["max_retries"], "max_retries", 0)
    number(recovery["validation_sigma"], "validation_sigma", strict=False)
    integer(value["runs"], "fits")
    integer(value["epochs"], "epochs")
    allowed = ("identity", "random") if value.get("input_space") == "physical" else ("background", "random")
    if value["initialization"] not in allowed:
        raise ValueError("Initialization must be " + " or ".join(allowed))
    fractions = value["fractions"]
    if not isinstance(fractions, list) or not fractions:
        raise ValueError("Provide at least one mixture-fraction configuration")
    normalized = []
    for fraction in fractions:
        if fraction == "learned":
            normalized.append("learned")
        else:
            if isinstance(fraction, bool):
                raise ValueError("Invalid mixture fraction")
            fraction = float(fraction)
            number(fraction, "mixture fraction", maximum=1)
            normalized.append(fraction)
    if len(set(normalized)) != len(normalized):
        raise ValueError("Duplicate mixture fractions")
    if value["core"] == "stein_witness" and normalized != ["learned"]:
        raise ValueError("stein_witness uses no mixture fraction; keep fractions: [learned]")
    if value["core"] == "stein_witness" and e["checkpoint_weighting"] != "uniform":
        raise ValueError("stein_witness requires checkpoint_weighting=uniform")
    value["fractions"] = normalized
    f, t = value["flow"], value["training"]
    from .roles import validate_replay_batch
    validate_replay_batch(value["data_policy"], t["batch_size"])
    keys(
        f,
        "layers hidden_features num_blocks use_residual_blocks use_batch_norm dropout_probability activation random_mask num_bins tails tail_bound min_bin_width min_bin_height min_derivative",
        "residual flow",
    )
    keys(
        t,
        "batch_size validation_batch_size learning_rate weight_decay gradient_clip_norm selected_checkpoints",
        "residual training",
    )
    for k in ("layers", "hidden_features", "num_blocks", "num_bins"):
        integer(f[k], k)
    if (value["core"] == "residual" and value.get("background_correction") == "bgcorr_40_reguide"
            and e["qphi_hidden_features"] > f["hidden_features"]):
        raise ValueError("qphi_hidden_features cannot exceed residual flow hidden_features")
    for k in ("use_residual_blocks", "use_batch_norm", "random_mask"):
        if type(f[k]) is not bool:
            raise ValueError(f"{k} must be boolean")
    if f["use_residual_blocks"] and f["random_mask"]:
        raise ValueError("nflows residual blocks do not support random masks")
    if f["activation"] != "leaky_relu" or f["tails"] != "linear":
        raise ValueError("RIDDLE uses leaky_relu and linear spline tails")
    number(f["dropout_probability"], "dropout", maximum=1, strict=False)
    for k in ("tail_bound", "min_bin_width", "min_bin_height", "min_derivative"):
        number(f[k], k)
    if max(f["min_bin_width"], f["min_bin_height"]) * f["num_bins"] >= 1 or f["min_derivative"] >= 1:
        raise ValueError("Invalid spline bounds for background-matched initialization")
    for k in ("batch_size", "validation_batch_size", "selected_checkpoints"):
        integer(t[k], k, 2 if k == "batch_size" else 1)
    for k in ("learning_rate", "gradient_clip_norm"):
        number(t[k], k)
    number(t["weight_decay"], "weight_decay", strict=False)
    if value["epochs"] < t["selected_checkpoints"]:
        raise ValueError("Epochs must cover the number of selected checkpoints")
    if (value["core"] == "residual" and e["guided_fit"]
            and e["guide_warmup_epochs"] + t["selected_checkpoints"] > value["epochs"]):
        raise ValueError("Require enough post-guidance epochs for checkpoint selection")
    if value.get("input_space") == "physical" and any(e[k] for k in FEATURES):
        raise ValueError("Physical-input pilot requires all six RIDDLE enhancements disabled")
    if value.get("mass_conditioning") and e["score_flow"]:
        raise ValueError(
            "Mass-conditioned residual scoring is currently signal-region scoped; disable score_flow "
            "until a sideband-trained conditional-score calibration is defined"
        )
    if "optimization" in value:
        defaults = dict(initial_fraction=None, fraction_warmup_epochs=0,
                        lr_factor=1.0, lr_patience=10, min_lr=1e-5,
                        min_delta=1e-5, early_stopping_patience=0, minimum_epochs=40)
        supplied = value["optimization"]
        if not isinstance(supplied, dict) or set(supplied) - defaults.keys():
            raise ValueError("Invalid residual optimization settings")
        o = value["optimization"] = {**defaults, **supplied}
        if o["initial_fraction"] is not None:
            number(o["initial_fraction"], "initial_fraction", maximum=1)
        for name in ("fraction_warmup_epochs", "early_stopping_patience"):
            integer(o[name], name, 0)
        for name in ("lr_patience", "minimum_epochs"):
            integer(o[name], name)
        number(o["lr_factor"], "lr_factor", maximum=1.00000001)
        if o["lr_factor"] > 1:
            raise ValueError("lr_factor must be at most 1")
        number(o["min_lr"], "min_lr")
        number(o["min_delta"], "min_delta", strict=False)
        if o["lr_factor"] < 1 and o["min_lr"] > t["learning_rate"]:
            raise ValueError("min_lr exceeds the starting learning rate")
        if (value["core"] == "residual"
                and o["fraction_warmup_epochs"] + t["selected_checkpoints"] > value["epochs"]):
            raise ValueError("Require enough post-warm-up epochs for checkpoint selection")
        effective_warmup = (0 if value["core"] == "stein_witness" else
                            e["guide_warmup_epochs"] if e["guided_fit"] else o["fraction_warmup_epochs"])
        if o["early_stopping_patience"] and not (
            effective_warmup + t["selected_checkpoints"] <= o["minimum_epochs"] <= value["epochs"]
        ):
            raise ValueError("Invalid minimum_epochs for early stopping")
    return value



def load_settings(path=DEFAULT_PATH):
    path = Path(path).resolve()
    value = yaml.safe_load(path.read_text())
    keys(value, "schema inputs riddle background injection_scan", "top-level")
    if type(value["schema"]) is not int or value["schema"] != 1:
        raise ValueError("Unsupported settings schema")
    inputs = value["inputs"]
    keys(inputs, "features deltaR_features mass_column label_column", "input")
    from .features import FEATURES

    if inputs["features"] != list(FEATURES) or inputs["deltaR_features"] != ["deltaR"]:
        raise ValueError(
            "Input features must match the ordered LHCO schema: m1, delta_m, tau21_j1, tau21_j2; the DeltaR control adds deltaR"
        )
    if inputs["mass_column"] != "mjj" or inputs["label_column"] != "label":
        raise ValueError("Keep mjj and label separate from the input features")
    value["riddle"] = validate_residual(value["riddle"])
    b = value["background"]
    background_defaults = {"mapping_validation_batch_size": 65536, "mapping_inference_batch_size": 65536}
    for name, default in background_defaults.items():
        b.setdefault(name, default)
    keys(b, "configuration epochs batch_size reference_samples mapping_validation_batch_size mapping_inference_batch_size", "background")
    from .options import effective_features
    enhanced = any(effective_features(value["riddle"]).values()) or value["riddle"].get("background_correction") == "bgcorr_40_reguide"
    integer(b["epochs"], "background epochs", 10 if enhanced else 11)
    integer(b["batch_size"], "background batch_size", 2)
    integer(b["reference_samples"], "reference_samples", 2)
    integer(b["mapping_validation_batch_size"], "mapping_validation_batch_size", 1)
    integer(b["mapping_inference_batch_size"], "mapping_inference_batch_size", 1)
    if not isinstance(b["configuration"], dict):
        raise ValueError("Background configuration must be a nested YAML mapping")
    keys(
        b["configuration"],
        "ModelType Transform num_inputs num_cond_inputs num_blocks num_hidden activation_function pre_exp_tanh batch_norm batch_norm_momentum optimizer"
        + (" affine_log_scale_bound" if "affine_log_scale_bound" in b["configuration"] else ""),
        "background flow",
    )
    if "affine_log_scale_bound" in b["configuration"]:
        number(b["configuration"]["affine_log_scale_bound"], "affine_log_scale_bound")
        if b["configuration"]["ModelType"] != "MAF" or b["configuration"]["Transform"] != "Affine":
            raise ValueError("affine_log_scale_bound requires an affine MAF")
    if b["configuration"]["num_inputs"] != 4:
        raise ValueError("Keep num_inputs=4; the DeltaR control automatically selects five dimensions")
    scan = value["injection_scan"]
    scan.setdefault("background_mode", "reuse")
    keys(scan, "signal_events seeds background_mode", "injection scan")
    if scan["background_mode"] not in ("reuse", "retrain"):
        raise ValueError("injection_scan.background_mode must be reuse or retrain")
    if not isinstance(scan["signal_events"], list) or not scan["signal_events"]:
        raise ValueError("Provide injection strengths as positive total signal counts")
    from .data_spec import DatasetSpec
    for n in scan["signal_events"]:
        integer(n, "injected signal count", 3)
        if n >= DatasetSpec().signal_rows:
            raise ValueError("Reserve uninjected signal for independent evaluation")
    if len(set(scan["signal_events"])) != len(scan["signal_events"]):
        raise ValueError("Duplicate injection strengths")
    if not isinstance(scan["seeds"], list) or not scan["seeds"]:
        raise ValueError("Provide explicit injection-scan training seeds")
    for seed in scan["seeds"]:
        integer(seed, "scan seed", 0)
        if seed >= 2**32:
            raise ValueError("Scan seeds exceed the NumPy seed range")
    if len(set(scan["seeds"])) != len(scan["seeds"]):
        raise ValueError("Duplicate injection-scan seeds")
    return value


def input_features(settings, manifest):
    from .controls import VARIANTS

    variant = manifest.get("variant", "default")
    if variant not in VARIANTS:
        raise ValueError("Unknown LHCO dataset variant")
    inputs = settings["inputs"]
    features = list(inputs["features"])
    if variant == "deltaR":
        features.extend(inputs["deltaR_features"])
    expected = [inputs["mass_column"], *features, inputs["label_column"]]
    if manifest.get("columns", expected) != expected:
        raise ValueError("Prepared input columns disagree with settings.yaml")
    return features


DEFAULTS = load_settings()


def resolve(args):
    effective = load_settings(args.config)
    native_methods = any(method in args.methods for method in ("riddle", "iad", "supervised"))
    requested_fits = getattr(args, "fits", None)
    requested_epochs = getattr(args, "epochs", None)
    if native_methods:
        common_overrides = {
            "runs": requested_fits,
            "epochs": requested_epochs,
            "data_policy": getattr(args, "data_policy", None),
            "ensemble_completion": getattr(args, "ensemble_completion", None),
            "fractions": getattr(args, "fractions", None),
            "mass_conditioning": getattr(args, "mass_conditioning", None),
            "background_correction": getattr(args, "background_correction", None),
        }
        for key, override in common_overrides.items():
            if override is not None:
                effective["riddle"][key] = override
        from .options import FEATURES
        for key in FEATURES:
            override = getattr(args, key, None)
            if override is not None:
                effective["riddle"]["enhancements"][key] = override
        effective["riddle"] = validate_residual(effective["riddle"])
        args.fits = effective["riddle"]["runs"]
        args.epochs = effective["riddle"]["epochs"]
        args.fractions = effective["riddle"]["fractions"]
    args.settings = effective
    return args
