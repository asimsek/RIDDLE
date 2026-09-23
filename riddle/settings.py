from copy import deepcopy
import math
from pathlib import Path

import yaml

def default_config_path(name):
    """Locate bundled configuration files."""
    import sysconfig
    if name not in ("settings.yaml", "datasets.yaml"):
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


def validate_residual(value):
    value = deepcopy(value)
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
    for name in ("guide_reference_multiplier", "tail_candidate_multiplier", "qphi_mass_bins"):
        integer(e[name], name, 1)
    integer(e["qphi_epochs"], "qphi_epochs", 10)
    for name in ("guide_mass_conditioning", "guide_refresh_reference", "guide_ratio_calibration", "tail_rank"):
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
    # Older configuration files acquire the current, recorded recovery policy.
    value.setdefault("fit_recovery", {"max_retries": 2, "validation_sigma": 2.0})
    # production_v2 is the HC-derived production baseline: study mapping roles
    # with the normal production residual/member policy.
    from .roles import DEFAULT_POLICY
    value.setdefault("data_policy", DEFAULT_POLICY)
    # Production must never silently finalize fewer residual members than were
    # requested.  Diagnostics can still opt into ``partial`` explicitly.
    value.setdefault("ensemble_completion", "strict")
    # Optional pilot ablation; the resolved data policy is always explicit.
    extra = "".join(" " + k for k in ("mass_conditioning", "input_space", "optimization", "data_policy", "ensemble_completion", "background_correction") if k in value)
    keys(value, "runs epochs fractions initialization flow training fit_recovery enhancements" + extra, "RIDDLE")
    if "data_policy" in value:
        from .roles import policy_parts, DIAGNOSTIC_POLICIES
        policy_parts(value["data_policy"])
        if value["data_policy"] in DIAGNOSTIC_POLICIES and (value["runs"] != 1 or not all(e[k] for k in FEATURES)):
            raise ValueError("Diagnostic role replay requires one fit and all six features")
    if value["ensemble_completion"] not in ("strict", "partial"):
        raise ValueError("ensemble_completion must be strict or partial")
    if "mass_conditioning" in value and type(value["mass_conditioning"]) is not bool:
        raise ValueError("mass_conditioning must be boolean")
    correction = value.get("background_correction", "none")
    if correction not in ("none", "bgcorr_40_reguide"):
        raise ValueError("background_correction must be none or bgcorr_40_reguide")
    value["background_correction"] = correction
    if correction == "bgcorr_40_reguide":
        if not value.get("mass_conditioning"):
            raise ValueError("bgcorr_40_reguide requires mass_conditioning=true")
        if not e["guided_fit"] or not e["contrastive_fit"]:
            raise ValueError("bgcorr_40_reguide requires guided_fit and contrastive_fit")
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
    if e["guided_fit"] and e["guide_warmup_epochs"] + t["selected_checkpoints"] > value["epochs"]:
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
        if o["fraction_warmup_epochs"] + t["selected_checkpoints"] > value["epochs"]:
            raise ValueError("Require enough post-warm-up epochs for checkpoint selection")
        effective_warmup = e["guide_warmup_epochs"] if e["guided_fit"] else o["fraction_warmup_epochs"]
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
    keys(b, "configuration epochs batch_size reference_samples", "background")
    integer(b["epochs"], "background epochs", 11)
    integer(b["batch_size"], "background batch_size", 2)
    integer(b["reference_samples"], "reference_samples", 2)
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
    keys(scan, "signal_events replicas preparation_seed training_seed", "injection scan")
    if not isinstance(scan["signal_events"], list) or not scan["signal_events"]:
        raise ValueError("Provide injection strengths as positive total signal counts")
    from .data_spec import DatasetSpec
    for n in scan["signal_events"]:
        integer(n, "injected signal count", 3)
        if n >= DatasetSpec().signal_rows:
            raise ValueError("Reserve uninjected signal for independent evaluation")
    if len(set(scan["signal_events"])) != len(scan["signal_events"]):
        raise ValueError("Duplicate injection strengths")
    integer(scan["replicas"], "scan replicas")
    for key in ("preparation_seed", "training_seed"):
        integer(scan[key], key, 0)
        if scan[key] + scan["replicas"] >= 2**32:
            raise ValueError("Scan seeds exceed the NumPy seed range")
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
    if "riddle" in args.methods:
        from .options import FEATURES
        for key in FEATURES:
            override = getattr(args, key, None)
            if override is not None:
                effective["riddle"]["enhancements"][key] = override
        for key in ("runs", "epochs", "fractions", "data_policy", "ensemble_completion", "mass_conditioning", "background_correction"):
            override = getattr(args, "fits" if key == "runs" else key, None)
            if override is not None:
                effective["riddle"][key] = override
        effective["riddle"] = validate_residual(effective["riddle"])
    args.fits = effective["riddle"]["runs"]
    args.epochs = effective["riddle"]["epochs"]
    args.fractions = effective["riddle"]["fractions"]
    args.settings = effective
    return args
