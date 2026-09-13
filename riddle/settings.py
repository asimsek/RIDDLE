from copy import deepcopy
import math
from pathlib import Path

import yaml

DEFAULT_PATH = Path(__file__).resolve().parents[1] / "config/settings.yaml"


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
    keys(value, "runs epochs fractions initialization flow training", "RIDDLE")
    integer(value["runs"], "runs")
    integer(value["epochs"], "epochs")
    if value["initialization"] not in ("background", "random"):
        raise ValueError("Initialization must be background or random")
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
    return value


def load_settings(path=DEFAULT_PATH):
    path = Path(path).resolve()
    value = yaml.safe_load(path.read_text())
    keys(value, "schema inputs riddle background", "top-level")
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
        "ModelType Transform num_inputs num_cond_inputs num_blocks num_hidden activation_function pre_exp_tanh batch_norm batch_norm_momentum optimizer",
        "background flow",
    )
    if b["configuration"]["num_inputs"] != 4:
        raise ValueError("Keep num_inputs=4; the DeltaR control automatically selects five dimensions")
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
        for key in ("runs", "epochs", "fractions"):
            override = getattr(args, key, None)
            if override is not None:
                effective["riddle"][key] = override
        effective["riddle"] = validate_residual(effective["riddle"])
    args.runs = effective["riddle"]["runs"]
    args.epochs = effective["riddle"]["epochs"]
    args.fractions = effective["riddle"]["fractions"]
    args.settings = effective
    return args
