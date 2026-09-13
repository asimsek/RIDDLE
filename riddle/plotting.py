import argparse
from contextlib import contextmanager
import csv
import json
from pathlib import Path

import numpy as np
import yaml
from scipy.interpolate import interp1d
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from . import figures as f
from .storage import atomic_write, file_digest, locked, write_json
from .progress import set_verbosity, colored_status
from .worker_progress import ProgressStage, local_progress

KEYS = {"lacathode": "raw", "riddle": "residual"}
STYLES = dict(f.METHODS)


class PopulationMismatch(ValueError):
    pass


@contextmanager
def methods(keys):
    previous = f.METHODS, f.VIEWS
    f.METHODS = {k: STYLES[k] for k in keys}
    f.VIEWS = {"comparison" if len(keys) > 1 else STYLES[keys[0]][0]: tuple(keys)}
    try:
        yield
    finally:
        f.METHODS, f.VIEWS = previous


def discover(root, requested):
    paths = [root / "result.json"] if (root / "result.json").is_file() else sorted(root.rglob("result.json"))
    groups = {}
    variants = set()
    for path in paths:
        report = json.loads(path.read_text())
        method = report.get("method")
        if method not in requested or not report.get("completed"):
            continue
        variants.add(
            report.get("variant", report.get("contract", {}).get("inputs", {}).get("variant", "default"))
        )
        if len(variants) > 1:
            raise ValueError("Plot each dataset variant separately; do not mix controls in one comparison")
        scenario, seed = report["scenario"], report["seed"]
        if scenario not in f.SCENARIOS or type(seed) is not int:
            raise ValueError("Invalid result identity")
        group = groups.setdefault((scenario, seed), {})
        if method in group:
            raise ValueError("Duplicate method/scenario/seed results; choose a narrower input directory")
        group[method] = (path.parent, report)
    return groups


def load_scores(root, report, name):
    path = root / (name + "_scores.npz")
    expected = report.get("artifacts_sha256", {}).get(path.name)
    if not expected or file_digest(path) != expected:
        raise ValueError("Score artifact is missing or changed")
    with np.load(path, allow_pickle=False) as archive:
        data = {k: archive[k] for k in ("mass", "labels", "mask", "scores", "physical", "latent")}
    n = len(data["mass"])
    variant = report.get("variant", report.get("contract", {}).get("inputs", {}).get("variant", "default"))
    dimensions = 5 if variant == "deltaR" else 4
    if any(data[k].shape != (n,) for k in ("mass", "labels", "scores", "mask")):
        raise ValueError("Misaligned score arrays")
    if (
        data["mask"].dtype != bool
        or data["physical"].shape != (n, dimensions)
        or data["latent"].shape != (data["mask"].sum(), dimensions)
    ):
        raise ValueError("Invalid feature/mapping shapes")
    if not np.isin(data["labels"], [0, 1]).all() or not all(
        np.isfinite(data[k]).all() for k in ("mass", "physical", "latent")
    ):
        raise ValueError("Invalid event features or labels")
    if (
        not np.isfinite(data["scores"][data["mask"]]).all()
        or np.isfinite(data["scores"][~data["mask"]]).any()
    ):
        raise ValueError("Scores disagree with the mapping mask")
    return data


def make_bundle(group, confidence):
    keys = [KEYS[m] for m in group]
    bundle = {"samples": {}, "curves": {}, "warnings": [], "sources": group, "latents_by_method": {}}
    for partition in ("validation", "test", "signal_region"):
        records = {KEYS[m]: load_scores(root, report, partition) for m, (root, report) in group.items()}
        base = records[keys[0]]
        for record in records.values():
            if any(not np.array_equal(base[k], record[k]) for k in ("mass", "labels", "mask", "physical")):
                raise PopulationMismatch(
                    "Methods have different evaluation populations or mapping masks; plot them separately"
                )
        sample = {k: base[k] for k in ("mass", "labels", "mask")}
        sample.update({k + "_scores": r["scores"] for k, r in records.items()})
        if partition == "signal_region":
            bundle["curves"]["signal_region"] = (
                base["labels"][base["mask"]],
                {k: r["scores"][base["mask"]] for k, r in records.items()},
            )
        else:
            bundle["samples"][partition] = sample
        if partition == "test":
            bundle["feature_values"] = np.column_stack((base["mass"], base["physical"]))
            bundle["latents_by_method"] = {k: r["latent"] for k, r in records.items()}
    report = next(iter(group.values()))[1]
    bundle["report"] = report
    bundle["variant"] = report.get(
        "variant", report.get("contract", {}).get("inputs", {}).get("variant", "default")
    )
    with methods(keys):
        f.build_metrics(bundle, confidence)
    return bundle


def bundles_for_group(group, confidence):
    try:
        return [(None, make_bundle(group, confidence))]
    except PopulationMismatch as error:
        colored_status("Evaluation populations differ; producing separate method figures", kind="WARNING")
        result = []
        for method, source in group.items():
            bundle = make_bundle({method: source}, confidence)
            bundle["individual"] = method
            bundle["warnings"].append(str(error))
            result.append((method, bundle))
        return result


def render_bundle(bundle, target, args):
    group = bundle["sources"]
    keys = [KEYS[m] for m in group]
    with methods(keys):
        for renderer in (
            f.render_roc,
            f.render_scores,
            f.render_efficiency,
            f.render_mass,
            f.render_features,
        ):
            with ProgressStage(renderer.__name__, renderer.__name__.replace("render_", "Plot ")):
                renderer(bundle, target, args) if renderer in (
                    f.render_roc,
                    f.render_efficiency,
                ) else renderer(bundle, target)
    for key in keys:
        with methods([key]), ProgressStage("representation", "Plot input and latent distributions"):
            f.render_representation({**bundle, "latents": bundle["latents_by_method"][key]}, target)
    training_figures(group, target)
    write_json(target / "metrics.json", json_safe(bundle["metrics"]))


def comparison_rows(bundle, args):
    rows = []
    scenario, seed = bundle["report"]["scenario"], bundle["report"]["seed"]

    def add(metric, values, selection="all", scope="signal_region", better="higher", ci=None):
        row = {
            "scenario": scenario,
            "variant": bundle.get("variant", "default"),
            "seed": seed,
            "scope": scope,
            "selection": selection,
            "metric": metric,
            "better": better,
        }
        for key in STYLES:
            name = STYLES[key][0]
            row[name] = f.scalar(values.get(key))
            row[name + "_ci_low"], row[name + "_ci_high"] = (ci or {}).get(key, (None, None))
        a, b = row["LaCathode"], row["RIDDLE"]
        row["RIDDLE_minus_LaCathode"] = (
            b - a if a is not None and b is not None and better != "not_comparable" else None
        )
        row["confidence_level"] = args.confidence if ci else None
        rows.append(row)

    for dataset, (labels, scores) in bundle["curves"].items():
        if len(np.unique(labels)) != 2:
            continue
        parts = {k: f.auc_components(labels, s) for k, s in scores.items()}
        add(
            "auc",
            {k: roc_auc_score(labels, s) for k, s in scores.items()},
            ci={k: f.auc_interval(p, args.confidence) for k, p in parts.items()},
        )
        if set(parts) == set(STYLES):
            low, high = f.paired_auc_interval(parts, args.confidence)
            rows[-1].update(RIDDLE_minus_LaCathode_ci_low=low, RIDDLE_minus_LaCathode_ci_high=high)
        add("average_precision", {k: average_precision_score(labels, s) for k, s in scores.items()})
        maxima = {}
        for key, score in scores.items():
            b, s, _ = roc_curve(labels, score, drop_intermediate=False)
            good = (b >= 1e-4) & (np.rint(b * (labels == 0).sum()) >= args.min_background)
            maxima[key] = np.max(s[good] / np.sqrt(b[good])) if good.any() else None
        add("max_sic_supported", maxima)
    for point in bundle["metrics"]["working_points"]:
        records = point["methods"]
        selection = ("<" if point["name"] == "extra_tight" else "<=") + str(
            point["validation_background_budget"]
        )
        for metric, better in (
            ("signal", "higher"),
            ("background", "lower"),
            ("signal_passed", "higher"),
            ("background_passed", "lower"),
            ("signal_total", "not_comparable"),
            ("background_total", "not_comparable"),
            ("relative_mass_rms", "lower"),
            ("chi2_ndof", "near_one"),
            ("cut", "not_comparable"),
        ):
            ci = (
                {k: (r.get(metric + "_ci_low"), r.get(metric + "_ci_high")) for k, r in records.items()}
                if metric in ("signal", "background")
                else None
            )
            add(
                metric,
                {k: r.get(metric) for k, r in records.items()},
                selection,
                "full_mass_range" if metric in ("relative_mass_rms", "chi2_ndof") else "physical_test_sr",
                better,
                ci,
            )
        add(
            "background_rejection",
            {k: f.safe_div(1, r.get("background")) for k, r in records.items()},
            selection,
            "physical_test_sr",
        )
        add(
            "sic",
            {
                k: f.safe_div(r.get("signal"), np.sqrt(r["background"]) if r.get("background") else None)
                for k, r in records.items()
            },
            selection,
            "physical_test_sr",
        )
    return rows


def csv_write(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))

    def writer(temp):
        with temp.open("w", newline="") as stream:
            out = csv.DictWriter(stream, fieldnames=fields)
            out.writeheader()
            out.writerows(rows)

    atomic_write(path, writer)


def settings_rows(group):
    rows = []

    def flatten(value, prefix=""):
        if isinstance(value, dict):
            for key, item in value.items():
                yield from flatten(item, (prefix + "." if prefix else "") + str(key))
        else:
            yield prefix, json.dumps(value) if isinstance(value, (list, tuple)) else value

    for method, (root, report) in group.items():
        protocol = read_metadata(root, report, "protocol.json")
        if method == "lacathode":
            protocol["configuration"] = {
                k: yaml.safe_load(v) for k, v in protocol.get("configuration", {}).items()
            }
        payload = {
            "settings": report["contract"]["settings"],
            "environment": report["contract"]["environment"],
            "protocol": protocol,
            "preparation": report["contract"].get("inputs", {}).get("preparation", {}),
        }
        if method == "riddle":
            result = read_metadata(root, report, "density/ensemble_selection.json")
            payload["ensemble"] = {
                k: result[k]
                for k in (
                    "selected_configuration",
                    "requested_runs",
                    "valid_runs",
                    "selected_checkpoints",
                    "failures",
                    "members",
                )
            }
        for key, value in flatten(payload):
            rows.append(
                {
                    "method": STYLES[KEYS[method]][0],
                    "scenario": report["scenario"],
                    "variant": report.get("variant", "default"),
                    "seed": report["seed"],
                    "setting": key,
                    "value": value,
                }
            )
    return rows


def training_figures(group, output):
    for method, (root, report) in group.items():
        destination = output / STYLES[KEYS[method]][0] / "02_training"
        if method == "riddle":
            history = read_metadata(root, report, "density/residual_losses.json")["history"]
            for kind, ylabel in (
                ("nll", "Mixture negative log likelihood"),
                ("fraction", "Fitted mixture fraction"),
            ):
                fig, ax, _ = f.canvas(ylabel, "Epoch")
                x = [r["epoch"] + 1 for r in history]
                fields = (
                    (("train_nll", "Train"), ("validation_nll", "Validation"))
                    if kind == "nll"
                    else (("signal_fraction", "RIDDLE"),)
                )
                for field, label in fields:
                    ax.plot(x, [r[field] for r in history], label=label)
                f.legend(fig)
                f.save(fig, destination / kind)
            members = read_metadata(root, report, "density/ensemble_selection.json")["members"]
            if members:
                fig, ax, _ = f.canvas("Fit", "Selected epoch")
                for i, member in enumerate(members, 1):
                    epochs = np.asarray(member["epochs"]) + 1
                    ax.scatter(
                        epochs,
                        np.full(len(epochs), i),
                        s=12,
                        color=STYLES["residual"][1],
                        label="RIDDLE" if i == 1 else None,
                    )
                ax.set(yticks=range(1, len(members) + 1), ylim=(0.3, len(members) + 0.7))
                f.legend(fig)
                f.save(fig, destination / "selected_epochs_by_fit")
                fig, ax, _ = f.canvas("Validation mixture NLL", "Fit")
                for i, member in enumerate(members, 1):
                    history = read_metadata(
                        root, report, "density/" + member["directory"] + "/residual_losses.json"
                    )["history"]
                    ax.plot(
                        [i],
                        [history[-1]["validation_nll"]],
                        "x",
                        color=".6",
                        label="Final epoch" if i == 1 else None,
                    )
                    ax.scatter(
                        np.full(len(member["epochs"]), i),
                        [history[e]["validation_nll"] for e in member["epochs"]],
                        s=12,
                        color=STYLES["residual"][1],
                        label="Selected" if i == 1 else None,
                    )
                ax.set(xticks=range(1, len(members) + 1))
                f.legend(fig)
                f.save(fig, destination / "selected_vs_final_validation")
        for stage in ("background", "classifier") if method == "lacathode" else ("background",):
            directory = root / ("training" if method == "lacathode" else "background")
            names = (
                ("my_ANODE_model_train_losses.npy", "my_ANODE_model_val_losses.npy")
                if stage == "background"
                else ("loss_matris.npy", "val_loss_matris.npy")
            )
            if not all((directory / name).is_file() for name in names):
                continue
            fig, ax, _ = f.canvas(
                "Negative log likelihood" if stage == "background" else "Classification loss", "Epoch"
            )
            for name, label in zip(names, ("Train", "Validation")):
                verify_plot_input(root, report, str((directory / name).relative_to(root)))
                values = np.load(directory / name).squeeze()
                ax.plot(np.arange(len(values)), values, label=label)
            f.legend(fig)
            f.save(fig, destination / stage)


def verify_plot_input(root, report, name):
    path = (root / name).resolve()
    checksum = report.get("artifacts_sha256", {}).get(name)
    if not path.is_relative_to(root.resolve()) or not checksum or file_digest(path) != checksum:
        raise ValueError("Plot input is missing or changed")
    return path


def read_metadata(root, report, name):
    return json.loads(verify_plot_input(root, report, name).read_text())


def summary_figures(bundles, output, args):
    audit = {}
    for scenario in f.SCENARIOS:
        cohort = [b for b in bundles if b["report"]["scenario"] == scenario]
        if not cohort:
            continue
        fig, ax, _ = f.canvas("Significance improvement", "Background efficiency")
        any_curve = False
        for key in STYLES:
            values = []
            for bundle in cohort:
                labels, scores = bundle["curves"]["signal_region"]
                if key not in scores or len(np.unique(labels)) != 2:
                    continue
                b, s, _ = roc_curve(labels, scores[key])
                supported = b > 1.6173483250740743e-5
                if supported.sum() < 2:
                    continue
                values.append(
                    interp1d(
                        b[supported],
                        s[supported] / np.sqrt(b[supported]),
                        bounds_error=False,
                        fill_value=np.nan,
                    )(f.GRID)
                )
            if values:
                name, color, ls = STYLES[key]
                audit[scenario + "/" + name + "/sic"] = f.draw_band(ax, f.GRID, values, name, color, ls)
                any_curve = True
        if any_curve:
            ax.plot(f.GRID, np.sqrt(f.GRID), color=".5", ls=":", label="Random")
            ax.set(xscale="log", xlim=(1e-4, 1), ylim=(0, None))
            f.legend(fig, title=f.SCENARIO_LABELS[scenario])
            f.save(fig, output / scenario / "signal_region_sic")
        else:
            f.plt.close(fig)
        if scenario != "background_only":
            continue
        fig, ax, _ = f.canvas(r"$\chi^2/n_{\mathrm{dof}}$", "SR selection efficiency")
        found = False
        for key in STYLES:
            curves = []
            for bundle in cohort:
                sample = bundle["samples"]["test"]
                if key + "_scores" not in sample:
                    continue
                pop = sample["labels"] == 0
                mass, score = sample["mass"][pop], sample[key + "_scores"][pop]
                mask = sample["mask"][pop]
                region = f.sr(mass) & mask
                if len(mass) < 300 or not region.any():
                    continue
                edges = f.equal_occupancy(mass)
                full = np.histogram(mass, edges)[0]
                if np.any(full == 0):
                    continue
                curves.append(
                    [
                        f.shape_chi2(
                            full,
                            np.histogram(mass[mask & (score > np.quantile(score[region], 1 - e))], edges)[0],
                            e,
                        )
                        for e in f.EFFICIENCIES
                    ]
                )
            if curves:
                name, color, ls = STYLES[key]
                audit[scenario + "/" + name + "/mass_flatness"] = f.draw_band(
                    ax, f.EFFICIENCIES, curves, name, color, ls
                )
                found = True
        if found:
            sample = cohort[0]["samples"]["test"]
            mass = sample["mass"][sample["labels"] == 0]
            audit[scenario + "/random/mass_flatness"] = f.draw_band(
                ax, f.EFFICIENCIES, random_reference(mass), "Random", ".5", ":"
            )
            ax.set(xlim=(0.205, 0), ylim=(0, None))
            f.legend(fig, title="BG-Only")
            f.save(fig, output / scenario / "mass_flatness_vs_selection")
        else:
            f.plt.close(fig)
    return audit


def random_reference(mass, trials=100):
    edges = f.equal_occupancy(mass)
    full = np.histogram(mass, edges)[0]
    bin_id = np.minimum(np.searchsorted(edges, mass, side="right") - 1, len(full) - 1)
    result = []
    for efficiency in f.EFFICIENCIES:
        rng = np.random.RandomState(42)
        values = []
        for _ in range(trials):
            indices = rng.choice(len(mass), size=int(efficiency * len(mass)), replace=True)
            values.append(f.shape_chi2(full, np.bincount(bin_id[indices], minlength=len(full)), efficiency))
        result.append(values)
    return np.asarray(result).T


def main(argv=None):
    parser = argparse.ArgumentParser(description="Figures from frozen LaCathode/RIDDLE results; no fitting")
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, default=Path("plots"))
    parser.add_argument("--methods", nargs="+", choices=tuple(KEYS), default=list(KEYS))
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--min-background", type=int, default=10)
    parser.add_argument("--allow-smoke", action="store_true", help="Permit synthetic QA fixtures")
    parser.add_argument("--verbose", type=int, choices=(0, 1, 2), default=1)
    args = parser.parse_args(argv)
    args.population_summary = True
    set_verbosity(args.verbose)
    try:
        if not 0 < args.confidence < 1 or args.min_background < 1:
            raise ValueError("Invalid confidence level or background support")
        groups = discover(args.results, args.methods)
        if not groups:
            colored_status("No completed results for the requested methods", kind="WARNING")
            return 0
        args.results, args.output = args.results.resolve(), args.output.resolve()
        if args.output.exists():
            raise FileExistsError("Plot output exists; choose a new output directory")
        if args.output.is_relative_to(args.results) or args.results.is_relative_to(args.output):
            raise ValueError("Keep plots outside the result input tree")
        args.output.mkdir(parents=True)
        rows, settings, bundles = [], [], []
        with local_progress("Plots"), f.plt.style.context(f.STYLE), locked(args.output / ".plot.lock"):
            for (scenario, seed), group in groups.items():
                if (
                    any(
                        r["contract"].get("inputs", {}).get("synthetic_smoke_fixture")
                        for _, r in group.values()
                    )
                    and not args.allow_smoke
                ):
                    raise ValueError("Synthetic fixtures require --allow-smoke for QA")
                missing = set(args.methods) - set(group)
                if missing:
                    colored_status("Using available method; requested counterpart is absent", kind="INFO")
                target = args.output / scenario / f"seed_{seed:03d}"
                for individual, bundle in bundles_for_group(group, args.confidence):
                    rows.extend(comparison_rows(bundle, args))
                    settings.extend(settings_rows(bundle["sources"]))
                    render_bundle(bundle, target / individual if individual else target, args)
                    bundles.append(bundle)
            audit = summary_figures([b for b in bundles if not b.get("individual")], args.output, args)
            for method in KEYS:
                individual = [b for b in bundles if b.get("individual") == method]
                separate = summary_figures(individual, args.output / "individual" / method, args)
                audit.update({f"individual/{method}/{k}": v for k, v in separate.items()})
            csv_write(args.output / "comparison.csv", rows)
            csv_write(args.output / "configuration.csv", settings)
            write_json(
                args.output / "plot_manifest.json",
                json_safe(
                    {
                        "schema": 1,
                        "uncertainty": audit,
                        "cuts": "Validation-SR background thresholds; frozen cuts evaluated on independent physical test rows",
                        "mass_summary": "BG-Only: test-SR quantiles, 300 equal-occupancy full-range bins, normalized-shape Poisson chi2",
                        "uncertainty_scope": "SIC/mass bands: 16/50/84 percentiles across independent seeds; not internal fits",
                        "warnings": [w for b in bundles for w in b["warnings"]],
                    }
                ),
            )
        colored_status("Figures and CSV tables completed", kind="PASS")
        return 0
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, f"[ERROR] {error}\n")


def json_safe(value):
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return f.scalar(value)
    if isinstance(value, np.integer):
        return int(value)
    return value
