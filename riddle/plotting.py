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
from .metrics import acceptance_report, efficiency_curve, oracle_metrics

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


def discover(root, requested, *, scan=False):
    paths = [root / "result.json"] if (root / "result.json").is_file() else sorted(root.rglob("result.json"))
    groups = {}
    variants = set()
    versions = {}
    for path in paths:
        report = json.loads(path.read_text())
        method = report.get("method")
        if method not in requested or not report.get("completed"):
            continue
        point = report.get("contract", {}).get("inputs", {}).get("injection_scan")
        if bool(point) != scan:
            continue
        method_versions = versions.setdefault(method, set())
        method_versions.add(report.get("contract", {}).get("scientific_version", 1))
        if len(method_versions) > 1:
            raise ValueError(f"Do not combine different scientific protocols for {method}")
        variants.add(
            report.get("variant", report.get("contract", {}).get("inputs", {}).get("variant", "default"))
        )
        if len(variants) > 1:
            raise ValueError("Plot each dataset variant separately; do not mix controls in one comparison")
        scenario, seed = report["scenario"], report["seed"]
        if scenario not in f.SCENARIOS or type(seed) is not int:
            raise ValueError("Invalid result identity")
        identity = (point["signal_events"], point["replica"]) if scan else (scenario, seed)
        group = groups.setdefault(identity, {})
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
        if report["method"] == "riddle":
            from .production import validate_region

            if "is_signal_region" not in archive:
                raise ValueError("RIDDLE SR membership is missing; regenerate its score artifacts")
            data["is_signal_region"] = validate_region(archive["is_signal_region"], len(data["mass"]))
            if name == "signal_region" and not data["is_signal_region"].all():
                raise ValueError("RIDDLE signal-region evaluation contains non-SR events")
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
    bundle = {"samples": {}, "curves": {}, "warnings": [], "sources": group, "latents_by_method": {},
              "evaluation": {}, "acceptance": {}}
    for partition in ("validation", "test", "signal_region"):
        records = {KEYS[m]: load_scores(root, report, partition) for m, (root, report) in group.items()}
        bundle["evaluation"][partition] = records
        from .production import region_acceptance
        bundle["acceptance"][partition] = {
            k: (region_acceptance(r["labels"], r["mask"], r["is_signal_region"])
                if k == "residual" else acceptance_report(r["labels"], r["mask"], r["mass"]))
            for k, r in records.items()
        }
        base = records[keys[0]]
        for record in records.values():
            if any(not np.array_equal(base[k], record[k]) for k in ("mass", "labels", "mask", "physical")):
                raise PopulationMismatch(
                    "Methods have different evaluation populations or mapping masks; plot them separately"
                )
        sr_masks = {k: r["is_signal_region"] for k, r in records.items() if k == "residual"}
        if partition != "signal_region" and "residual" in sr_masks and "raw" in records:
            if not np.array_equal(sr_masks["residual"], f.sr(records["raw"]["mass"])):
                raise PopulationMismatch(
                    "Methods have different SR memberships; separate plots preserve LaCathode's unchanged definition"
                )
        sample = {k: base[k] for k in ("mass", "labels", "mask")}
        sample["sr_masks"] = sr_masks
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
            f.render_mass_scan,
            f.render_features,
        ):
            if renderer is f.render_mass_scan:
                renderer(bundle, target)  # This renderer reports its own 70-cut progress.
                continue
            with ProgressStage(renderer.__name__, renderer.__name__.replace("render_", "Plot ")):
                renderer(bundle, target, args) if renderer in (
                    f.render_roc,
                    f.render_efficiency,
                ) else renderer(bundle, target)
    for key in keys:
        with methods([key]), ProgressStage("representation", "Plot input and latent distributions"):
            f.render_representation({**bundle, "latents": bundle["latents_by_method"][key]}, target)
    training_figures(group, target)
    render_full_pipeline(bundle, target, args)
    bundle["metrics"]["mapping_acceptance"] = bundle["acceptance"]
    write_json(target / "metrics.json", json_safe(bundle["metrics"]))


def render_full_pipeline(bundle, output, args):
    records = bundle["evaluation"]["signal_region"]
    audit = {}
    for metric in ("roc", "sic"):
        fig, ax, _ = f.canvas("Signal efficiency" if metric == "roc" else "Significance improvement",
                             "Background efficiency")
        drawn = False
        for key, record in records.items():
            if len(np.unique(record["labels"][record["mask"]])) != 2:
                continue
            b, s, _ = efficiency_curve(record["labels"], record["scores"], record["mask"], full_pipeline=True)
            supported = (b >= 1e-4) & (np.rint(b * (record["labels"] == 0).sum()) >= args.min_background)
            keep = b > 0 if metric == "roc" else supported
            label, color, ls = STYLES[key]
            if keep.any():
                ax.plot(b[keep], s[keep] if metric == "roc" else s[keep]/np.sqrt(b[keep]),
                        color=color, ls=ls, label=label)
                drawn = True
            audit[label] = oracle_metrics(record["labels"], record["scores"], record["mask"],
                                          min_background=args.min_background)
        if drawn:
            ax.set(xscale="log", xlim=(1e-4, 1), ylim=(0, None))
            f.legend(fig, title=f.SCENARIO_LABELS[bundle["report"]["scenario"]] + " | Full pipeline")
            f.save(fig, output / "full_pipeline" / ("signal_region_" + metric))
        else:
            f.plt.close(fig)
    bundle["metrics"]["oracle_benchmarks"] = audit


def render_injection_scan(groups, output, args):
    rows, cohorts = [], {}
    for (count, replica), group in sorted(groups.items()):
        reference = None
        for method, (root, report) in group.items():
            inputs = report["contract"]["inputs"]
            if inputs.get("synthetic_smoke_fixture") and not args.allow_smoke:
                raise ValueError("Synthetic scan requires --allow-smoke for QA")
            expected_protocol = 2 if method == "riddle" else "pinned_upstream"
            if report["contract"].get("scientific_version") != expected_protocol:
                raise ValueError(f"Injection scans require protocol {expected_protocol!r} for {method}")
            point = inputs["injection_scan"]
            if report["seed"] != point["training_seed"]:
                raise ValueError("Scan result has an incompatible training seed")
            if reference is not None and inputs != reference:
                raise ValueError("Scan methods must share identical prepared data at each point")
            reference = inputs
            data = load_scores(root, report, "signal_region")
            values = oracle_metrics(data["labels"], data["scores"], data["mask"],
                                    min_background=args.min_background)
            initial = inputs["uncut_signal_region"]
            b0, s0 = initial["background"], initial["signal"]
            if b0 <= 0 or s0 <= 0:
                raise ValueError("Positive realized SR background and signal counts required for a scan")
            maximum = values["full_pipeline_max_sic"]
            row = dict(method=method, signal_events=count, replica=replica,
                       training_seed=point["training_seed"], preparation_seed=point["preparation_seed"],
                       sr_background=b0, sr_signal=s0, signal_to_background_percent=100*s0/b0,
                       uncut_nominal_significance=s0/np.sqrt(b0),
                       conditional_mapped_auc=values["conditional_auc"],
                       oracle_conditional_max_sic=values["conditional_max_sic"],
                       oracle_full_pipeline_max_sic=maximum,
                       oracle_max_nominal_significance=maximum*s0/np.sqrt(b0) if maximum is not None else None,
                       background_mapping_acceptance=values["acceptance"]["background"]["acceptance"],
                       signal_mapping_acceptance=values["acceptance"]["signal"]["acceptance"])
            rows.append(row)
            cohorts.setdefault((method, count), []).append(row)
    csv_write(output / "injection_scan.csv", rows)
    audit = dict(
        metric="Oracle maximum on truth-labelled SR test sample; mapping failures never pass",
        nominal_significance="Per replica: maximum SIC times realized uncut SR S/sqrt(B), then aggregate",
        uncertainty="16/50/84 percentiles across independently seeded partitions and training; finite source pool is reused, not independent collision datasets",
        minimum_background_count=args.min_background, minimum_background_efficiency=1e-4,
        significance_caveat="Nominal S/sqrt(B); no systematics, background fit, Poisson calibration or trials correction",
        points=[], plotted_metrics={},
    )
    for field, title, filename in (
        ("oracle_full_pipeline_max_sic", "Oracle maximum significance improvement", "maximum_sic_vs_injection"),
        ("oracle_max_nominal_significance", "Oracle maximum nominal significance", "maximum_nominal_significance_vs_injection"),
    ):
        fig, ax, _ = f.canvas(title, "Injected SR S/B [%]")
        any_drawn = False
        plotted = {}
        for method in KEYS:
            values = []
            for (name, count), cohort in cohorts.items():
                if name != method:
                    continue
                finite = [r for r in cohort if r[field] is not None and np.isfinite(r[field])]
                if not finite:
                    continue
                x = float(np.median([r["signal_to_background_percent"] for r in finite]))
                low, median, high = np.quantile([r[field] for r in finite], [.16, .5, .84])
                values.append((x, low, median, high, len(finite), count))
            if not values:
                continue
            values.sort()
            matrix = np.asarray(values)
            x, low, median, high = matrix[:, :4].T
            name, color, ls = STYLES[KEYS[method]]
            ax.plot(x, median, marker="x", color=color, ls=ls, label=name)
            # A single replica has no uncertainty band, even beside multi-replica points.
            supported = matrix[:, 4] >= 2
            if supported.any():
                ax.fill_between(x, low, high, where=supported, color=color, alpha=.22)
            if not supported.all():
                colored_status(f"{name}: scan band unavailable at single-replica points", kind="WARNING")
            plotted[method] = [dict(x=float(a), low=float(l), median=float(m), high=float(h),
                                   replicas=int(n), signal_events=int(c)) for a,l,m,h,n,c in values]
            any_drawn = True
        if any_drawn:
            if field == "oracle_max_nominal_significance":
                for level in (3, 5):
                    ax.axhline(level, ls=":", color=".5", lw=1)
            ax.set_ylim(bottom=0)
            ax.invert_xaxis()
            backgrounds = np.asarray([r["sr_background"] for r in rows])
            if len(backgrounds) and np.all(backgrounds == backgrounds[0]):
                factor = np.sqrt(backgrounds[0])/100
                top = ax.secondary_xaxis("top", functions=(lambda x: x*factor, lambda x: x/factor))
                top.set_xlabel(r"Uncut $S/\sqrt{B}$")
            f.legend(fig, title="Signal-Injected")
            f.save(fig, output / filename)
        else:
            f.plt.close(fig)
        audit["plotted_metrics"][field] = plotted
    audit["points"] = rows
    write_json(output / "injection_scan.json", json_safe(audit))
    return audit


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
            "conditional_mapped_auc",
            {k: roc_auc_score(labels, s) for k, s in scores.items()},
            ci={k: f.auc_interval(p, args.confidence) for k, p in parts.items()},
        )
        if set(parts) == set(STYLES):
            low, high = f.paired_auc_interval(parts, args.confidence)
            rows[-1].update(RIDDLE_minus_LaCathode_ci_low=low, RIDDLE_minus_LaCathode_ci_high=high)
        add("conditional_mapped_average_precision", {k: average_precision_score(labels, s) for k, s in scores.items()})
        maxima = {}
        for key, score in scores.items():
            b, s, _ = roc_curve(labels, score, drop_intermediate=False)
            good = (b >= 1e-4) & (np.rint(b * (labels == 0).sum()) >= args.min_background)
            maxima[key] = np.max(s[good] / np.sqrt(b[good])) if good.any() else None
        add("oracle_conditional_max_sic_supported", maxima)
    evaluation = {
        key: oracle_metrics(record["labels"], record["scores"], record["mask"], min_background=args.min_background)
        for key, record in bundle["evaluation"]["signal_region"].items()
    }
    add("oracle_full_pipeline_max_sic_supported", {key: r["full_pipeline_max_sic"] for key,r in evaluation.items()})
    for name in ("signal", "background"):
        add(name + "_mapping_acceptance", {key: r["acceptance"][name]["acceptance"] for key,r in evaluation.items()})
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
                (f"{method}_model_train_losses.npy", f"{method}_model_val_losses.npy")
                if stage == "background"
                else ("loss_matris.npy", "val_loss_matris.npy")
            )
            if stage == "background" and not all((directory / name).is_file() for name in names):
                # Read legacy artifacts without renaming or modifying saved results.
                names = ("my_ANODE_model_train_losses.npy", "my_ANODE_model_val_losses.npy")
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


def method_band(ax, x, values, key, scenario, metric, seeds):
    name, color, ls = STYLES[key]
    summary = f.draw_band(ax, x, values, name, color, ls)
    summary.update(uncertainty_source="training_seed_variation_on_fixed_data_partition", seeds=seeds)
    if not summary["band_drawn"]:
        colored_status(
            f"{name} | {f.SCENARIO_LABELS[scenario]} | {metric}: "
            "uncertainty band unavailable with only one independent run; "
            "include additional training seeds (ten for the paper comparison). "
            "Internal fits and checkpoints are not independent full runs.",
            kind="WARNING",
        )
    return summary


def summary_figures(bundles, output, args):
    audit = {}
    for scenario in f.SCENARIOS:
        cohort = [b for b in bundles if b["report"]["scenario"] == scenario]
        if not cohort:
            continue
        partitions = {
            json.dumps(report.get("contract", {}).get("inputs", {}).get("files"), sort_keys=True)
            for bundle in cohort for _, report in bundle["sources"].values()
        }
        if len(partitions) > 1:
            raise ValueError("Ordinary seed bands require a fixed prepared partition; use the injection-scan workflow for varied partitions")
        fig, ax, _ = f.canvas("Significance improvement (mapped)", "Background efficiency (mapped)")
        any_curve = False
        for key in STYLES:
            values = []
            seeds = []
            for bundle in cohort:
                labels, scores = bundle["curves"]["signal_region"]
                if key not in scores or len(np.unique(labels)) != 2:
                    continue
                b, s, _ = roc_curve(labels, scores[key])
                supported = (b >= 1e-4) & (np.rint(b * (labels == 0).sum()) >= args.min_background)
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
                seeds.append(bundle["report"]["seed"])
            if values:
                name = STYLES[key][0]
                audit[scenario + "/" + name + "/sic"] = method_band(
                    ax, f.GRID, values, key, scenario, "SIC", seeds
                )
                any_curve = True
        if any_curve:
            ax.plot(f.GRID, np.sqrt(f.GRID), color=".5", ls=":", label="Random")
            f.summary_axes(ax, "sic")
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
            seeds = []
            for bundle in cohort:
                sample = bundle["samples"]["test"]
                if key + "_scores" not in sample:
                    continue
                pop = sample["labels"] == 0
                mass, score = sample["mass"][pop], sample[key + "_scores"][pop]
                mask = sample["mask"][pop]
                region = f.sample_sr(sample, key)[pop] & mask
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
                seeds.append(bundle["report"]["seed"])
            if curves:
                name = STYLES[key][0]
                audit[scenario + "/" + name + "/mass_flatness"] = method_band(
                    ax, f.EFFICIENCIES, curves, key, scenario, "mass flatness", seeds
                )
                found = True
        if found:
            sample = cohort[0]["samples"]["test"]
            mass = sample["mass"][sample["labels"] == 0]
            random_summary = f.draw_band(
                ax, f.EFFICIENCIES, random_reference(mass, mode=args.random_reference),
                "Random bootstrap" if args.random_reference == "bootstrap" else "Random subset", ".5", ":"
            )
            random_summary["uncertainty_source"] = "random_" + args.random_reference + "_trials"
            random_summary["sampling_with_replacement"] = args.random_reference == "bootstrap"
            audit[scenario + "/random/mass_flatness"] = random_summary
            f.summary_axes(ax, "mass_flatness")
            f.legend(fig, title="BG-Only")
            f.save(fig, output / scenario / "mass_flatness_vs_selection")
        else:
            f.plt.close(fig)
    return audit


def random_reference(mass, trials=100, *, mode="bootstrap"):
    if mode not in ("bootstrap", "subset"):
        raise ValueError("Random reference must be bootstrap or subset")
    edges = f.equal_occupancy(mass)
    full = np.histogram(mass, edges)[0]
    bin_id = np.minimum(np.searchsorted(edges, mass, side="right") - 1, len(full) - 1)
    result = []
    for efficiency in f.EFFICIENCIES:
        rng = np.random.RandomState(42)
        values = []
        for _ in range(trials):
            indices = rng.choice(len(mass), size=int(efficiency * len(mass)), replace=mode == "bootstrap")
            values.append(f.shape_chi2(full, np.bincount(bin_id[indices], minlength=len(full)), efficiency))
        result.append(values)
    return np.asarray(result).T


def main(argv=None):
    parser = argparse.ArgumentParser(description="Figures from frozen LaCathode/RIDDLE results; no fitting")
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, default=Path("plots"))
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Regenerate matching plots and tables in an existing output directory; preserve other files",
    )
    parser.add_argument("--methods", nargs="+", choices=tuple(KEYS), default=list(KEYS))
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--min-background", type=int, default=10)
    parser.add_argument("--random-reference", choices=("bootstrap", "subset"), default="bootstrap")
    parser.add_argument("--allow-smoke", action="store_true", help="Permit synthetic QA fixtures")
    parser.add_argument("--verbose", type=int, choices=(0, 1, 2), default=1)
    args = parser.parse_args(argv)
    args.population_summary = True
    set_verbosity(args.verbose)
    try:
        if not 0 < args.confidence < 1 or args.min_background < 1:
            raise ValueError("Invalid confidence level or background support")
        groups = discover(args.results, args.methods)
        scan_groups = discover(args.results, args.methods, scan=True)
        if not groups and not scan_groups:
            colored_status("No completed results for the requested methods", kind="WARNING")
            return 0
        args.results, args.output = args.results.resolve(), args.output.resolve()
        if args.output.is_relative_to(args.results) or args.results.is_relative_to(args.output):
            raise ValueError("Keep plots outside the result input tree")
        if args.output.exists():
            if not args.output.is_dir():
                raise NotADirectoryError("Plot output must be a directory")
            if not args.overwrite:
                raise FileExistsError("Plot output exists; use --overwrite or choose a new output directory")
            if any(path.is_symlink() for path in args.output.rglob("*")):
                raise ValueError("Cannot overwrite a plot directory containing symbolic links")
        args.output.mkdir(parents=True, exist_ok=args.overwrite)
        rows, settings, bundles = [], [], []
        with local_progress("Plots"), f.plt.style.context(f.STYLE), locked(args.output / ".plot.lock"):
            scan_audit = render_injection_scan(scan_groups, args.output / "injection_scan", args) if scan_groups else None
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
                        "cuts": "Truth-assisted MC benchmark: validation-SR background thresholds; frozen cuts evaluated on independent physical test rows",
                        "injection_scan": scan_audit,
                        "mass_cut_scan": {
                            "thresholds": list(f.SCORE_CUTS),
                            "comparison": "strict score > threshold",
                            "score_coordinate": "LaCathode classifier score; sigmoid of RIDDLE log density ratio (not a signal probability)",
                            "scope": "full physical test mass range",
                            "retention_denominator": "all physical test events of the corresponding class, before cuts and mapping rejection",
                            "panels_per_page": 6,
                            "individual_directory": "04_mass_cuts/individual_cuts",
                            "histograms": "unweighted event counts, identical mass bins, no smoothing; B + S is the sum of background and signal counts",
                        },
                        "mass_summary": "BG-Only: test-SR quantiles, 300 equal-occupancy full-range bins, normalized-shape Poisson chi2",
                        "uncertainty_scope": "SIC/mass bands: 16/50/84 percentiles of training-seed variation on a fixed data partition; not internal fits or statistical pseudoexperiments",
                        "summary_axes": f.SUMMARY_AXES,
                        "warnings": [w for b in bundles for w in b["warnings"]],
                    }
                ),
            )
        colored_status("Figures and CSV tables completed", kind="PASS")
        return 0
    except (ValueError, OSError, KeyError, RuntimeError) as error:
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
