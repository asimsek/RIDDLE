"""Production mapping and disjoint fitting/calibration/evidence populations."""
from copy import deepcopy
import json
from pathlib import Path
import shutil

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .enhancements import Rosenblatt, save_torch, load_torch, clip_gradients
from .integrity import SCIENTIFIC_VERSION, require_finite, train_flow_epoch
from .preprocessing import load_dataset
from .storage import digest, file_digest, write_json, save_array, seed_start, rng_state, restore_rng, atomic_write, save_npz, persist_boundary
from .worker_progress import emit_message

PREPROCESS_KEYS = ("min", "max", "mean2", "std2", "std2_logit_fix")


def _inference_chunks(function, *arrays, device, size):
    if not arrays or not len(arrays[0]) or any(len(a) != len(arrays[0]) for a in arrays):
        raise ValueError("Inference requires aligned, nonempty arrays")
    target = torch.device(device)
    requested = int(size)
    if requested <= 0:
        raise ValueError("Mapping inference batch size must be positive")
    current = min(requested, 8192) if target.type == "cpu" else requested
    pieces = []
    offset = 0
    while offset < len(arrays[0]):
        stop = min(offset + current, len(arrays[0]))
        try:
            with torch.inference_mode():
                value = function(*(a[offset:stop].to(target) for a in arrays)).detach().cpu()
            pieces.append(value)
            offset = stop
        except torch.cuda.OutOfMemoryError:
            if target.type != "cuda" or current <= 256:
                raise
            torch.cuda.empty_cache()
            current = max(256, current // 2)
    result = torch.cat(pieces)
    require_finite(result, "RIDDLE mapping inference")
    return result, current


def _map_roles_by_source(mapper, sources, roles):
    mapped = {}
    used = []
    for source in sorted({role["source"] for role in roles.values()}):
        role_names = [name for name, role in roles.items() if role["source"] == source]
        union_indices = np.unique(np.concatenate([np.asarray(roles[name]["indices"], dtype=np.int64)
                                                   for name in role_names]))
        union_rows = sources[source][union_indices]
        union_z, union_mask = mapper.map(union_rows)
        if mapper.last_inference_batch_size is not None:
            used.append(int(mapper.last_inference_batch_size))
        latent_index = np.full(len(union_indices), -1, dtype=np.int64)
        latent_index[union_mask] = np.arange(int(union_mask.sum()), dtype=np.int64)
        for name in role_names:
            indices = np.asarray(roles[name]["indices"], dtype=np.int64)
            positions = np.searchsorted(union_indices, indices)
            if (np.any(positions >= len(union_indices))
                    or not np.array_equal(union_indices[positions], indices)):
                raise ValueError("Role index reconstruction failed")
            mask = union_mask[positions]
            rows = sources[source][indices]
            mapped[name] = dict(z=union_z[latent_index[positions[mask]]], mass=rows[mask, 0],
                                mask=mask, rows=rows)
    return mapped, (min(used) if used else None)


def split_roles(data, seed, *, calibration):
    """Source-row partitions are saved and never depend on MC labels."""
    data = Path(data)
    sources = {name: np.load(data / (name+".npy")).astype(np.float32)
               for name in ("outerdata_train", "outerdata_val", "innerdata_train", "innerdata_val")}
    for name, rows in sources.items():
        require_finite(rows[:, :-1], name)
        in_sr = (rows[:, 0] > 3.3) & (rows[:, 0] < 3.7)
        if (name.startswith("outer") and in_sr.any()) or (name.startswith("inner") and not in_sr.all()):
            raise ValueError("Source rows do not match their declared mass region")
    roles = {}
    def assign(source, names, fractions, tag):
        n = len(sources[source]); order = np.random.default_rng(np.random.SeedSequence([seed, tag])).permutation(n)
        pieces = np.split(order, [int(n*f) for f in np.cumsum(fractions)[:-1]])
        for name, indices in zip(names, pieces):
            if len(indices) < 2:
                raise ValueError(f"Too few events for independent {name}")
            roles[name] = dict(source=source, indices=indices)
    # Keep upstream populations fixed when disabling score-flow ablations.


    assign("outerdata_train", ("map_train", "calibration_train"), (.75, .25), 100)
    assign("outerdata_val", ("map_val", "calibration_val", "closure"), (.5, .25, .25), 101)
    assign("innerdata_train", ("residual_train",), (1.,), 102)
    assign("innerdata_val", ("mixture_validation", "evidence"), (.5, .5), 103)
    return sources, roles


def build_mapping(config, options, features, seed, device):
    if options["rosenblatt"]:
        return Rosenblatt(features, seed, bins=options["rosenblatt_bins"],
                          hidden=options["rosenblatt_hidden"], bound=options["rosenblatt_bound"]).to(device)
    from .density_estimator import DensityEstimator
    return DensityEstimator({**config, "num_inputs": features}, device=device, verbose=False, bound=False).model


class Mapper:
    """Reloadable frozen map, with preprocessing fitted on map-training rows only."""
    def __init__(self, output, device="cpu"):
        self.output, self.device = Path(output), device
        self.selection = json.loads((self.output / "flow_selection.json").read_text())
        metadata = json.loads((self.output / "mapping_settings.json").read_text())
        self.options = metadata["options"]
        runtime_background = (json.loads((self.output / "background_settings.json").read_text())
                              if (self.output / "background_settings.json").exists() else metadata["background"])
        self.inference_batch_size = int(runtime_background.get("mapping_inference_batch_size", 65536))
        self.reference = load_torch(self.output / "preprocessing.pt")
        self.mass_mean, self.mass_std = metadata["mass_parameters"]
        self.model = build_mapping(metadata["configuration"], self.options, metadata["features"], metadata["seed"], device)
        self.model.load_state_dict(torch.load(self.output / "model.pt", map_location=device, weights_only=True))
        self.model.eval().requires_grad_(False)
        self.last_inference_batch_size = None

    def map(self, rows):
        # Remove truth labels before preprocessing.
        clean = np.asarray(rows, dtype=np.float32).copy(); clean[:, -1] = 0
        prepared = load_dataset(clean, external_datadict=self.reference)
        x, m = prepared["tensor2"], prepared["labels"]
        if self.options["rosenblatt"]:
            z, used = _inference_chunks(
                self.model, x, (m-self.mass_mean)/self.mass_std,
                device=self.device, size=self.inference_batch_size,
            )
        else:
            z, used = _inference_chunks(
                lambda x, m: self.model(x, m)[0], x, m,
                device=self.device, size=self.inference_batch_size,
            )
        self.last_inference_batch_size = int(used)
        return z.numpy().astype(np.float32), prepared["mask"].numpy()


def _reuse_signature(contract):
    keys = ("schema", "implementation", "preprocessing", "configuration", "options", "background",
            "features", "data_policy", "residual_batch_size")
    return {key: contract.get(key) for key in keys if key in contract}


def _verified_source_artifact(source, report, relative):
    path = source / relative
    expected = report.get("artifacts_sha256", {}).get(str(relative))
    if expected is None or not path.is_file() or file_digest(path) != expected:
        return None
    return path


def _mapping_reuse_candidate(candidate, current_contract, seed):
    source = Path(candidate).resolve()
    report_path = source / "result.json"
    if not report_path.is_file():
        return None
    try:
        report = json.loads(report_path.read_text())
    except json.JSONDecodeError:
        return None
    if report.get("completed") is not True or not str(report.get("method", "")).startswith("riddle"):
        return None
    if report.get("contract", {}).get("scientific_version") != SCIENTIFIC_VERSION:
        return None
    if report.get("scenario") != "signal_injection":
        return None
    relative_files = (
        Path("background/model.pt"),
        Path("background/preprocessing.pt"),
        Path("background/flow_selection.json"),
        Path("background/mapping_settings.json"),
    )
    verified = {}
    for relative in relative_files:
        path = _verified_source_artifact(source, report, relative)
        if path is None:
            return None
        verified[str(relative)] = path
    try:
        metadata = json.loads(verified["background/mapping_settings.json"].read_text())
        selection = json.loads(verified["background/flow_selection.json"].read_text())
    except json.JSONDecodeError:
        return None
    if _reuse_signature(metadata) != _reuse_signature(current_contract):
        return None
    if selection.get("implementation") != current_contract["implementation"]:
        return None
    optional = {}
    for relative in (Path("background/history.json"),):
        path = _verified_source_artifact(source, report, relative)
        if path is not None:
            optional[str(relative)] = path
    return source, report, metadata, selection, verified, optional


def _activate_mapping_reuse(output, candidates, current_contract, seed):
    manifest_path = output / "mapping_reuse.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("policy") not in ("same_replica_fixed_background_v1", "shared_fixed_background_v1"):
            raise ValueError("Unrecognized mapping reuse policy")
        if manifest.get("target_seed") != seed or manifest.get("target_signature") != _reuse_signature(current_contract):
            raise ValueError("Mapping reuse target settings changed; use a new output")
        for name, expected in manifest.get("local_artifacts_sha256", {}).items():
            path = output / name
            if not path.is_file() or file_digest(path) != expected:
                raise ValueError("Reused mapping artifact changed or is missing")
        return manifest
    found = None
    for candidate in candidates or ():
        candidate_result = _mapping_reuse_candidate(candidate, current_contract, seed)
        if candidate_result is not None:
            found = candidate_result
            break
    if found is None:
        return None
    downstream = output.parent / "density"
    local_training = (output / "mapping_settings.json").exists() or (output / ".resume/latest.pt").exists()
    if local_training and downstream.exists() and any(path.is_file() for path in downstream.rglob("*")):
        return None
    if local_training:
        for name in ("model.pt", "preprocessing.pt", "flow_selection.json", "mapping_settings.json", "history.json", "mapping_runtime.json"):
            path = output / name
            if path.exists():
                path.unlink()
        shutil.rmtree(output / ".resume", ignore_errors=True)
    source, report, metadata, selection, verified, optional = found
    copied = {}
    for relative, source_path in {**verified, **optional}.items():
        target_name = Path(relative).name
        target = output / target_name
        shutil.copy2(source_path, target)
        copied[target_name] = file_digest(target)
    manifest = {
        "schema": 1,
        "policy": "shared_fixed_background_v1",
        "source_result": str(source),
        "source_result_sha256": file_digest(source / "result.json"),
        "source_seed": report.get("seed"),
        "source_scenario": report.get("scenario"),
        "source_variant": report.get("variant", report.get("contract", {}).get("inputs", {}).get("variant", "default")),
        "target_seed": seed,
        "target_signature": _reuse_signature(current_contract),
        "source_mapping_settings_sha256": file_digest(verified["background/mapping_settings.json"]),
        "source_model_sha256": file_digest(verified["background/model.pt"]),
        "source_preprocessing_sha256": file_digest(verified["background/preprocessing.pt"]),
        "local_artifacts_sha256": copied,
        "training_reused": True,
        "truth_labels_used": False,
    }
    write_json(manifest_path, manifest)
    emit_message(f"Reuse frozen RIDDLE background map from {source}", kind="PASS", level=0)
    return manifest

def prepare(data, output, seed, device, *, background, options, data_policy=None, residual_batch_size=256,
            experiment=None, reuse_candidates=None):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    sources, roles = split_roles(data, seed, calibration=options["score_flow"])
    from .roles import DEFAULT_POLICY, override_roles, attach_source_ids, finalize_mapped
    data_policy = DEFAULT_POLICY if data_policy is None else data_policy
    roles = override_roles(sources, roles, seed, data_policy, residual_batch_size)
    experiment_reference = None
    if experiment is not None:
        if not options["rosenblatt"]:
            raise ValueError("Mapping-component experiments require Rosenblatt mapping")
        from .mapping_experiment import mapping_inputs
        roles, experiment_reference = mapping_inputs(experiment, roles, sources, device)
    arrays = {k: sources[v["source"]][v["indices"]] for k, v in roles.items()}
    clean = {}
    for name, rows in arrays.items():
        clean[name] = rows.copy(); clean[name][:, -1] = 0
    config = deepcopy(background["configuration"])
    features = clean["map_train"].shape[1]-2
    config["num_inputs"] = features
    mass_parameters = [float(clean["map_train"][:, 0].mean()), float(clean["map_train"][:, 0].std())]
    if experiment is not None:
        mass_parameters = experiment["mass_parameters"]
    if mass_parameters[1] <= 0:
        raise ValueError("Mapping requires a nonzero sideband mass span")
    background_contract = {k: v for k, v in background.items()
                           if k not in ("mapping_validation_batch_size", "mapping_inference_batch_size")}
    contract = dict(schema=1, implementation="production_conditional_mapping_v2_tail_safe_preprocessing",
                    preprocessing="nonrejecting_minmax_logit_clip_eps_1e-6_v1", configuration=config,
                    options=options, background=background_contract, seed=seed, features=features, device=str(device),
                    hashes={k: digest(a[:, :-1]) for k, a in clean.items()}, mass_parameters=mass_parameters)
    if data_policy != "production_v1":
        contract.update(data_policy=data_policy, residual_batch_size=residual_batch_size)
    if experiment is not None:
        contract["mapping_experiment"] = experiment
    mapping_reuse = None if experiment is not None else _activate_mapping_reuse(
        output, reuse_candidates, contract, seed
    )
    settings_path = output / "mapping_settings.json"
    write_json(output / "background_settings.json", background)
    atomic_write(output / "source_partitions.npz", lambda p: save_npz(p, **{k: v["indices"] for k, v in roles.items()}))
    write_json(output / "data_roles.json", {k: dict(source=v["source"]+".npy", events=len(v["indices"]),
               indices_sha256=digest(v["indices"]), truth_labels_used=False) for k, v in roles.items()})
    validation_batch_used = None
    if mapping_reuse is None:
        if settings_path.exists() and json.loads(settings_path.read_text()) != contract:
            raise ValueError("Mapping data/settings changed; use a new output")
        if not settings_path.exists() and (output / ".resume/latest.pt").exists():
            raise ValueError("Mapping checkpoint has no provenance contract")
        write_json(settings_path, contract)
        fit = (load_dataset(clean["map_train"]) if experiment is None else
               load_dataset(clean["map_train"], external_datadict=experiment_reference))
        val = load_dataset(clean["map_val"], external_datadict=fit)
        seed_start(seed)
        model = build_mapping(config, options, features, seed, device)
        if options["rosenblatt"]:
            optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-6)
            seed_start((seed+1) % 2**32)
        else:
            cfg = config["optimizer"]
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        loader = DataLoader(TensorDataset(fit["tensor2"], fit["labels"]), batch_size=background["batch_size"], shuffle=True)
        latest = output / ".resume/latest.pt"
        history, start, best, best_model, best_epoch = [], 0, float("inf"), None, None
        if latest.exists():
            state = load_torch(latest, device)
            if state["contract"] != contract:
                raise ValueError("Mapping recovery contract changed")
            model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
            history, start, best, best_model, best_epoch = (state[k] for k in ("history", "next_epoch", "best", "best_model", "best_epoch"))
            restore_rng(state["rng"])
        mm, ms = mass_parameters
        validation_batch_used = (min(int(background["mapping_validation_batch_size"]), 8192)
                                 if torch.device(device).type == "cpu"
                                 else int(background["mapping_validation_batch_size"]))
        for epoch in range(start, background["epochs"]):
            model.train()
            if options["rosenblatt"]:
                total = 0.
                for x, m in loader:
                    optimizer.zero_grad(); loss = -model.log_probs(x.to(device), ((m-mm)/ms).to(device)).mean()
                    require_finite(loss, "Rosenblatt NLL"); loss.backward()
                    for head in model.heads:
                        clip_gradients(head.parameters(), 1.)
                    optimizer.step(); total += float(loss.detach())*len(x)
                train_nll = total/len(fit["tensor2"])
            else:
                from .flows import BatchNormFlow
                train_nll = train_flow_epoch(model, optimizer, loader, device, batch_norm_class=BatchNormFlow, verbose=False)[0]
            model.eval()
            context = (val["labels"]-mm)/ms if options["rosenblatt"] else val["labels"]
            log_prob, used = _inference_chunks(
                model.log_probs, val["tensor2"], context, device=device,
                size=background["mapping_validation_batch_size"],
            )
            validation_batch_used = min(validation_batch_used, int(used))
            loss = -float(log_prob.double().mean())
            history.append(dict(epoch=epoch, train_nll=train_nll, validation_nll=loss))
            if loss < best:
                best, best_model, best_epoch = loss, deepcopy(model.state_dict()), epoch
            if experiment is not None:
                from .mapping_experiment import observe_epoch
                observe_epoch(experiment, output, model, epoch, clean, sources, fit, mass_parameters, device, loss)
            if persist_boundary(epoch, background["epochs"]):
                save_torch(latest, dict(contract=contract, next_epoch=epoch+1, model=model.state_dict(),
                           optimizer=optimizer.state_dict(), rng=rng_state(), history=history,
                           best=best, best_model=best_model, best_epoch=best_epoch))
                write_json(output / "history.json", history)
            emit_message(f"Background map {epoch+1}/{background['epochs']}: validation NLL={loss:.6g}")
        save_torch(output / "model.pt", best_model)
        save_torch(output / "preprocessing.pt", {k: fit[k].cpu() for k in PREPROCESS_KEYS})
        selection = dict(training_mapping_epoch=best_epoch, inference_mapping_epoch=best_epoch,
                         trained_epochs=background["epochs"], criterion="lowest independent map-validation NLL",
                         implementation="production_conditional_mapping_v2_tail_safe_preprocessing",
                         preprocessing="nonrejecting_minmax_logit_clip_eps_1e-6_v1",
                         architecture="Rosenblatt" if options["rosenblatt"] else "MAF",
                         affine_log_scale_bound=None if options["rosenblatt"] else config.get("affine_log_scale_bound"),
                         truth_labels_used=False)
        write_json(output / "flow_selection.json", selection)
    else:
        selection = json.loads((output / "flow_selection.json").read_text())
    mapper = Mapper(output, device)
    mapped, inference_batch_used = _map_roles_by_source(mapper, sources, roles)
    if inference_batch_used is None:
        raise ValueError("Mapping produced no source-union inference batches")
    write_json(output / "mapping_runtime.json", {
        "device": str(device),
        "mapping_validation_batch_size_requested": int(background["mapping_validation_batch_size"]),
        "mapping_validation_batch_size_used": validation_batch_used,
        "mapping_inference_batch_size_requested": int(background["mapping_inference_batch_size"]),
        "mapping_inference_batch_size_used": int(inference_batch_used),
        "source_union_mapping": True,
        "training_reused": mapping_reuse is not None,
    })
    attach_source_ids(data, roles, mapped)
    mapped = finalize_mapped(mapped, seed, data_policy, residual_batch_size)
    identity_arrays = {}
    for role, item in mapped.items():
        if role == "member_splits":
            continue
        for key in ("ids", "source_ids", "source_indices", "mask"):
            if key in item:
                identity_arrays[role+"__"+key] = item[key]
    if identity_arrays:
        atomic_write(output / "event_roles.npz", lambda p: save_npz(p, **identity_arrays))
    write_json(output / "role_policy.json", dict(policy=data_policy, truth_labels_used=False))
    def latent_rows(item):
        return np.column_stack((item["mass"], item["z"], np.ones(len(item["z"])), np.zeros(len(item["z"])))).astype(np.float32)
    for name, role in (("training_latents.npy", "residual_train"), ("validation_latents.npy", "evidence"),
                       ("mixture_validation_latents.npy", "mixture_validation")):
        atomic_write(output / name, lambda p, value=latent_rows(mapped[role]): save_array(p, value))
    return selection, mapper, mapped, mapping_reuse

