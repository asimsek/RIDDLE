"""Production mapping and disjoint fitting/calibration/evidence populations."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .enhancements import Rosenblatt, chunks, save_torch, load_torch, clip_gradients
from .integrity import require_finite, train_flow_epoch
from .preprocessing import load_dataset
from .storage import digest, write_json, save_array, seed_start, rng_state, restore_rng, atomic_write, save_npz, persist_boundary
from .worker_progress import emit_message

PREPROCESS_KEYS = ("min", "max", "mean2", "std2", "std2_logit_fix")


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
        self.reference = load_torch(self.output / "preprocessing.pt")
        self.mass_mean, self.mass_std = metadata["mass_parameters"]
        self.model = build_mapping(metadata["configuration"], self.options, metadata["features"], metadata["seed"], device)
        self.model.load_state_dict(torch.load(self.output / "model.pt", map_location=device, weights_only=True))
        self.model.eval().requires_grad_(False)

    def map(self, rows):
        # Remove truth labels before preprocessing.
        clean = np.asarray(rows, dtype=np.float32).copy(); clean[:, -1] = 0
        prepared = load_dataset(clean, external_datadict=self.reference)
        x, m = prepared["tensor2"], prepared["labels"]
        if self.options["rosenblatt"]:
            z = chunks(self.model, x, (m-self.mass_mean)/self.mass_std, device=self.device)
        else:
            z = chunks(lambda x, m: self.model(x, m)[0], x, m, device=self.device)
        return z.numpy().astype(np.float32), prepared["mask"].numpy()


def prepare(data, output, seed, device, *, background, options, data_policy=None, residual_batch_size=256,
            experiment=None):
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
    # Only mass and observables enter fitting and reproducibility checks.
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
    contract = dict(schema=1, implementation="production_conditional_mapping_v2_tail_safe_preprocessing",
                    preprocessing="nonrejecting_minmax_logit_clip_eps_1e-6_v1", configuration=config,
                    options=options, background=background, seed=seed, features=features, device=str(device),
                    hashes={k: digest(a[:, :-1]) for k, a in clean.items()}, mass_parameters=mass_parameters)
    if data_policy != "production_v1":
        contract.update(data_policy=data_policy, residual_batch_size=residual_batch_size)
    if experiment is not None:
        contract["mapping_experiment"] = experiment
    settings_path = output / "mapping_settings.json"
    if settings_path.exists() and json.loads(settings_path.read_text()) != contract:
        raise ValueError("Mapping data/settings changed; use a new output")
    if not settings_path.exists() and (output / ".resume/latest.pt").exists():
        raise ValueError("Mapping checkpoint has no provenance contract")
    write_json(settings_path, contract)
    write_json(output / "background_settings.json", background)
    atomic_write(output / "source_partitions.npz", lambda p: save_npz(p, **{k: v["indices"] for k, v in roles.items()}))
    write_json(output / "data_roles.json", {k: dict(source=v["source"]+".npy", events=len(v["indices"]),
               indices_sha256=digest(v["indices"]), truth_labels_used=False) for k, v in roles.items()})
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
        loss = -float(chunks(model.log_probs, val["tensor2"], context, device=device).double().mean())
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
    mapper = Mapper(output, device)
    mapped = {}
    for name, rows in arrays.items():
        z, mask = mapper.map(rows)
        mapped[name] = dict(z=z, mass=rows[mask, 0], mask=mask, rows=rows)
    attach_source_ids(data, roles, mapped)
    mapped = finalize_mapped(mapped, seed, data_policy, residual_batch_size)
    identity_arrays = {}
    for role, item in mapped.items():
        if role == "member_splits": continue
        for key in ("ids", "source_ids", "source_indices", "mask"):
            if key in item: identity_arrays[role+"__"+key] = item[key]
    if identity_arrays:
        atomic_write(output / "event_roles.npz", lambda p: save_npz(p, **identity_arrays))
    write_json(output / "role_policy.json", dict(policy=data_policy, truth_labels_used=False))
    def latent_rows(item):
        return np.column_stack((item["mass"], item["z"], np.ones(len(item["z"])), np.zeros(len(item["z"])))).astype(np.float32)
    for name, role in (("training_latents.npy", "residual_train"), ("validation_latents.npy", "evidence"),
                       ("mixture_validation_latents.npy", "mixture_validation")):
        atomic_write(output / name, lambda p, value=latent_rows(mapped[role]): save_array(p, value))
    return selection, mapper, mapped
