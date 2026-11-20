from __future__ import annotations
import ast
import copy
import functools
import inspect
import weakref
import numpy as np
import torch


def _valid_moments(moments):
    means, variances = (torch.cat(values) for values in zip(*moments))
    return bool((torch.isfinite(means) & torch.isfinite(variances) & (variances > 0)).all())


def _install_wide_normalization(norm_type):
    original_forward = norm_type.forward
    if getattr(original_forward, "_runtime_wide_moments", False):
        return
    original_load = norm_type._load_from_state_dict

    @functools.wraps(original_forward)
    def forward(layer, inputs, cond_inputs=None, mode="direct"):
        if layer.training or layer.running_var.dtype == inputs.dtype:
            return original_forward(layer, inputs, cond_inputs, mode)
        output, logdet = original_forward(layer, inputs.to(layer.running_var.dtype), cond_inputs, mode)
        return output.to(inputs.dtype), logdet.to(inputs.dtype)

    @functools.wraps(original_load)
    def load(layer, state_dict, prefix, *args, **kwargs):
        saved = state_dict.get(prefix + "running_var")
        if isinstance(saved, torch.Tensor):
            dtype = torch.promote_types(layer.log_gamma.dtype, saved.dtype)
            if layer.running_var.dtype != dtype:
                layer.running_mean = torch.empty_like(layer.running_mean, dtype=dtype)
                layer.running_var = torch.empty_like(layer.running_var, dtype=dtype)
        return original_load(layer, state_dict, prefix, *args, **kwargs)

    forward._runtime_wide_moments = True
    norm_type.forward = forward
    norm_type._load_from_state_dict = load


@torch.no_grad()
def _refresh_flow_statistics(model, norm_type, inputs, conditions):
    layers = [layer for layer in model.modules() if isinstance(layer, norm_type)]
    for layer in layers:
        if layer.running_var.dtype != inputs.dtype:
            layer.running_mean = torch.zeros_like(layer.running_mean, dtype=inputs.dtype)
            layer.running_var = torch.zeros_like(layer.running_var, dtype=inputs.dtype)
    model(inputs, conditions)
    if _valid_moments([(layer.running_mean, layer.running_var) for layer in layers]):
        return
    from riddle.production import NumericalFitError
    from .worker_progress import emit_message

    reference = copy.deepcopy(model).double()
    wide_layers = [layer for layer in reference.modules() if isinstance(layer, norm_type)]
    for layer in wide_layers:
        layer.running_mean.zero_()
        layer.running_var.zero_()
    reference(inputs.double(), conditions.double())
    moments = [
        (wide.running_mean.to(layer.running_mean), wide.running_var.to(layer.running_var))
        for layer, wide in zip(layers, wide_layers)
    ]
    needs_wide = not _valid_moments(moments)
    if needs_wide and not _valid_moments([(wide.running_mean, wide.running_var) for wide in wide_layers]):
        raise NumericalFitError("Flow population moments are nonfinite or nonpositive in float64 on the training sample")
    for layer, wide, (mean, variance) in zip(layers, wide_layers, moments):
        if needs_wide and not _valid_moments([(mean, variance)]):
            mean, variance = wide.running_mean, wide.running_var
            layer.running_mean, layer.running_var = mean, variance
        else:
            layer.running_mean.copy_(mean)
            layer.running_var.copy_(variance)
        layer.batch_mean = mean
        layer.batch_var = variance
    detail = " and retained out-of-range moments in float64" if needs_wide else ""
    emit_message(f"Recomputed flow normalization on the full training sample in float64{detail}", level=0)


def _epoch_loss_mean(chunks):
    if not chunks:
        return np.empty(0, dtype=np.float64).mean()
    return torch.cat(chunks).cpu().numpy().astype(np.float64, copy=False).mean()


def install_flow_loss_reporting(module):
    original = module.train_epoch
    if getattr(original, "_runtime_epoch_losses", False):
        return
    _install_wide_normalization(original.__globals__["fnn"].BatchNormFlow)
    source = inspect.getsource(original)
    replacements = [
        ("train_loss += loss.mean().item()", "if verbose: train_loss += loss.mean().item()", 1),
        ("model(data_loader.dataset.tensors[0].to(data.device),",
         "_refresh_flow_statistics(model, fnn.BatchNormFlow, data_loader.dataset.tensors[0].to(data.device),", 1),
    ]
    for name, loss, count in (("train_loss_avg", "loss", 2), ("corrected_train_loss_avg", "corrected_loss", 1)):
        replacements.extend([
            (f"{name}.extend({loss}.tolist())", f"{name}.append({loss}.detach().reshape(-1))", 1),
            (f"np.array({name}).flatten().mean()", f"_epoch_loss_mean({name})", count),
        ])
    for expected, replacement, count in replacements:
        if source.count(expected) != count:
            raise RuntimeError("Unrecognized LaCATHODE flow loss reporting")
        source = source.replace(expected, replacement)
    namespace = dict(original.__globals__, _epoch_loss_mean=_epoch_loss_mean,
                     _refresh_flow_statistics=_refresh_flow_statistics)
    exec(compile(source, f"<runtime flow loss reporting: {original.__code__.co_filename}>", "exec"), namespace)
    patched = functools.update_wrapper(namespace[original.__name__], original)
    patched._runtime_epoch_losses = True
    module.train_epoch = patched


def install_validation_counts(module):
    original = module.compute_loss_over_batches
    if getattr(original, "_runtime_vectorized_counts", False):
        return
    tree = ast.parse(inspect.getsource(original))
    for expression in ("torch.isnan(loss_vals)", "torch.abs(loss_vals) >= 1000"):
        expected = ast.parse(f"sum({expression})", mode="eval").body
        matches = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and ast.dump(node) == ast.dump(expected)
        ]
        if len(matches) != 1:
            raise RuntimeError("Unexpected runtime validation counters; refusing acceleration patch")
        matches[0].func = ast.Attribute(
            value=ast.Name(id="torch", ctx=ast.Load()), attr="count_nonzero", ctx=ast.Load()
        )
    ast.fix_missing_locations(tree)
    ast.increment_lineno(tree, original.__code__.co_firstlineno - 1)
    namespace = {}
    exec(
        compile(tree, f"<runtime validation counters: {original.__code__.co_filename}>", "exec"),
        original.__globals__,
        namespace,
    )
    patched = functools.update_wrapper(namespace[original.__name__], original)
    patched._runtime_vectorized_counts = True
    module.compute_loss_over_batches = patched


class TensorBatchAcceleration:
    """Gather TensorDataset batches from optional CUDA mirrors without changing
    datasets, sampling order, or training loops."""

    def __init__(self, device=None):
        self.device = None
        self._resident = weakref.WeakKeyDictionary()
        self.report = {
            "verified_loaders": 0,
            "gather_batches": 0,
            "fallback_batches": 0,
            "unsupported_iterators": 0,
            "resident_loaders": 0,
            "already_resident_loaders": 0,
            "resident_cache_hits": 0,
            "resident_bytes": 0,
            "resident_fallbacks": 0,
        }
        self.set_device(device)

    def set_device(self, device):
        if device is None or str(device).lower() == "cpu":
            self.device = None
            return
        resolved = torch.device(device)
        if resolved.type != "cuda":
            self.device = None
            return
        self.device = resolved

    @staticmethod
    def _signature(tensors):
        return tuple(
            (
                id(t),
                int(getattr(t, "_version", 0)),
                tuple(t.shape),
                tuple(t.stride()),
                t.dtype,
                t.device,
            )
            for t in tensors
        )

    def _resident_tensors(self, loader):
        tensors = loader.dataset.tensors
        if self.device is None:
            return tensors, False
        if any(t.device.type not in ("cpu", "cuda") for t in tensors):
            return tensors, False
        if any(t.device.type == "cuda" and t.device != self.device for t in tensors):
            return tensors, False
        if all(t.device == self.device for t in tensors):
            self.report["already_resident_loaders"] += 1
            return tensors, True

        signature = self._signature(tensors)
        cached = self._resident.get(loader.dataset)
        if cached is not None and cached[0] == signature:
            self.report["resident_cache_hits"] += 1
            return cached[1], True

        resident = None
        try:
            with torch.no_grad():
                resident = tuple(
                    t if t.device == self.device else t.to(self.device, non_blocking=False)
                    for t in tensors
                )
        except torch.cuda.OutOfMemoryError:
            resident = None
            self.report["resident_fallbacks"] += 1
            torch.cuda.empty_cache()
            return tensors, False

        self._resident[loader.dataset] = (signature, resident)
        self.report["resident_loaders"] += 1
        self.report["resident_bytes"] += sum(
            t.numel() * t.element_size() for source, t in zip(tensors, resident)
            if source.device != self.device
        )
        return resident, True

    @staticmethod
    def _matched(expected, batch):
        if len(expected) != len(batch):
            return False
        for a, b in zip(expected, batch):
            if a.shape != b.shape or a.dtype != b.dtype or a.stride() != b.stride():
                return False
            if a.device == b.device:
                equal = torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))
            else:
                equal = torch.equal(
                    a.detach().cpu().contiguous().view(torch.uint8),
                    b.detach().cpu().contiguous().view(torch.uint8),
                )
            if not equal:
                return False
        return True

    def install(self, iterator, loader):
        fetcher = getattr(iterator, "_dataset_fetcher", None)
        if (
            type(loader.dataset) is not torch.utils.data.TensorDataset
            or loader.num_workers != 0
            or loader.pin_memory
            or (loader.collate_fn is not torch.utils.data.default_collate)
            or (
                type(loader.sampler)
                not in (torch.utils.data.RandomSampler, torch.utils.data.SequentialSampler)
            )
            or (fetcher is None)
            or (not getattr(fetcher, "auto_collation", False))
            or any(
                t.layout != torch.strided or t.requires_grad or t.device.type not in ("cpu", "cuda")
                for t in loader.dataset.tensors
            )
        ):
            self.report["unsupported_iterators"] += 1
            return
        original = fetcher.fetch
        tensors, resident = self._resident_tensors(loader)
        devices = tuple(dict.fromkeys(tensor.device for tensor in tensors))

        def fetch(indices):
            if (
                getattr(loader, "_runtime_gather_verified", None) is False
                or not isinstance(indices, (list, tuple))
                or (not indices)
                or any(type(index) is not int or index < 0 for index in indices)
            ):
                self.report["fallback_batches"] += 1
                return original(indices)
            try:
                cpu_indices = torch.tensor(indices, dtype=torch.long, device="cpu")
                by_device = {device: cpu_indices.to(device, non_blocking=True) for device in devices}
                batch = [tensor.index_select(0, by_device[tensor.device]) for tensor in tensors]
            except torch.cuda.OutOfMemoryError:
                loader._runtime_gather_verified = False
                self.report["fallback_batches"] += 1
                return original(indices)
            if getattr(loader, "_runtime_gather_verified", None) is None:
                expected = original(indices)
                matched = self._matched(expected, batch)
                loader._runtime_gather_verified = matched
                if not matched:
                    self.report["fallback_batches"] += 1
                    return expected
                self.report["verified_loaders"] += 1
            if resident:
                loader._runtime_gpu_resident = True
            self.report["gather_batches"] += 1
            return batch

        fetcher.fetch = fetch


def install_tensor_batches(device=None):
    original = torch.utils.data.DataLoader.__iter__
    if hasattr(original, "_tensor_acceleration"):
        acceleration = original._tensor_acceleration
        acceleration.set_device(device)
        return acceleration
    acceleration = TensorBatchAcceleration(device)

    def iterator(loader):
        result = original(loader)
        acceleration.install(result, loader)
        return result

    iterator._tensor_acceleration = acceleration
    torch.utils.data.DataLoader.__iter__ = iterator
    return acceleration


def execution_report(batches=None):
    return {
        "tensor_minibatches": batches.report if batches is not None else "not used by full-array evaluation",
        "tensor_residency": (
            "compatible TensorDataset tensors mirrored once to CUDA; Dataset/sampler/RNG unchanged; exact first-batch verification"
            if batches is not None and batches.device is not None
            else "CPU gather acceleration only"
        ) if batches is not None else "not used by full-array evaluation",
        "validation_counters": "exact integer count_nonzero; floating loss reductions unchanged"
        if batches is not None
        else "not used by full-array evaluation",
        "array_hashing": "zero-copy contiguous buffer hashing; identical SHA256 contract",
        "precision": "upstream training dtype; invalid full-sample moments recomputed in float64 and stored at sufficient precision; no AMP or TF32",
        "unchanged": [
            "optimizer",
            "batch sizes",
            "sampler and RNG",
            "loss reductions",
            "checkpoint retention and selection",
            "full-array CPU SR evaluation",
        ],
    }
