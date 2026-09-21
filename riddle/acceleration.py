from __future__ import annotations
import weakref
import torch


def install_validation_counts(module, diagnostics=None):
    from .integrity import install_flow_validation
    install_flow_validation(module, diagnostics)


class TensorBatchAcceleration:
    """Exact TensorDataset gather acceleration with optional CUDA residency.

    The Dataset object itself is never mutated.  When a CUDA device is supplied,
    compatible CPU TensorDataset tensors are mirrored to that GPU once and the
    DataLoader fetcher gathers batches from the mirror.  Samplers, generators,
    batch membership/order and the upstream training loop remain unchanged.
    """

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

        def fetch(indices):
            if (
                getattr(loader, "_runtime_gather_verified", None) is False
                or not isinstance(indices, (list, tuple))
                or (not indices)
                or any(type(index) is not int or index < 0 for index in indices)
            ):
                self.report["fallback_batches"] += 1
                return original(indices)
            by_device = {tensor.device: None for tensor in tensors}
            try:
                for device in by_device:
                    by_device[device] = torch.tensor(indices, dtype=torch.long, device=device)
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
        "precision": "dtypes; no AMP or TF32",
        "unchanged": [
            "optimizer",
            "batch sizes",
            "sampler and RNG",
            "floating reductions",
            "checkpoint retention and selection",
            "full-array CPU SR evaluation",
        ],
    }
