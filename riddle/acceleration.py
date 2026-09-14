from __future__ import annotations
import torch


def install_validation_counts(module, diagnostics=None):
    from .integrity import install_flow_validation
    install_flow_validation(module, diagnostics)


class TensorBatchAcceleration:
    def __init__(self):
        self.report = {
            "verified_loaders": 0,
            "gather_batches": 0,
            "fallback_batches": 0,
            "unsupported_iterators": 0,
        }

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
                (
                    t.layout != torch.strided or t.requires_grad or t.device.type not in ("cpu", "cuda")
                    for t in loader.dataset.tensors
                )
            )
        ):
            self.report["unsupported_iterators"] += 1
            return
        original = fetcher.fetch

        def fetch(indices):
            if (
                getattr(loader, "_runtime_gather_verified", None) is False
                or not isinstance(indices, (list, tuple))
                or (not indices)
                or any((type(index) is not int or index < 0 for index in indices))
            ):
                self.report["fallback_batches"] += 1
                return original(indices)
            tensors = loader.dataset.tensors
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
                matched = len(expected) == len(batch) and all(
                    (
                        a.shape == b.shape
                        and a.dtype == b.dtype
                        and (a.device == b.device)
                        and (a.stride() == b.stride())
                        and torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))
                        for a, b in zip(expected, batch)
                    )
                )
                loader._runtime_gather_verified = matched
                if not matched:
                    self.report["fallback_batches"] += 1
                    return expected
                self.report["verified_loaders"] += 1
            self.report["gather_batches"] += 1
            return batch

        fetcher.fetch = fetch


def install_tensor_batches():
    original = torch.utils.data.DataLoader.__iter__
    if hasattr(original, "_tensor_acceleration"):
        return original._tensor_acceleration
    acceleration = TensorBatchAcceleration()

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
