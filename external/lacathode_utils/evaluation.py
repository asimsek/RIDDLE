from copy import deepcopy

import numpy as np
import torch

from riddle.production import NumericalFitError, score_diagnostics


@torch.no_grad()
def native_coordinates(model, prepared, batch_size=None):
    x, mass = prepared["tensor2"], prepared["labels"]
    if batch_size is None:
        return model(x, mass)[0]
    return torch.cat([model(x[i:i + batch_size], mass[i:i + batch_size])[0]
                      for i in range(0, len(x), batch_size)])


@torch.no_grad()
def frozen_coordinates(model, prepared, batch_size=None):
    x, mass = prepared["tensor2"], prepared["labels"]
    preprocessing_mask = prepared["mask"].detach().cpu().numpy()
    if not len(x) or len(x) != int(preprocessing_mask.sum()) or len(x) != len(mass):
        raise ValueError("Empty or misaligned LaCATHODE evaluation inputs")
    if model.training:
        raise ValueError("LaCATHODE evaluation requires a frozen flow in evaluation mode")
    z = native_coordinates(model, prepared, batch_size)
    finite = torch.isfinite(z).all(dim=1)
    retry_rows = np.flatnonzero(preprocessing_mask)[(~finite).cpu().numpy()]
    if not len(retry_rows):
        return z.cpu().numpy(), preprocessing_mask, retry_rows
    retry = ~finite
    if not (torch.isfinite(x[retry]).all() and torch.isfinite(mass[retry]).all()
            and all(torch.isfinite(t).all() for t in model.state_dict().values())):
        raise NumericalFitError("Nonfinite LaCATHODE flow input or checkpoint before evaluation retry")
    wide = deepcopy(model).double().eval()
    recovered = wide(x[retry].double(), mass[retry].double())[0]
    recovered_finite = torch.isfinite(recovered).all(dim=1)
    if not torch.isfinite(recovered[recovered_finite].to(z.dtype)).all():
        z = z.double()
    z[retry] = recovered.to(z.dtype)
    finite[retry] = recovered_finite
    if not finite.any():
        raise NumericalFitError("No finite LaCATHODE evaluation coordinates after the float64 retry")
    mask = preprocessing_mask.copy()
    mask[preprocessing_mask] = finite.cpu().numpy()
    return z[finite].cpu().numpy(), mask, retry_rows


@torch.no_grad()
def frozen_predictions(classifier, coordinates, stage, batch_size=None):
    parameter = next(classifier.parameters())
    x = torch.as_tensor(coordinates, device=parameter.device)

    def predict(network, inputs):
        if batch_size is None:
            return network(inputs).reshape(-1)
        return torch.cat([network(inputs[i:i + batch_size]).reshape(-1)
                          for i in range(0, len(inputs), batch_size)])

    if x.dtype == parameter.dtype:
        scores = predict(classifier, x)
    else:
        native = x.to(parameter.dtype)
        representable = torch.isfinite(native).all(dim=1)
        if representable.all():
            scores = predict(classifier, native)
        else:
            scores = torch.empty(len(x), dtype=torch.float64, device=x.device)
            if representable.any():
                scores[representable] = predict(classifier, native[representable]).double()
            scores[~representable] = predict(deepcopy(classifier).double().eval(), x[~representable].double())
    result = scores.cpu().numpy()
    score_diagnostics(result, stage=stage, probability=True, summarize=False)
    return result


def mapping_audit(preprocessing_mask, mask, retry_rows, mass, names, sizes):
    rejected = np.flatnonzero(preprocessing_mask & ~mask)
    ends = np.cumsum(sizes)
    events = []
    for row in rejected:
        source = int(np.searchsorted(ends, row, side="right"))
        start = int(ends[source - 1]) if source else 0
        events.append(dict(row=int(row), source=names[source], source_row=int(row) - start,
                           mass_tev=float(mass[row])))
    return dict(preprocessing_accepted=int(preprocessing_mask.sum()), scorable=int(mask.sum()),
                float64_retried=len(retry_rows), float64_recovered=len(retry_rows) - len(rejected),
                unscorable=len(rejected), unscorable_events=events)
