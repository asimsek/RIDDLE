from contextlib import contextmanager


PROTOCOL = "bounded_affine_background_v1"
LOG_SCALE_BOUND = 2.0


def stability_contract():
    return {
        "protocol": PROTOCOL,
        "log_scale_bound": LOG_SCALE_BOUND,
        "scale_transform": "bound * tanh(raw_log_scale / bound)",
        "scope": "background MADE training, density evaluation and inverse sampling",
    }


def bounded_forward(self, inputs, cond_inputs=None, mode="direct"):
    import torch

    if mode == "direct":
        shift, raw_scale = self.trunk(self.joiner(inputs, cond_inputs)).chunk(2, 1)
        scale = LOG_SCALE_BOUND * torch.tanh(raw_scale / LOG_SCALE_BOUND)
        return (inputs - shift) * torch.exp(-scale), -scale.sum(-1, keepdim=True)
    if mode != "inverse":
        raise ValueError("Unknown background affine-flow direction")
    values = torch.zeros_like(inputs)
    for index in range(inputs.shape[1]):
        shift, raw_scale = self.trunk(self.joiner(values, cond_inputs)).chunk(2, 1)
        scale = LOG_SCALE_BOUND * torch.tanh(raw_scale / LOG_SCALE_BOUND)
        column = inputs[:, index] * torch.exp(scale[:, index]) + shift[:, index]
        values = values.clone()
        values[:, index] = column
    return values, scale.sum(-1, keepdim=True)


@contextmanager
def bounded_background(flows):
    original = flows.MADE.forward
    flows.MADE.forward = bounded_forward
    try:
        yield
    finally:
        flows.MADE.forward = original
