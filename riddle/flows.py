import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_mask(in_features, out_features, in_flow_features, mask_type=None):
    if mask_type == "input":
        in_degrees = torch.arange(in_features) % in_flow_features
    else:
        in_degrees = torch.arange(in_features) % (in_flow_features - 1)
    if mask_type == "output":
        out_degrees = torch.arange(out_features) % in_flow_features - 1
    else:
        out_degrees = torch.arange(out_features) % (in_flow_features - 1)
    return (out_degrees.unsqueeze(-1) >= in_degrees.unsqueeze(0)).float()


class MaskedLinear(nn.Module):
    def __init__(self, in_features, out_features, mask, cond_in_features=None, bias=True):
        super(MaskedLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        if cond_in_features is not None:
            self.cond_linear = nn.Linear(cond_in_features, out_features, bias=False)
        self.register_buffer("mask", mask)

    def forward(self, inputs, cond_inputs=None):
        output = F.linear(inputs, self.linear.weight * self.mask, self.linear.bias)
        if cond_inputs is not None:
            output += self.cond_linear(cond_inputs)
        return output


class MADE(nn.Module):
    def __init__(self, num_inputs, num_hidden, num_cond_inputs=None, act="relu", pre_exp_tanh=False,
                 affine_log_scale_bound=None):
        super(MADE, self).__init__()
        if type(pre_exp_tanh) is not bool:
            raise ValueError("pre_exp_tanh must be boolean")
        # An explicit bound takes precedence; the legacy switch means tanh(a).
        if affine_log_scale_bound is None and pre_exp_tanh:
            affine_log_scale_bound = 1.
        if affine_log_scale_bound is not None and (
            type(affine_log_scale_bound) not in (int, float)
            or not math.isfinite(affine_log_scale_bound) or affine_log_scale_bound <= 0
        ):
            raise ValueError("affine_log_scale_bound must be finite and positive")
        self.affine_log_scale_bound = affine_log_scale_bound
        activations = {"relu": nn.ReLU, "sigmoid": nn.Sigmoid, "tanh": nn.Tanh}
        act_func = activations[act]
        input_mask = get_mask(num_inputs, num_hidden, num_inputs, mask_type="input")
        hidden_mask = get_mask(num_hidden, num_hidden, num_inputs)
        output_mask = get_mask(num_hidden, num_inputs * 2, num_inputs, mask_type="output")
        self.joiner = MaskedLinear(num_inputs, num_hidden, input_mask, num_cond_inputs)
        self.trunk = nn.Sequential(
            act_func(),
            MaskedLinear(num_hidden, num_hidden, hidden_mask),
            act_func(),
            MaskedLinear(num_hidden, num_inputs * 2, output_mask),
        )

    def bounded_log_scale(self, a):
        bound = self.affine_log_scale_bound
        return a if bound is None else bound * torch.tanh(a / bound)

    def forward(self, inputs, cond_inputs=None, mode="direct"):
        if mode == "direct":
            h = self.joiner(inputs, cond_inputs)
            m, a = self.trunk(h).chunk(2, 1)
            a = self.bounded_log_scale(a)
            u = (inputs - m) * torch.exp(-a)
            return (u, -a.sum(-1, keepdim=True))
        else:
            x = torch.zeros_like(inputs)
            for i_col in range(inputs.shape[1]):
                h = self.joiner(x, cond_inputs)
                m, a = self.trunk(h).chunk(2, 1)
                a = self.bounded_log_scale(a)
                x[:, i_col] = inputs[:, i_col] * torch.exp(a[:, i_col]) + m[:, i_col]
            return (x, a.sum(-1, keepdim=True))


class BatchNormFlow(nn.Module):
    def __init__(self, num_inputs, momentum=0.0, eps=1e-05):
        super(BatchNormFlow, self).__init__()
        self.log_gamma = nn.Parameter(torch.zeros(num_inputs))
        self.beta = nn.Parameter(torch.zeros(num_inputs))
        self.momentum = momentum
        self.eps = eps
        self.register_buffer("running_mean", torch.zeros(num_inputs))
        self.register_buffer("running_var", torch.ones(num_inputs))

    def forward(self, inputs, cond_inputs=None, mode="direct"):
        if mode == "direct":
            if self.training:
                self.batch_mean = inputs.mean(0)
                self.batch_var = (inputs - self.batch_mean).pow(2).mean(0) + self.eps
                self.running_mean.mul_(self.momentum)
                self.running_var.mul_(self.momentum)
                self.running_mean.add_(self.batch_mean.data * (1 - self.momentum))
                self.running_var.add_(self.batch_var.data * (1 - self.momentum))
                mean = self.batch_mean
                var = self.batch_var
            else:
                mean = self.running_mean
                var = self.running_var
            x_hat = (inputs - mean) / var.sqrt()
            y = torch.exp(self.log_gamma) * x_hat + self.beta
            return (y, (self.log_gamma - 0.5 * torch.log(var)).sum(-1, keepdim=True))
        else:
            if self.training:
                mean = self.batch_mean
                var = self.batch_var
            else:
                mean = self.running_mean
                var = self.running_var
            x_hat = (inputs - self.beta) / torch.exp(self.log_gamma)
            y = x_hat * var.sqrt() + mean
            return (y, (-self.log_gamma + 0.5 * torch.log(var)).sum(-1, keepdim=True))


class Reverse(nn.Module):
    def __init__(self, num_inputs):
        super(Reverse, self).__init__()
        self.perm = np.array(np.arange(0, num_inputs)[::-1])
        self.inv_perm = np.argsort(self.perm)

    def forward(self, inputs, cond_inputs=None, mode="direct"):
        if mode == "direct":
            return (inputs[:, self.perm], torch.zeros(inputs.size(0), 1, device=inputs.device))
        else:
            return (inputs[:, self.inv_perm], torch.zeros(inputs.size(0), 1, device=inputs.device))


class FlowSequential(nn.Sequential):
    def forward(self, inputs, cond_inputs=None, mode="direct", logdets=None):
        self.num_inputs = inputs.size(-1)
        if logdets is None:
            logdets = torch.zeros(inputs.size(0), 1, device=inputs.device)
        assert mode in ["direct", "inverse"]
        if mode == "direct":
            for module in self._modules.values():
                inputs, logdet = module(inputs, cond_inputs, mode)
                logdets += logdet
        else:
            for module in reversed(self._modules.values()):
                inputs, logdet = module(inputs, cond_inputs, mode)
                logdets += logdet
        return (inputs, logdets)

    def log_probs(self, inputs, cond_inputs=None):
        u, log_jacob = self(inputs, cond_inputs)
        log_probs = (-0.5 * u.pow(2) - 0.5 * math.log(2 * math.pi)).sum(-1, keepdim=True)
        return (log_probs + log_jacob).sum(-1, keepdim=True)

    def sample(self, num_samples=None, noise=None, cond_inputs=None):
        if noise is None:
            noise = torch.Tensor(num_samples, self.num_inputs).normal_()
        device = next(self.parameters()).device
        noise = noise.to(device)
        if cond_inputs is not None:
            cond_inputs = cond_inputs.to(device)
        samples = self.forward(noise, cond_inputs, mode="inverse")[0]
        return samples
