import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Normal


def get_acq_fn(acq_fn='none'):
    if acq_fn.lower() == "ucb":
        return UCB
    elif acq_fn.lower() == "ei":
        return EI
    else:
        return NoAF


class AcquisitionFunctionWrapper():
    def __init__(self, model, l2r, dataset, device):
        self.model = model
        self.l2r = l2r

    def __call__(self, x):
        raise NotImplementedError()
    
    def update(self, data):
        self.fit(data)

    def fit(self, data):
        self.model.fit(data, reset=True)

class NoAF(AcquisitionFunctionWrapper):
    def __call__(self, x):
        return self.l2r(self.model(x))

class UCB(AcquisitionFunctionWrapper):
    def __init__(self, model, l2r, dataset, device='cpu', kappa=0.1):
        super().__init__(model, l2r, dataset, device)
        self.kappa = kappa
        self.model.to(device)
    
    def __call__(self, x):
        mean, std = self.model.forward_with_uncertainty(x)
        return self.l2r(mean + self.kappa * std)

class EI(AcquisitionFunctionWrapper):
    def __init__(self, model, l2r, dataset, device='cpu', proxy_type="regression", max_percentile=80):
        super().__init__(model, l2r, dataset, device)
        self.device = device
        self.sigmoid = nn.Sigmoid()
        self.proxy_type = proxy_type
        self.max_percentile = max_percentile

    def _get_best_f(self, dataset):
        f_values = []
        data_it = dataset.pos_train if self.proxy_type == "classification" else dataset.train
        for sample in data_it:
            outputs = self.model([sample])
            f_values.append(outputs.item())
        return torch.tensor(np.percentile(f_values, self.max_percentile))

    def __call__(self, x):
        self.best_f = self.best_f.to(self.device)
        mean, sigma = self.model.forward_with_uncertainty(x)
        u = (mean - self.best_f.expand_as(mean)) / sigma

        normal = Normal(torch.zeros_like(u), torch.ones_like(u))
        ucdf = normal.cdf(u)
        updf = torch.exp(normal.log_prob(u))
        ei = sigma * (updf + u * ucdf)
        return ei

    def update(self, data):
        super().fit(data)
        self.best_f = self._get_best_f(data)
