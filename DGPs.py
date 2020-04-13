import torch
from torch import nn

from dgps.likelihood import BroadcastingLikelihood


class DGPs(nn.Module):
    def __init__(self, likelihood, layers,
                 minibatch_size=None, num_samples=1, num_outputs=None):
        super(DGPs, self).__init__()
        self.likelihood = BroadcastingLikelihood(likelihood)
        self.layers = nn.ModuleList()
        self.batch_size = minibatch_size
        self.num_samples = num_samples

        for layer in layers:
            self.layers.append(layer)

    def forward(self, inputs):
        f = inputs.expand([self.num_samples, -1, -1])

        fs, means, covars = [], [], []

        for idx in range(len(self.layers)):
            f, mean, covar = self.layers[idx](f)

            fs.append(f)
            means.append(mean)
            covars.append(covar)

        # return f, mean, covar
        return fs, means, covars

    def exp_log_p(self, mean, covar, Y):
        var_exp = self.likelihood.var_exp(mean, covar, Y)
        return var_exp.mean(0)

    def m_likelihood(self, mean, covar, Y):
        L = self.exp_log_p(mean, covar, Y).sum()
        KLs = torch.tensor([layer.KL() for layer in self.layers])
        KL = torch.sum(KLs)
        return L.div(self.batch_size) - KL
