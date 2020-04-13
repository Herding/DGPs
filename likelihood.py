import numpy as np
import math

import torch
from torch.nn.functional import one_hot


def hermgauss(n):
    x, w = np.polynomial.hermite.hermgauss(n)
    x, w = x.astype(np.float), w.astype(np.float)
    return x, w


class Likelihood(object):
    def __init__(self):
        self.num_gauss_hermite_points = 20


class BroadcastingLikelihood(object):
    """
    A wrapper for the likelihood to broadcast over the samples dimension. The Gaussian doesn't
    need this, but for the others we can apply reshaping and tiling.
    With this wrapper all likelihood functions behave correctly with inputs of shape S,N,D,
    but with Y still of shape N,D
    """
    def __init__(self, likelihood):
        self.likelihood = likelihood
        self.needs_broadcasting = True

        # if isinstance(likelihood, Gaussian):
        #     self.needs_broadcasting = False
        # else:
        #     self.needs_broadcasting = True

    def _broadcast(self, f, vars_SND, vars_ND):
        if self.needs_broadcasting is False:
            return f(vars_SND, [torch.unsqueeze(v, 0) for v in vars_ND])

        else:
            S, N, D = vars_SND[0].shape[0], vars_SND[0].shape[1], vars_SND[0].shape[2]
            vars_tiled = [x.expand([S, -1, -1]) for x in vars_ND]

            flattened_SND = [torch.reshape(x, [S*N, D]) for x in vars_SND]
            flattened_tiled = [torch.reshape(x, [S*N, -1]) for x in vars_tiled]

            flattened_result = f(flattened_SND, flattened_tiled)
            if isinstance(flattened_result, tuple):
                return [torch.reshape(x, [S, N, -1]) for x in flattened_result]
            else:
                return torch.reshape(flattened_result, [S, N, -1])

    def var_exp(self, Fmu, Fvar, Y):
        f = lambda vars_SND, vars_ND: self.likelihood.var_exp(vars_SND[0], vars_SND[1], vars_ND[0])
        return self._broadcast(f, [Fmu, Fvar], [Y])

    def logp(self, F, Y):
        f = lambda vars_SND, vars_ND: self.likelihood.logp(vars_SND[0], vars_ND[0])
        return self._broadcast(f, [F], [Y])

    def conditional_mean(self, F):
         f = lambda vars_SND, vars_ND: self.likelihood.conditional_mean(vars_SND[0])
         return self._broadcast(f, [F], [])

    def conditional_variance(self, F):
         f = lambda vars_SND, vars_ND: self.likelihood.conditional_variance(vars_SND[0])
         return self._broadcast(f, [F], [])

    def predict_mean_and_var(self, Fmu, Fvar):
         f = lambda vars_SND, vars_ND: self.likelihood.predict_mean_and_var(vars_SND[0],
                                                                             vars_SND[1])
         return self._broadcast(f, [Fmu, Fvar], [])

    def predict_density(self, Fmu, Fvar, Y):
        f = lambda vars_SND, vars_ND: self.likelihood.predict_density(vars_SND[0],
                                                                       vars_SND[1],
                                                                       vars_ND[0])
        return self._broadcast(f, [Fmu, Fvar], [Y])


class RobustMax(object):
    """
    This class represent a multi-class inverse-link function. Given a vector
    f=[f_1, f_2, ... f_k], the result of the mapping is

    y = [y_1 ... y_k]

    with

    y_i = (1-eps)  i == argmax(f)
          eps/(k-1)  otherwise.


    """

    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, F):
        _, i = torch.max(F, 1)
        return one_hot(i, self.num_classes)

    def prob_is_largest(self, Y, mu, var, gh_x, gh_w):
        Y = Y.long()
        # work out what the mean and variance is of the indicated latent function.
        oh_on = one_hot(torch.reshape(Y, (-1,)), self.num_classes).float()
        mu_selected = torch.sum(oh_on * mu, 1)
        var_selected = torch.sum(oh_on * var, 1)

        # generate Gauss Hermite grid
        X = torch.reshape(mu_selected, (-1, 1)) + gh_x * torch.reshape(
            torch.sqrt(torch.clamp(var_selected * 2, 1e-10, float('inf'))), (-1, 1))

        # compute the CDF of the Gaussian between the latent functions and the grid (including the selected function)
        dist = (torch.unsqueeze(X, 1) - torch.unsqueeze(mu, 2)) / torch.unsqueeze(
            torch.sqrt(torch.clamp(var, 1e-10, float('inf'))), 2)
        cdfs = 0.5 * (1.0 + torch.erf(dist / np.sqrt(2.0)))

        cdfs = cdfs * (1 - 2e-4) + 1e-4

        # blank out all the distances on the selected latent function
        oh_off = (1 - one_hot(torch.reshape(Y, (-1,)), self.num_classes)).float()
        cdfs = cdfs * torch.unsqueeze(oh_off, 2) + torch.unsqueeze(oh_on, 2)

        # take the product over the latent functions, and the sum over the GH grid.
        return torch.matmul(torch.prod(cdfs, 1), torch.reshape(gh_w / math.sqrt(math.pi), (-1, 1)))


class MultiClass(Likelihood):
    def __init__(self, num_classes, invlink=None):
        """
        A likelihood that can do multi-way classification.
        Currently the only valid choice
        of inverse-link function (invlink) is an instance of RobustMax.
        """
        Likelihood.__init__(self)
        self.num_classes = num_classes
        if invlink is None:
            invlink = RobustMax(self.num_classes)
        elif not isinstance(invlink, RobustMax):
            raise NotImplementedError
        self.invlink = invlink

    def logp(self, F, Y):
        if isinstance(self.invlink, RobustMax):
            _, i = torch.max(F, 1)
            hits = torch.eq(torch.unsqueeze(i, 1), Y.long())
            yes = torch.ones(Y.shape, dtype=torch.float)
            no = torch.zeros(Y.shape, dtype=torch.float)
            p = torch.where(hits, yes, no)
            return torch.log(p)
        else:
            raise NotImplementedError

    def var_exp(self, Fmu, Fvar, Y):
        if isinstance(self.invlink, RobustMax):
            gh_x, gh_w = hermgauss(self.num_gauss_hermite_points)
            p = self.invlink.prob_is_largest(Y, Fmu, Fvar, torch.from_numpy(gh_x), torch.from_numpy(gh_w))
            return p
        else:
            raise NotImplementedError

    def predict_mean_and_var(self, Fmu, Fvar):
        if isinstance(self.invlink, RobustMax):
            # To compute this, we'll compute the density for each possible output
            possible_outputs = [torch.full([Fmu.shape[0], 1], i, dtype=torch.long) for i in
                                range(self.num_classes)]
            ps = [self._predict_non_logged_density(Fmu, Fvar, po) for po in possible_outputs]
            ps = torch.stack([torch.reshape(p, (-1,)) for p in ps])
            return ps.T, ps.T - torch.pow(ps.T, 2)
        else:
            raise NotImplementedError

    def predict_density(self, Fmu, Fvar, Y):
        return torch.log(self._predict_non_logged_density(Fmu, Fvar, Y))

    def _predict_non_logged_density(self, Fmu, Fvar, Y):
        if isinstance(self.invlink, RobustMax):
            gh_x, gh_w = hermgauss(self.num_gauss_hermite_points)
            p = self.invlink.prob_is_largest(Y, Fmu, Fvar, torch.from_numpy(gh_x), torch.from_numpy(gh_w))
            return p
        else:
            raise NotImplementedError
    #
    # def conditional_mean(self, F):
    #     return self.invlink(F)
    #
    # def conditional_variance(self, F):
    #     p = self.conditional_mean(F)
    #     return p - tf.square(p)