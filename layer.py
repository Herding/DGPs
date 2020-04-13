import torch
from torch import nn
from torch.distributions.kl import kl_divergence
from torch.distributions.multivariate_normal import MultivariateNormal

from dgps.variational import reparameterize


class Layer(nn.Module):
    def __init__(self, kern, inducing_points, output_dims, mean_function):
        super(Layer, self).__init__()
        self.kern = kern
        self.num_inducing = inducing_points.shape[0]
        self.num_outputs = output_dims
        self.mean_fun = mean_function

        q_mu = torch.zeros((self.num_inducing, self.num_outputs))

        Lu = torch.cholesky(self.kern.Ku + torch.eye(inducing_points.shape[0]))
        q_covar = Lu

        self.register_parameter(name="variational_mean", param=torch.nn.Parameter(q_mu, requires_grad=True))

        for idx in range(self.num_outputs):
            self.register_parameter(name=f"covar{idx}", param=torch.nn.Parameter(q_covar, requires_grad=True))

        self.register_parameter(name='inducing_points', param=torch.nn.Parameter(inducing_points, requires_grad=True))

    def conditional_ND(self, inputs):
        Kuf = self.kern(self.inducing_points, inputs)
        A = torch.triangular_solve(Kuf, self.Lu, upper=False)[0]
        A = torch.triangular_solve(A, self.Lu.T)[0]

        mean = torch.matmul(A.T, self.variational_mean)
        A_expanded = A.expand([self.num_outputs, -1, -1])
        SK = -self.Ku.expand([self.num_outputs, -1, -1])
        SK += torch.matmul(self.variational_covar, self.variational_covar.T)

        B = torch.matmul(SK, A_expanded)

        delta_covar = torch.sum(A_expanded * B, 1)
        Kff = self.kern(inputs, diagal=True)

        covar = Kff.unsqueeze(0) + delta_covar
        # return mean, covar.T
        return mean + self.mean_fun(inputs), covar.T

    def conditional_SND(self, inputs):
        S, N, D = inputs.shape[0], inputs.shape[1], inputs.shape[2]
        X_flat = torch.reshape(inputs, [S * N, D])
        mean, var = self.conditional_ND(X_flat)
        return [torch.reshape(m, [S, N, self.num_outputs]) for m in [mean, var]]

    def KL(self):
        Lu_expand = self.Lu.expand([self.num_outputs, -1, -1])

        KL = -0.5 * self.num_outputs * self.num_inducing
        KL -= 0.5 * torch.sum(torch.log(torch.diagonal(self.variational_covar, dim1=-2, dim2=-1) ** 2))

        KL += torch.sum(torch.log(torch.diag(self.Lu))) * self.num_outputs
        KL += 0.5 * torch.sum(torch.pow(torch.triangular_solve(self.variational_covar, Lu_expand, upper=False)[0], 2))
        Kinv_m = torch.cholesky_solve(self.variational_mean, self.Lu)
        KL += 0.5 * torch.sum(self.variational_mean * Kinv_m)

        return KL

    def forward(self, inputs):
        S = inputs.shape[0]
        N = inputs.shape[1]
        D = self.num_outputs
        var_covars = []

        self.Ku = self.kern(self.inducing_points) + torch.eye(self.num_inducing)
        self.Lu = torch.cholesky(self.Ku)

        for idx in range(self.num_outputs):
            var_covars.append(self.__getattr__(f'covar{idx}'))
        self.variational_covar = torch.stack(var_covars)

        means, covars = self.conditional_SND(inputs)

        means = torch.reshape(means, [S, N, D])
        covars = torch.reshape(covars, [S, N, D])

        samples = reparameterize(means, covars, torch.randn(means.shape, dtype=torch.float))
        return samples, means, covars
