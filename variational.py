import torch


def reparameterize(mean, covar, z, full_cov=False):
    """
    Implements the 'reparameterization trick' for the Gaussian, either full rank or diagonal
    If the z is a sample from N(0, 1), the output is a sample from N(mean, var)
    If full_cov=True then var must be of shape S,N,N,D and the full covariance is used. Otherwise
    var must be S,N,D and the operation is elementwise
    :param mean: mean of shape S,N,D
    :param var: covariance of shape S,N,D or S,N,N,D
    :param z: samples form unit Gaussian of shape S,N,D
    :param full_cov: bool to indicate whether var is of shape S,N,N,D or S,N,D
    :return sample from N(mean, var) of shape S,N,D
    """
    if covar is None:
        return mean

    return mean + z * covar ** 0.5

    # S, N, D = mean.shape[0], mean.shape[1], mean.shape[2]
    # mean = mean.permute(0, 2, 1)  # SND -> SDN
    # covar = covar.permute(0, 3, 1, 2)  # SNND -> SDNN
    # I = torch.eye(N, dtype=torch.float).expand(1, 1, -1, -1)  # 11NN
    # chol = torch.cholesky(covar + I)  # SDNN
    # z_SDN1 = z.permute(0, 2, 1).unsqueeze(-1)  # SND->SDN1
    # f = mean + torch.matmul(chol, z_SDN1)[:, :, :, 0]  # SDN(1)
    # return f.permute(0, 2, 1)  # SND
