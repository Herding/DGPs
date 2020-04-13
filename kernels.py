import torch
import torch.nn as nn

class Kernel(nn.Module):
    """
    The basic kernel class. Handles input_dim and active dims, and provides a
    generic '_slice' function to implement them.
    """

    def __init__(self, input_dim, active_dims=None, name=None):
        """
        input dim is an integer
        active dims is either an iterable of integers or None.

        Input dim is the number of input dimensions to the kernel. If the
        kernel is computed on a matrix X which has more columns than input_dim,
        then by default, only the first input_dim columns are used. If
        different columns are required, then they may be specified by
        active_dims.

        If active dims is None, it effectively defaults to range(input_dim),
        but we store it as a slice for efficiency.
        """
        super(Kernel, self).__init__()
        self.input_dim = int(input_dim)

        if active_dims is None:
            self.active_dims = slice(input_dim)
        # elif isinstance(active_dims, slice):
        #     self.active_dims = active_dims
        #     if active_dims.start is not None and active_dims.stop is not None and active_dims.step is not None:
        #         assert len(range(*active_dims)) == input_dim  # pragma: no cover
        # else:
        #     self.active_dims = np.array(active_dims, dtype=np.int32)
        #     assert len(active_dims) == input_dim

        self.num_gauss_hermite_points = 20

    # def compute_K(self, X, Z):
    #     return self.K(X, Z)
    #
    # def compute_K_symm(self, X, X2=None):
    #     return self.variance * torch.exp(-self.scaled_square_dist(X, X2) / 2)
    #
    # def compute_Kdiag(self, X):
    #     return self.Kdiag(X)
    #
    # def on_separate_dims(self, other_kernel):
    #     """
    #     Checks if the dimensions, over which the kernels are specified, overlap.
    #     Returns True if they are defined on different/separate dimensions and False otherwise.
    #     """
    #     if isinstance(self.active_dims, slice) or isinstance(other_kernel.active_dims, slice):
    #         # Be very conservative for kernels defined over slices of dimensions
    #         return False
    #
    #     if np.any(self.active_dims.reshape(-1, 1) == other_kernel.active_dims.reshape(1, -1)):
    #         return False
    #
    #     return True

    # def _slice(self, X, X2):
    #     """
    #     Slice the correct dimensions for use in the kernel, as indicated by
    #     `self.active_dims`.
    #     :param X: Input 1 (NxD).
    #     :param X2: Input 2 (MxD), may be None.
    #     :return: Sliced X, X2, (Nxself.input_dim).
    #     """
    #     if isinstance(self.active_dims, slice):
    #         X = X[:, self.active_dims]
    #         if X2 is not None:
    #             X2 = X2[:, self.active_dims]
    #     else:
    #         X = tf.gather(X.T, self.active_dims).T
    #         if X2 is not None:
    #             X2 = tf.gather(X2.T, self.active_dims).T
    #     input_dim_shape = X.shape[1]
    #     input_dim = tf.convert_to_tensor(self.input_dim, dtype=torch.int)
    #     with tf.control_dependencies([tf.assert_equal(input_dim_shape, input_dim)]):
    #         X = tf.identity(X)
    #
    #     return X, X2
    #
    # def _slice_cov(self, cov):
    #     """
    #     Slice the correct dimensions for use in the kernel, as indicated by
    #     `self.active_dims` for covariance matrices. This requires slicing the
    #     rows *and* columns. This will also turn flattened diagonal
    #     matrices into a tensor of full diagonal matrices.
    #     :param cov: Tensor of covariance matrices (NxDxD or NxD).
    #     :return: N x self.input_dim x self.input_dim.
    #     """
    #     cov = tf.cond(tf.equal(tf.rank(cov), 2), lambda: tf.matrix_diag(cov), lambda: cov)
    #
    #     if isinstance(self.active_dims, slice):
    #         cov = cov[..., self.active_dims, self.active_dims]
    #     else:
    #         cov_shape = tf.shape(cov)
    #         covr = tf.reshape(cov, [-1, cov_shape[-1], cov_shape[-1]])
    #         gather1 = tf.gather(tf.transpose(covr, [2, 1, 0]), self.active_dims)
    #         gather2 = tf.gather(tf.transpose(gather1, [1, 0, 2]), self.active_dims)
    #         cov = tf.reshape(tf.transpose(gather2, [2, 0, 1]),
    #                          tf.concat([cov_shape[:-2], [len(self.active_dims), len(self.active_dims)]], 0))
    #     return cov
    #
    # def __add__(self, other):
    #     return Sum([self, other])
    #
    # def __mul__(self, other):
    #     return Product([self, other])


class Stationary(Kernel):
    """
    Base class for kernels that are stationary, that is, they only depend on

        r = || x - x' ||

    This class handles 'ARD' behaviour, which stands for 'Automatic Relevance
    Determination'. This means that the kernel has one lengthscale per
    dimension, otherwise the kernel is isotropic (has a single lengthscale).
    """

    def __init__(self, input_dim, variance=1.0, lengthscales=None,
                 active_dims=None, ARD=False, name=None):
        """
        - input_dim is the dimension of the input to the kernel
        - variance is the (initial) value for the variance parameter
        - lengthscales is the initial value for the lengthscales parameter
          defaults to 1.0 (ARD=False) or np.ones(input_dim) (ARD=True).
        - active_dims is a list of length input_dim which controls which
          columns of X are used.
        - ARD specifies whether the kernel has one lengthscale per dimension
          (ARD=True) or a single lengthscale (ARD=False).
        """
        super().__init__(input_dim, active_dims, name=name)
        self.register_parameter(name="variance", param=torch.nn.Parameter(torch.tensor(variance), requires_grad=True))
        if ARD:
            if lengthscales is None:
                lengthscales = torch.ones(input_dim, dtype=torch.float)
            else:
                # accepts float or array:
                lengthscales = lengthscales * torch.ones(input_dim, dtype=torch.float)
            self.register_parameter(name="lengthscales",
                                    param=torch.nn.Parameter(torch.tensor(lengthscales), requires_grad=True))
            self.ARD = True
        else:
            if lengthscales is None:
                lengthscales = 1.0
            self.register_parameter(name="lengthscales",
                                    param=torch.nn.Parameter(torch.tensor(lengthscales), requires_grad=True))
            self.ARD = False

    def square_dist(self, X, X2):  # pragma: no cover
        return self.scaled_square_dist(X, X2)

    def euclid_dist(self, X, X2):  # pragma: no cover
        return self.scaled_euclid_dist(X, X2)

    def scaled_square_dist(self, X, X2):
        """
        Returns ((X - X2ᵀ)/lengthscales)².
        Due to the implementation and floating-point imprecision, the
        result may actually be very slightly negative for entries very
        close to each other.
        """
        X = X / self.lengthscales
        Xs = torch.sum(torch.pow(X, 2), 1)

        if X2 is None:
            dist = -2 * torch.matmul(X, X.T)
            dist += torch.reshape(Xs, (-1, 1)) + torch.reshape(Xs, (1, -1))
            return dist

        X2 = X2 / self.lengthscales
        X2s = torch.sum(torch.pow(X2, 2), 1)
        dist = -2 * torch.matmul(X, X2.T)
        dist += torch.reshape(Xs, (-1, 1)) + torch.reshape(X2s, (1, -1))
        return dist

    def scaled_euclid_dist(self, X, X2):
        """
        Returns |(X - X2ᵀ)/lengthscales| (L2-norm).
        """
        r2 = self.scaled_square_dist(X, X2)
        return torch.sqrt(r2 + 1e-12)

    def Kdiag(self, X, presliced=False):
        return torch.full([X.shape[0]], self.variance.squeeze().item())


class RBF(Stationary):
    """
    The radial basis function (RBF) or squared exponential kernel
    """
    def __init__(self, input_dim, X, variance=1.0, lengthscales=None,
                 active_dims=None, ARD=False, name=None):

        super().__init__(input_dim, variance, lengthscales, active_dims, ARD, name)
        X = X / lengthscales
        Xs = torch.sum(torch.pow(X, 2), 1)
        dist = -2 * torch.matmul(X, X.T)
        dist += torch.reshape(Xs, (-1, 1)) + torch.reshape(Xs, (1, -1))
        self.Ku = variance * torch.exp(-dist / 2)

    def forward(self, X, X2=None, diagal=False):
        # if not presliced:
        #     X, X2 = self._slice(X, X2)
        if diagal:
            return self.Kdiag(X)

        return self.variance * torch.exp(-self.scaled_square_dist(X, X2) / 2)
