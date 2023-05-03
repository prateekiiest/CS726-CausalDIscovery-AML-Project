
import numpy as np
from sklearn.utils import check_array
from sklearn.preprocessing import scale

from .utils.base import _BaseLiNGAM
from castle.common import BaseLearner, Tensor
from castle.common.consts import DIRECT_LINGAM_VALID_PARAMS
from castle.common.validator import check_args_value


class DirectLiNGAM(_BaseLiNGAM, BaseLearner):
    """
    DirectLiNGAM Algorithm.
    A direct learning algorithm for linear non-Gaussian acyclic model (LiNGAM).
    Implementation of DirectLiNGAM Algorithm [1]_ [2]_, Construct a DirectLiNGAM model.

    Parameters
    ----------
    prior_knowledge : array-like, shape (n_features, n_features), optional (default=None)
        Prior knowledge used for causal discovery, where ``n_features`` is the number of features.

        The elements of prior knowledge matrix are defined as follows [1]_:

        * ``0`` : :math:`x_i` does not have a directed path to :math:`x_j`
        * ``1`` : :math:`x_i` has a directed path to :math:`x_j`
        * ``-1`` : No prior knowledge is available to know if either of the two cases above (0 or 1) is true.
    measure : {'pwling', 'kernel'}, default='pwling'
        Measure to evaluate independence: 'pwling' [2]_ or 'kernel' [1]_.
    thresh : float,  default='0.3'
        Drop edge if |weight| < threshold

    Attributes
    ----------
    causal_matrix : numpy.ndarray
        Learned causal structure matrix.
    weight_causal_matrix: numpy.ndarray
        Learned weighted causal structure matrix.

    References
    ----------
    .. [1] S. Shimizu, T. Inazumi, Y. Sogawa, A. Hyvärinen, Y. Kawahara, T. Washio, P. O. Hoyer and K. Bollen. 
       DirectLiNGAM: A direct method for learning a linear non-Gaussian structural equation model. 
       Journal of Machine Learning Research, 12(Apr): 1225--1248, 2011.
    .. [2] A. Hyvärinen and S. M. Smith. Pairwise likelihood ratios for estimation of non-Gaussian structural eauation models. 
       Journal of Machine Learning Research 14:111-152, 2013. 

    Examples
    --------
    >>> from castle.algorithms import DirectLiNGAM
    >>> from castle.datasets import load_dataset
    >>> from castle.common import GraphDAG
    >>> from castle.metrics import MetricsDAG
    >>> X, true_dag, _ = load_dataset(name='IID_Test')
    >>> n = DirectLiNGAM()
    >>> n.learn(X)
    >>> GraphDAG(n.causal_matrix, true_dag)
    >>> met = MetricsDAG(n.causal_matrix, true_dag)
    >>> print(met.metrics)
    """

    @check_args_value(DIRECT_LINGAM_VALID_PARAMS)
    def __init__(self, prior_knowledge=None, measure='pwling', thresh=0.3):

        super().__init__()

        self._prior_knowledge = prior_knowledge
        self._measure = measure
        self._thresh = thresh

    def learn(self, data, columns=None, **kwargs):
        """
        Set up and run the DirectLiNGAM algorithm.

        Parameters
        ----------
        data: castle.Tensor or numpy.ndarray
            The castle.Tensor or numpy.ndarray format data you want to learn.
        columns : Index or array-like
            Column labels to use for resulting tensor. Will default to
            RangeIndex (0, 1, 2, ..., n) if no column labels are provided.
        """

        X = Tensor(data, columns=columns)
    
        self.fit(X)
        
        weight_causal_matrix = self.adjacency_matrix_.T
        self.weight_causal_matrix = Tensor(weight_causal_matrix,
                                           index=X.columns,
                                           columns=X.columns)

        causal_matrix = (abs(self.adjacency_matrix_) > self._thresh).astype(int).T
        self.causal_matrix = Tensor(causal_matrix,
                                    index=X.columns,
                                    columns=X.columns)

    def fit(self, X):
        """
        Fit the model to X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where ``n_samples`` is the number of samples
            and ``n_features`` is the number of features.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        # Check parameters
        X = check_array(X)
        n_features = X.shape[1]

        if self._prior_knowledge is not None:
            self._Aknw = check_array(self._prior_knowledge)
            self._Aknw = np.where(self._Aknw < 0, np.nan, self._Aknw)
            if (n_features, n_features) != self._Aknw.shape:
                raise ValueError(
                    'The shape of prior knowledge must be (n_features, n_features)')
        else:
            self._Aknw = None

        # Causal discovery
        U = np.arange(n_features)
        K = []
        X_ = np.copy(X)
        if self._measure == 'kernel':
            X_ = scale(X_)

        for _ in range(n_features):
            if self._measure == 'kernel':
                m = self._search_causal_order_kernel(X_, U)
            else:
                m = self._search_causal_order(X_, U)
            for i in U:
                if i != m:
                    X_[:, i] = self._residual(X_[:, i], X_[:, m])
            K.append(m)
            U = U[U != m]

        self._causal_order = K
        return self._estimate_adjacency_matrix(X)

    def _residual(self, xi, xj):
        """The residual when xi is regressed on xj."""
        return xi - (np.cov(xi, xj)[0, 1] / np.var(xj)) * xj

    def _entropy(self, u):
        """Calculate entropy using the maximum entropy approximations."""
        k1 = 79.047
        k2 = 7.4129
        gamma = 0.37457
        return (1 + np.log(2 * np.pi)) / 2 - \
            k1 * (np.mean(np.log(np.cosh(u))) - gamma)**2 - \
            k2 * (np.mean(u * np.exp((-u**2) / 2)))**2

    def _diff_mutual_info(self, xi_std, xj_std, ri_j, rj_i):
        """Calculate the difference of the mutual informations."""
        return (self._entropy(xj_std) + self._entropy(ri_j / np.std(ri_j))) - \
               (self._entropy(xi_std) + self._entropy(rj_i / np.std(rj_i)))

    def _search_candidate(self, U):
        """ Search for candidate features """
        # If no prior knowledge is specified, nothing to do.
        if self._Aknw is None:
            return U, []

        # Find exogenous features
        Uc = []
        for j in U:
            index = U[U != j]
            if self._Aknw[j][index].sum() == 0:
                Uc.append(j)

        # Find endogenous features, and then find candidate features
        if len(Uc) == 0:
            U_end = []
            for j in U:
                index = U[U != j]
                if np.nansum(self._Aknw[j][index]) > 0:
                    U_end.append(j)

            # Find sink features (original)
            for i in U:
                index = U[U != i]
                if self._Aknw[index, i].sum() == 0:
                    U_end.append(i)
            Uc = [i for i in U if i not in set(U_end)]

        # make V^(j)
        Vj = []
        for i in U:
            if i in Uc:
                continue
            if self._Aknw[i][Uc].sum() == 0:
                Vj.append(i)
        return Uc, Vj

    def _search_causal_order(self, X, U):
        """Search the causal ordering."""
        Uc, Vj = self._search_candidate(U)
        if len(Uc) == 1:
            return Uc[0]

        M_list = []
        for i in Uc:
            M = 0
            for j in U:
                if i != j:
                    xi_std = (X[:, i] - np.mean(X[:, i])) / np.std(X[:, i])
                    xj_std = (X[:, j] - np.mean(X[:, j])) / np.std(X[:, j])
                    ri_j = xi_std if i in Vj and j in Uc else self._residual(xi_std, xj_std)
                    rj_i = xj_std if j in Vj and i in Uc else self._residual(xj_std, xi_std)
                    M += np.min([0, self._diff_mutual_info(xi_std, xj_std, ri_j, rj_i)])**2
            M_list.append(-1.0 * M)
        return Uc[np.argmax(M_list)]

    def _mutual_information(self, x1, x2, param):
        """Calculate the mutual informations."""
        kappa, sigma = param
        n = len(x1)
        X1 = np.tile(x1, (n, 1))
        K1 = np.exp(-1/(2*sigma**2) * (X1**2 + X1.T**2 - 2*X1*X1.T))
        X2 = np.tile(x2, (n, 1))
        K2 = np.exp(-1/(2*sigma**2) * (X2**2 + X2.T**2 - 2*X2*X2.T))

        tmp1 = K1 + n*kappa*np.identity(n)/2
        tmp2 = K2 + n*kappa*np.identity(n)/2
        K_kappa = np.r_[np.c_[tmp1 @ tmp1, K1 @ K2],
                        np.c_[K2 @ K1, tmp2 @ tmp2]]
        D_kappa = np.r_[np.c_[tmp1 @ tmp1, np.zeros([n, n])],
                        np.c_[np.zeros([n, n]), tmp2 @ tmp2]]

        sigma_K = np.linalg.svd(K_kappa, compute_uv=False)
        sigma_D = np.linalg.svd(D_kappa, compute_uv=False)

        return (-1/2)*(np.sum(np.log(sigma_K)) - np.sum(np.log(sigma_D)))

    def _search_causal_order_kernel(self, X, U):
        """Search the causal ordering by kernel method."""
        Uc, Vj = self._search_candidate(U)
        if len(Uc) == 1:
            return Uc[0]

        if X.shape[0] > 1000:
            param = [2e-3, 0.5]
        else:
            param = [2e-2, 1.0]

        Tkernels = []
        for j in Uc:
            Tkernel = 0
            for i in U:
                if i != j:
                    ri_j = X[:, i] if j in Vj and i in Uc else self._residual(
                        X[:, i], X[:, j])
                    Tkernel += self._mutual_information(X[:, j], ri_j, param)
            Tkernels.append(Tkernel)

        return Uc[np.argmin(Tkernels)]
