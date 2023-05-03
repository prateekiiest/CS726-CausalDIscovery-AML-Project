import numpy as np
import pandas as pd
import math
from typing import Any, Dict, List, Tuple
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import cholesky, cho_solve
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.preprocessing import PolynomialFeatures
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

from ..utils.validation import Validation
from ..utils.ScoreUtils import *

class GPRMine(object):
    def __init__(self, optimize=False):
        self.is_fit = False
        self.train_X, self.train_y = None, None
        self.params = {"l": 1, "sigma_f": 1}
        self.optimize = optimize
        self.alpha = 1e-10
        self.m = None

    def fit(self, y, median, p_eu):
        self.train_y = np.asarray(y)
        K = self.kernel(median, p_eu)
        np.fill_diagonal(K, 1)
        self.K_trans = K.copy()
        K[np.diag_indices_from(K)] += self.alpha
        # self.KK = K.copy()

        self.L_ = cholesky(K, lower=True)  # Line 2
        # self.L_ changed, self._K_inv needs to be recomputed
        self._K_inv = None
        self.alpha_ = cho_solve((self.L_, True), self.train_y)  # Line 3
        self.is_fit = True

    def predict(self, return_std=False):
        if not self.is_fit:
            print("GPR Model not fit yet.")
            return

        K_trans = self.K_trans
        y_mean = K_trans.dot(self.alpha_)  # Line 4 (y_mean = f_star)
        if return_std == False:
            return y_mean
        else:
            raise ('To cal std')

    def kernel(self, median, p_eu):
        p_eu_nor = p_eu / median
        K = np.exp(-0.5 * p_eu_nor)
        K = squareform(K)
        return K


class Reward(object):
    """
    Used for calculate reward for ordering-based Causal discovery

    In ordering-based methods, only the variables selected in previous decision
    steps can be the potential parents of the currently selected variable.
    Hence, author design the rewards in the following cases:
    `episodic reward` and `dense reward`.
    
    """

    def __init__(self, input_data, reward_mode='episodic',
                 score_type='BIC', regression_type='LR', alpha=1.0):


        self.input_data = input_data
        self.reward_type = reward_mode
        self.alpha = alpha
        self.n_samples = input_data.shape[0]
        self.seq_length = input_data.shape[1]
        self.d = {}  # store results
        self.d_RSS = [{} for _ in range(self.seq_length)]  # store RSS for reuse
        self.bic_penalty = np.log(input_data.shape[0]) / input_data.shape[0]

        Validation.validate_value(score_type,
                                  ('BIC', 'BIC_different_var'))
        Validation.validate_value(regression_type,
                                  ('LR', 'QR', 'GPR', 'GPR_learnable'))

        self.score_type = score_type
        self.regression_type = regression_type

        self.poly = PolynomialFeatures()

        if self.regression_type == 'GPR_learnable':
            self.kernel_learnable = 1.0 * RBF(length_scale=1.0,
                                              length_scale_bounds=(1e-2, 1e2)) \
                                    + WhiteKernel(noise_level=1.0,
                                                  noise_level_bounds=(
                                                  1e-10, 1e+1))
        elif regression_type == 'LR':
            self.ones = np.ones((input_data.shape[0], 1), dtype=np.float32)
            X = np.hstack((self.input_data, self.ones))
            self.X = X
            self.XtX = X.T.dot(X)
        elif regression_type == 'GPR':
            self.gpr = GPRMine()
            m = input_data.shape[0]
            self.gpr.m = m
            dist_matrix = []
            for i in range(m):
                for j in range(i + 1, m):
                    dist_matrix.append((input_data[i] - input_data[j]) ** 2)
            self.dist_matrix = np.array(dist_matrix)

    def cal_rewards(self, Data, graphs, positions=None, ture_flag=False, gamma=0.98):
        # print('Shape')
        # print(positions.shape)
        rewards_batches = []
        if not ture_flag:
            for graphi, position in zip(graphs, positions):
                # print("position: ", position)
                reward_ = self.calculate_general_reward_single_graph(Data, graphi,position=position,ture_flag=ture_flag)
                # reward_ = self.calculate_reward_single_graph(graphi,position=position,ture_flag=ture_flag)
                rewards_batches.append(reward_)
        else:
            for graphi in graphs:
                reward_ = self.calculate_general_reward_single_graph(Data, graphi, ture_flag=ture_flag)
                # reward_ = self.calculate_reward_single_graph(graphi, ture_flag=ture_flag)
                rewards_batches.append(reward_)

        max_reward_batch = -float('inf')
        reward_list, normal_batch_reward = [], []
        for nu, (reward_, reward_list_) in enumerate(rewards_batches):
            reward_list.append(reward_list_)
            normalized_reward = -reward_
            normal_batch_reward.append(normalized_reward)
            if normalized_reward > max_reward_batch:
                max_reward_batch = normalized_reward
        normal_batch_reward = np.stack(normal_batch_reward)
        reward_list = - np.stack(reward_list)

        if self.reward_type == 'episodic':
            G = 0
            td_target = []
            for r in np.transpose(reward_list, [1, 0])[::-1]:
                G = r + gamma * G
                td_target.append(G)
        elif self.reward_type == 'dense':
            td_target = None
        else:
            raise ValueError(f"reward_type must be one of ['episodic', "
                             f"'dense'], but got ``{self.reward_type}``.")

        return reward_list, normal_batch_reward, max_reward_batch, td_target

    def calculate_yerr(self, X_train, y_train, XtX=None, Xty=None):
        if self.regression_type == 'LR':
            return self.calculate_LR(X_train, y_train, XtX, Xty)
        elif self.regression_type == 'QR':
            return self.calculate_QR(X_train, y_train)
        elif self.regression_type == 'GPR':
            return self.calculate_GPR(y_train, XtX)
        elif self.regression_type == 'GPR_learnable':
            return self.calculate_GPR_learnable(X_train, y_train)
        else:
            raise TypeError(f"The parameter `regression_type` must be one of "
                                f"[`LR`, `QR`, `GPR`, `GPR_learnable`], "
                                f"but got ``{self.regression_type}``.")

    def calculate_LR(self, X_train, y_train, XtX, Xty):
        """Linear regression"""

        theta = np.linalg.solve(XtX, Xty)
        y_pre = X_train.dot(theta)
        y_err = y_pre - y_train
        return y_err

    def calculate_QR(self, X_train, y_train):
        """quadratic regression"""

        X_train = self.poly.fit_transform(X_train)[:, 1:]
        X = np.hstack((X_train, self.ones))
        XtX = X.T.dot(X)
        Xty = X.T.dot(y_train)
        return self.calculate_LR(X_train, y_train, XtX, Xty)

    def calculate_GPR(self, y_train, XtX):
        p_eu = XtX  # our K1 don't sqrt
        med_w = np.median(p_eu)
        self.gpr.fit(y_train, med_w, p_eu)
        pre = self.gpr.predict()
        return y_train - pre

    def calculate_GPR_learnable(self, X_train, y_train):
        gpr = GPR(kernel=self.kernel_learnable, alpha=0.0).fit(X_train, y_train)
        return y_train.reshape(-1, 1) - gpr.predict(X_train).reshape(-1, 1)

    def calculate_reward_single_graph(self, graph_batch, position=None,ture_flag=False):
        graph_to_int2 = list(np.int32(position))
        graph_batch_to_tuple = tuple(graph_to_int2)
        if not ture_flag:
            if graph_batch_to_tuple in self.d:
                graph_score = self.d[graph_batch_to_tuple]
                reward = graph_score[0]
                return reward, np.array(graph_score[1])

        RSS_ls = []
        for i in range(self.seq_length):
            RSSi = self.cal_RSSi(i, graph_batch)
            RSS_ls.append(RSSi)

        RSS_ls = np.array(RSS_ls)
        if self.regression_type == 'GPR' or self.regression_type == 'GPR_learnable':
            reward_list = RSS_ls[position] / self.n_samples
        else:
            reward_list = RSS_ls[position] / self.n_samples

        if self.score_type == 'BIC':
            BIC = np.log(np.sum(RSS_ls) / self.n_samples + 1e-8)
            # + np.sum(graph_batch)*self.bic_penalty/self.seq_length
        elif self.score_type == 'BIC_different_var':
            BIC = np.sum(np.log(np.array(RSS_ls) / self.n_samples + 1e-8))
            # + np.sum(graph_batch)*self.bic_penalt
        else:
            raise TypeError(f"The parameter `score_type` must be one of "
                            f"[`BIC`,`BIC_different_var`], "
                            f"but got ``{self.score_type}``.")
        if not ture_flag:
            self.d[graph_batch_to_tuple] = (BIC, reward_list)
        # print(BIC, reward_list)
        return BIC, np.array(reward_list)

    def cal_RSSi(self, i, graph_batch):
        col = graph_batch[i]
        str_col = str(col)
        if str_col in self.d_RSS[i]:
            RSSi = self.d_RSS[i][str_col]
            return RSSi
        if np.sum(col) < 0.1:
            y_err = self.input_data[:, i]
            y_err = y_err - np.mean(y_err)
        else:
            cols_TrueFalse = col > 0.5
            if self.regression_type == 'LR':
                cols_TrueFalse = np.append(cols_TrueFalse, True)
                X_train = self.X[:, cols_TrueFalse]
                y_train = self.X[:, i]
                XtX = self.XtX[:, cols_TrueFalse][cols_TrueFalse, :]
                Xty = self.XtX[:, i][cols_TrueFalse]
                y_err = self.calculate_yerr(X_train, y_train, XtX, Xty)
            elif self.regression_type == 'GPR':
                X_train = self.input_data[:, cols_TrueFalse]
                y_train = self.input_data[:, i]
                p_eu = pdist(X_train, 'sqeuclidean')
                train_y = np.asarray(y_train)
                p_eu_nor = p_eu / np.median(p_eu)
                K = np.exp(-0.5 * p_eu_nor)
                K = squareform(K)
                np.fill_diagonal(K, 1)
                K_trans = K.copy()
                K[np.diag_indices_from(K)] += self.alpha  # 1e-10
                L_ = cholesky(K, lower=True)  # Line 2
                alpha_ = cho_solve((L_, True), train_y)  # Line 3
                y_mean = K_trans.dot(alpha_)  # Line 4 (y_mean = f_star)
                y_err = y_train - y_mean
            elif self.regression_type == 'GPR_learnable':
                X_train = self.input_data[:, cols_TrueFalse]
                y_train = self.input_data[:, i]
                y_err = self.calculate_yerr(X_train, y_train, X_train, y_train)
            else:
                raise TypeError(f"The parameter `regression_type` must be one of "
                                f"[`LR`, `GPR`, `GPR_learnable`], "
                                f"but got ``{self.regression_type}``.")
        RSSi = np.sum(np.square(y_err))
        self.d_RSS[i][str_col] = RSSi

        return RSSi

    def penalized_score(self, score_cyc, lambda1=1, lambda2=1):
        score, cyc = score_cyc
        return score + lambda1 * float(cyc > 1e-5) + lambda2 * cyc

    def update_scores(self, score_cycs):
        ls = []
        for score_cyc in score_cycs:
            ls.append(score_cyc)
        return ls

    def update_all_scores(self):
        score_cycs = list(self.d.items())
        ls = []
        for graph_int, score_l in score_cycs:
            ls.append((graph_int, (score_l[0], score_l[-1])))
        return sorted(ls, key=lambda x: x[1][0])

    def get_parents_given_topological_DAG(self, DAG_adjacency_matrix):
        n = DAG_adjacency_matrix.shape[0]
        parents = [np.array([]) for _ in range(n)]

        for j in range(n):
            col = DAG_adjacency_matrix[:,j]
            Pa_j = np.nonzero(col)[0]
            parents[j] = Pa_j

        return parents

    def calculate_general_reward_single_graph(self, Data, graph_batch, position=None, ture_flag=False):
        graph_to_int2 = list(np.int32(position))
        graph_batch_to_tuple = tuple(graph_to_int2)
        if not ture_flag:
            if graph_batch_to_tuple in self.d:
                graph_score = self.d[graph_batch_to_tuple]
                reward = graph_score[0]
                return reward, np.array(graph_score[1])

        parents = self.get_parents_given_topological_DAG(graph_batch)
        
        score = 0
        for Xi in range(self.seq_length):
            PAi = parents[Xi]
            # score += local_score_cv_general(self.input_data, Xi, PAi)
            score += local_score_cv_general(Data, Xi, PAi)
        # print('score:', score)

        RSS_ls = []
        for i in range(self.seq_length):
            RSSi = self.cal_RSSi(i, graph_batch)
            RSS_ls.append(RSSi)

        RSS_ls = np.array(RSS_ls)
        if self.regression_type == 'GPR' or self.regression_type == 'GPR_learnable':
            reward_list = RSS_ls[position] / self.n_samples
        else:
            reward_list = RSS_ls[position] / self.n_samples

        if not ture_flag:
            self.d[graph_batch_to_tuple] = (score, reward_list)
        # print(BIC, reward_list)
        return score, np.array(reward_list)

def local_score_cv_general(Data, Xi: int, PAi: List[int], parameters= {'kfold': 10, 'lambda': 0.01}):
    Data = Data.cpu().detach().numpy()
    Data = Data[:, :, 0]
    Data = np.mat(Data)
    
    PAi = list(PAi)

    T = Data.shape[0]
    X = Data[:, Xi]
    # print(X.shape)
    
    var_lambda = parameters["lambda"]  # regularization parameter
    k = parameters["kfold"]  # k-fold cross validation
    n0 = math.floor(T / k)
    gamma = 0.01
    Thresh = 1e-5
    # print('step 1')
    if len(PAi):
        PA = Data[:, PAi]

        # set the kernel for X
        GX = np.sum(np.multiply(X, X), axis=1)
        Q = np.tile(GX, (1, T))
        R = np.tile(GX.T, (T, 1))
        dists = Q + R - 2 * X * X.T
        dists = dists - np.tril(dists)
        dists = np.reshape(dists, (T**2, 1))
        width = np.sqrt(
            0.5 * np.median(dists[np.where(dists > 0)], axis=1)[0, 0]
        )  # median value
        width = width * 2
        theta = 1 / (width**2)

        Kx, _ = kernel(X, X, (theta, 1))  # Gaussian kernel
        H0 = (np.mat(np.eye(T)) - np.mat(np.ones((T, T))) / T)  # for centering of the data in feature space
        Kx = H0 * Kx * H0  # kernel matrix for X

        # set the kernel for PA
        Kpa = np.mat(np.ones((T, T)))
        # print('step 2')
        for m in range(PA.shape[1]):
            G = np.sum((np.multiply(PA[:, m], PA[:, m])), axis=1)
            Q = np.tile(G, (1, T))
            R = np.tile(G.T, (T, 1))
            dists = Q + R - 2 * PA[:, m] * PA[:, m].T
            dists = dists - np.tril(dists)
            dists = np.reshape(dists, (T**2, 1))
            width = np.sqrt(
                0.5 * np.median(dists[np.where(dists > 0)], axis=1)[0, 0]
            )  # median value
            width = width * 2
            theta = 1 / (width**2)
            Kpa = np.multiply(Kpa, kernel(PA[:, m], PA[:, m], (theta, 1))[0])

        # print('step 3')
        H0 = (
            np.mat(np.eye(T)) - np.mat(np.ones((T, T))) / T
        )  # for centering of the data in feature space
        Kpa = H0 * Kpa * H0  # kernel matrix for PA

        CV = 0
        # print('step 4')
        for kk in range(k):
            if kk == 0:
                Kx_te = Kx[0:n0, 0:n0]
                Kx_tr = Kx[n0:T, n0:T]
                Kx_tr_te = Kx[n0:T, 0:n0]
                Kpa_te = Kpa[0:n0, 0:n0]
                Kpa_tr = Kpa[n0:T, n0:T]
                Kpa_tr_te = Kpa[n0:T, 0:n0]
                nv = n0  # sample size of validated data
            elif kk == k - 1:
                Kx_te = Kx[kk * n0 : T, kk * n0 : T]
                Kx_tr = Kx[0 : kk * n0, 0 : kk * n0]
                Kx_tr_te = Kx[0 : kk * n0, kk * n0 : T]
                Kpa_te = Kpa[kk * n0 : T, kk * n0 : T]
                Kpa_tr = Kpa[0 : kk * n0, 0 : kk * n0]
                Kpa_tr_te = Kpa[0 : kk * n0, kk * n0 : T]
                nv = T - kk * n0
            elif kk < k - 1 and kk > 0:
                Kx_te = Kx[kk * n0 : (kk + 1) * n0, kk * n0 : (kk + 1) * n0]
                Kx_tr = Kx[
                    np.ix_(
                        np.concatenate(
                            [np.arange(0, kk * n0), np.arange((kk + 1) * n0, T)]
                        ),
                        np.concatenate(
                            [np.arange(0, kk * n0), np.arange((kk + 1) * n0, T)]
                        ),
                    )
                ]
                Kx_tr_te = Kx[
                    np.ix_(
                        np.concatenate(
                            [np.arange(0, kk * n0), np.arange((kk + 1) * n0, T)]
                        ),
                        np.arange(kk * n0, (kk + 1) * n0),
                    )
                ]
                Kpa_te = Kpa[kk * n0 : (kk + 1) * n0, kk * n0 : (kk + 1) * n0]
                Kpa_tr = Kpa[
                    np.ix_(
                        np.concatenate(
                            [np.arange(0, kk * n0), np.arange((kk + 1) * n0, T)]
                        ),
                        np.concatenate(
                            [np.arange(0, kk * n0), np.arange((kk + 1) * n0, T)]
                        ),
                    )
                ]
                Kpa_tr_te = Kpa[
                    np.ix_(
                        np.concatenate(
                            [np.arange(0, kk * n0), np.arange((kk + 1) * n0, T)]
                        ),
                        np.arange(kk * n0, (kk + 1) * n0),
                    )
                ]
                nv = n0
            else:
                raise ValueError("Not cover all logic path")

            n1 = T - nv
            tmp1 = pdinv(Kpa_tr + n1 * var_lambda * np.mat(np.eye(n1)))
            tmp2 = tmp1 * Kx_tr * tmp1
            tmp3 = (
                tmp1
                * pdinv(np.mat(np.eye(n1)) + n1 * var_lambda**2 / gamma * tmp2)
                * tmp1
            )
            A = (
                Kx_te
                + Kpa_tr_te.T * tmp2 * Kpa_tr_te
                - 2 * Kx_tr_te.T * tmp1 * Kpa_tr_te
                - n1 * var_lambda**2 / gamma * Kx_tr_te.T * tmp3 * Kx_tr_te
                - n1
                * var_lambda**2
                / gamma
                * Kpa_tr_te.T
                * tmp1
                * Kx_tr
                * tmp3
                * Kx_tr
                * tmp1
                * Kpa_tr_te
                + 2
                * n1
                * var_lambda**2
                / gamma
                * Kx_tr_te.T
                * tmp3
                * Kx_tr
                * tmp1
                * Kpa_tr_te
            ) / gamma

            B = n1 * var_lambda**2 / gamma * tmp2 + np.mat(np.eye(n1))
            L = np.linalg.cholesky(B)
            C = np.sum(np.log(np.diag(L)))
            #  CV = CV + (nv*nv*log(2*pi) + nv*C + nv*mx*log(gamma) + trace(A))/2;
            CV = CV + (nv * nv * np.log(2 * np.pi) + nv * C + np.trace(A)) / 2

        CV = CV / k
    else:
        # set the kernel for X
        GX = np.sum(np.multiply(X, X), axis=1)
        Q = np.tile(GX, (1, T))
        R = np.tile(GX.T, (T, 1))
        # print(GX.shape, T)
        dists = Q + R - 2 * X * X.T
        dists = dists - np.tril(dists)
        dists = np.reshape(dists, (T**2, 1))
        width = np.sqrt(
            0.5 * np.median(dists[np.where(dists > 0)], axis=1)[0, 0]
        )  # median value
        width = width * 2
        theta = 1 / (width**2)

        Kx, _ = kernel(X, X, (theta, 1))  # Gaussian kernel
        H0 = np.mat(np.eye(T)) - np.mat(np.ones((T, T))) / (
            T
        )  # for centering of the data in feature space
        Kx = H0 * Kx * H0  # kernel matrix for X

        # eig_Kx, eix = eigdec((Kx + Kx.T) / 2, np.min([400, math.floor(T / 2)]), evals_only=False)  # /2
        # IIx = np.where(eig_Kx > np.max(eig_Kx) * Thresh)[0]
        # mx = len(IIx)

        CV = 0
        for kk in range(k):
            if kk == 0:
                Kx_te = Kx[kk * n0 : (kk + 1) * n0, kk * n0 : (kk + 1) * n0]
                Kx_tr = Kx[(kk + 1) * n0 : T, (kk + 1) * n0 : T]
                Kx_tr_te = Kx[(kk + 1) * n0 : T, kk * n0 : (kk + 1) * n0]
                nv = n0
            elif kk == k - 1:
                Kx_te = Kx[kk * n0 : T, kk * n0 : T]
                Kx_tr = Kx[0 : kk * n0, 0 : kk * n0]
                Kx_tr_te = Kx[0 : kk * n0, kk * n0 : T]
                nv = T - kk * n0
            elif 0 < kk < k - 1:
                Kx_te = Kx[kk * n0 : (kk + 1) * n0, kk * n0 : (kk + 1) * n0]
                Kx_tr = Kx[
                    np.ix_(
                        np.concatenate(
                            [np.arange(0, kk * n0), np.arange((kk + 1) * n0, T)]
                        ),
                        np.concatenate(
                            [np.arange(0, kk * n0), np.arange((kk + 1) * n0, T)]
                        ),
                    )
                ]
                Kx_tr_te = Kx[
                    np.ix_(
                        np.concatenate(
                            [np.arange(0, kk * n0), np.arange((kk + 1) * n0, T)]
                        ),
                        np.arange(kk * n0, (kk + 1) * n0),
                    )
                ]
                nv = n0
            else:
                raise ValueError("Not cover all logic path")

            n1 = T - nv
            A = (
                Kx_te
                - 1
                / (gamma * n1)
                * Kx_tr_te.T
                * pdinv(np.mat(np.eye(n1)) + 1 / (gamma * n1) * Kx_tr)
                * Kx_tr_te
            ) / gamma
            B = 1 / (gamma * n1) * Kx_tr + np.mat(np.eye(n1))
            L = np.linalg.cholesky(B)
            C = np.sum(np.log(np.diag(L)))

            # CV = CV + (nv*nv*log(2*pi) + nv*C + nv*mx*log(gamma) + trace(A))/2;
            CV = CV + (nv * nv * np.log(2 * np.pi) + nv * C + np.trace(A)) / 2

        CV = CV / k
    # print('step 5')
    score = CV  # negative cross-validated likelihood
    return score

def local_score_marginal_general(Data, Xi: int, PAi: List[int], parameters={'kfold': 10, 'lambda': 0.01}) -> float:
    """
    Calculate the local score by negative marginal likelihood
    based on a regression model in RKHS
    Parameters
    ----------
    Data: (sample, features)
    Xi: current index
    PAi: parent indexes
    parameters: None
    Returns
    -------
    score: local score
    """

    T = Data.shape[0]
    X = Data[:, Xi]
    dX = X.shape[1]

    # set the kernel for X
    GX = np.sum(np.multiply(X, X), axis=1)
    Q = np.tile(GX, (1, T))
    R = np.tile(GX.T, (T, 1))
    dists = Q + R - 2 * X * X.T
    dists = dists - np.tril(dists)
    dists = np.reshape(dists, (T**2, 1))
    width = np.sqrt(0.5 * np.median(dists[np.where(dists > 0)], axis=1)[0, 0])
    width = width * 2.5  # kernel width
    theta = 1 / (width**2)
    H = np.mat(np.eye(T)) - np.mat(np.ones((T, T))) / T
    Kx, _ = kernel(X, X, (theta, 1))
    Kx = H * Kx * H

    Thresh = 1e-5
    eig_Kx, eix = eigdec(
        (Kx + Kx.T) / 2, np.min([400, math.floor(T / 4)]), evals_only=False
    )  # /2
    IIx = np.where(eig_Kx > np.max(eig_Kx) * Thresh)[0]
    eig_Kx = eig_Kx[IIx]
    eix = eix[:, IIx]

    if len(PAi):
        PA = Data[:, PAi]

        widthPA = np.mat(np.empty((PA.shape[1], 1)))
        # set the kernel for PA
        for m in range(PA.shape[1]):
            G = np.sum((np.multiply(PA[:, m], PA[:, m])), axis=1)
            Q = np.tile(G, (1, T))
            R = np.tile(G.T, (T, 1))
            dists = Q + R - 2 * PA[:, m] * PA[:, m].T
            dists = dists - np.tril(dists)
            dists = np.reshape(dists, (T**2, 1))
            widthPA[m] = np.sqrt(
                0.5 * np.median(dists[np.where(dists > 0)], axis=1)[0, 0]
            )
        widthPA = widthPA * 2.5  # kernel width

        covfunc = np.asarray(["covSum", ["covSEard", "covNoise"]])
        logtheta0 = np.vstack([np.log(widthPA), 0, np.log(np.sqrt(0.1))])
        logtheta, fvals, iter = minimize(
            logtheta0,
            "gpr_multi_new",
            -300,
            covfunc,
            PA,
            2 * np.sqrt(T) * eix * np.diag(np.sqrt(eig_Kx)) / np.sqrt(eig_Kx[0]),
        )

        nlml, dnlml = gpr_multi_new(
            logtheta,
            covfunc,
            PA,
            2 * np.sqrt(T) * eix * np.diag(np.sqrt(eig_Kx)) / np.sqrt(eig_Kx[0]),
            nargout=2,
        )
    else:
        covfunc = np.asarray(["covSum", ["covSEard", "covNoise"]])
        PA = np.mat(np.zeros((T, 1)))
        logtheta0 = np.mat([100, 0, np.log(np.sqrt(0.1))]).T
        logtheta, fvals, iter = minimize(
            logtheta0,
            "gpr_multi_new",
            -300,
            covfunc,
            PA,
            2 * np.sqrt(T) * eix * np.diag(np.sqrt(eig_Kx)) / np.sqrt(eig_Kx[0]),
        )

        nlml, dnlml = gpr_multi_new(
            logtheta,
            covfunc,
            PA,
            2 * np.sqrt(T) * eix * np.diag(np.sqrt(eig_Kx)) / np.sqrt(eig_Kx[0]),
            nargout=2,
        )
    score = nlml  # negative log-likelihood
    return score

def local_score_BDeu(Data, i: int, PAi: List[int], parameters=None) -> float:

    """
    Calculate the *negative* local score with BDeu for the discrete case
    Parameters
    ----------
    Data: (sample, features)
    i: current index
    PAi: parent indexes
    parameters:
                 sample_prior: sample prior
                 structure_prior: structure prior
                 r_i_map: number of states of the finite random variable X_{i}
    Returns
    -------
    score: local BDeu score
    """
    if parameters is None:
        sample_prior = 1  # default sample_prior = 1
        structure_prior = 1  # default structure_prior = 1
        r_i_map = {
            i: len(np.unique(np.asarray(Data[:, i]))) for i in range(Data.shape[1])
        }
    else:
        sample_prior = parameters["sample_prior"]
        structure_prior = parameters["structure_prior"]
        r_i_map = parameters["r_i_map"]

    # calculate q_{i}
    q_i = 1
    for pa in PAi:
        q_i *= r_i_map[pa]

    if len(PAi) != 0:
        # calculate N_{ij}
        names = ["x{}".format(i) for i in range(Data.shape[1])]
        Data_pd = pd.DataFrame(Data, columns=names)
        parant_names = ["x{}".format(i) for i in PAi]
        Data_pd_group_Nij = Data_pd.groupby(parant_names)
        Nij_map = {
            key: len(Data_pd_group_Nij.indices.get(key))
            for key in Data_pd_group_Nij.indices.keys()
        }
        Nij_map_keys_list = list(Nij_map.keys())

        # calculate N_{ijk}
        Nijk_map = {
            ij: Data_pd_group_Nij.get_group(ij)
            .groupby("x{}".format(i))
            .apply(len)
            .reset_index()
            for ij in Nij_map.keys()
        }
        for v in Nijk_map.values():
            v.columns = ["x{}".format(i), "times"]
    else:
        # calculate N_{ij}
        names = ["x{}".format(i) for i in range(Data.shape[1])]
        Nij_map = {"": len(Data[:, i])}
        Nij_map_keys_list = [""]
        Data_pd = pd.DataFrame(Data, columns=names)

        # calculate N_{ijk}
        Nijk_map = {
            ij: Data_pd.groupby("x{}".format(i)).apply(len).reset_index()
            for ij in Nij_map_keys_list
        }
        for v in Nijk_map.values():
            v.columns = ["x{}".format(i), "times"]

    BDeu_score = 0
    # first term
    vm = Data.shape[0] - 1
    BDeu_score += len(PAi) * np.log(structure_prior / vm) + (vm - len(PAi)) * np.log(
        1 - (structure_prior / vm)
    )

    # second term
    for pa in range(len(Nij_map_keys_list)):
        Nij = Nij_map.get(Nij_map_keys_list[pa])
        first_term = math.lgamma(sample_prior / q_i) - math.lgamma(
            Nij + sample_prior / q_i
        )

        second_term = 0
        Nijk_list = Nijk_map.get(Nij_map_keys_list[pa])["times"].to_numpy()
        for Nijk in Nijk_list:
            second_term += math.lgamma(
                Nijk + sample_prior / (r_i_map[i] * q_i)
            ) - math.lgamma(sample_prior / (r_i_map[i] * q_i))

        BDeu_score += first_term + second_term

    return -BDeu_score

def local_score_BIC(Data, i: int, PAi: List[int], parameters=None) -> float:
    """
    Calculate the *negative* local score with BIC for the linear Gaussian continue data case
    Parameters
    ----------
    Data: ndarray, (sample, features)
    i: current index
    PAi: parent indexes
    parameters: lambda_value, the penalty discount of bic
    Returns
    -------
    score: local BIC score
    """

    cov = np.cov(Data.T)
    n = Data.shape[0]
    # cov, n = Data

    if parameters is None:
        lambda_value = 1
    else:
        lambda_value = parameters["lambda_value"]

    if len(PAi) == 0:
        return n * np.log(cov[i, i])

    yX = np.mat(cov[np.ix_([i], PAi)])
    XX = np.mat(cov[np.ix_(PAi, PAi)])
    H = np.log(cov[i, i] - yX * np.linalg.inv(XX) * yX.T)

    return n * H + np.log(n) * len(PAi) * lambda_value