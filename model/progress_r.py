# =============================================================================
#  半监督核模型：LapRLS / LapSVM（原仓库实现，去除模块级 argparse/wandb 副作用）
#  供 REINFORCE 引擎与基线对比使用；S2MAM 主算法见 model/additive.py + model/bilevel.py
# =============================================================================

import numpy as np
import torch
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import kneighbors_graph
from scipy import sparse


def rbf(X1, X2, **kwargs):
    return np.exp(-cdist(X1, X2) ** 2 * kwargs['gamma'])


class LapRLS(object):
    """Laplacian Regularized Least Square（闭式解）。"""

    def __init__(self, n_neighbors, bandwidth, lambda_k, lambda_u,
                 learning_rate=1e-5, n_iterations=500, solver='closed-form'):
        self.n_neighbors = n_neighbors
        self.bandwidth = bandwidth
        self.lambda_k = lambda_k
        self.lambda_u = lambda_u
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.solver = solver

    def fit(self, X, Y, X_no_label):
        l = X.shape[0]
        u = X_no_label.shape[0]
        n = l + u

        self.X = np.concatenate([X, X_no_label], axis=0)
        try:
            self.Y = np.concatenate([Y, np.zeros(u).reshape(-1, 1)])
        except Exception:
            self.Y = np.concatenate([Y, np.zeros(u)])

        W = kneighbors_graph(self.X, self.n_neighbors, mode='connectivity')
        W = (((W + W.T) > 0) * 1)
        W = np.asarray(W.todense(), dtype=float)

        L = np.diag(np.asarray(W.sum(axis=1)).ravel()) - W

        K = rbf_kernel(self.X, gamma=self.bandwidth)

        J = np.diag(np.concatenate([np.ones(l), np.zeros(u)]))

        if self.solver == 'closed-form':
            final = (J.dot(K) + self.lambda_k * l * np.identity(l + u)
                     + ((self.lambda_u * l) / (l + u) ** 2) * L.dot(K))
            self.alpha = np.linalg.inv(final).dot(self.Y)
            del self.Y, W, L, K, J
        else:
            raise ValueError('仅保留 closed-form 求解器（原 gradient-descent/'
                             'L-BFGS-B 分支见 git 历史）')

        new_K = rbf_kernel(self.X, X, gamma=self.bandwidth)
        f = np.squeeze(np.array(self.alpha)).dot(new_K)
        return f

    def predict(self, Xtest):
        new_K = rbf_kernel(self.X, Xtest, gamma=self.bandwidth)
        f = np.squeeze(np.array(self.alpha)).dot(new_K)
        return f

    def accuracy(self, Xtest, Ytrue):
        predictions = self.predict(Xtest)
        mse = mean_squared_error(predictions, Ytrue)
        print('MSE: {}'.format(mse))


def rbf_kernel(X1, X2, gamma=1.0):
    return np.exp(-cdist(X1, X2) ** 2 * gamma)


class LapSVM(object):
    """Laplacian SVM（半监督 QP，原仓库实现）。"""

    def __init__(self, opt):
        self.opt = opt
        self.Q = 0

    def fit(self, X, Y, X_u):
        classes, y_indices = np.unique(Y, return_inverse=True)
        self.class_dict = {classes[0]: -1, classes[1]: 1}
        self.rev_class_dict = {-1: classes[0], 1: classes[1]}
        self.X = np.vstack([X, X_u])
        Y = np.diag(Y)
        if self.opt['neighbor_mode'] == 'connectivity':
            W = kneighbors_graph(self.X, self.opt['n_neighbor'],
                                 mode='connectivity', include_self=False)
            W = (((W + W.T) > 0) * 1)
        elif self.opt['neighbor_mode'] == 'distance':
            W = kneighbors_graph(self.X, self.opt['n_neighbor'],
                                 mode='distance', include_self=False)
            W = W.maximum(W.T)
            W = sparse.csr_matrix((np.exp(-W.data ** 2 / 4 / self.opt['t']),
                                   W.indices, W.indptr),
                                  shape=(self.X.shape[0], self.X.shape[0]))
        else:
            raise Exception()

        L = sparse.diags(np.array(W.sum(0))[0]).tocsr() - W

        K = self.opt['kernel_function'](self.X, self.X, **self.opt['kernel_parameters'])

        l = X.shape[0]
        u = X_u.shape[0]

        J = np.concatenate([np.identity(l), np.zeros(l * u).reshape(l, u)], axis=1)

        almost_alpha = np.linalg.inv(
            2 * self.opt['gamma_A'] * np.identity(l + u)
            + ((2 * self.opt['gamma_I']) / (l + u) ** 2) * L.dot(K)
        ).dot(J.T).dot(Y)

        self.Q = Y.dot(J).dot(K).dot(almost_alpha)
        self.Q = (self.Q + self.Q.T) / 2

        del W, L, K, J

        e = np.ones(l)
        q = -e

        def objective_func(beta):
            return (1 / 2) * beta.dot(self.Q).dot(beta) + q.dot(beta)

        def objective_grad(beta):
            return np.squeeze(np.array(beta.T.dot(self.Q) + q))

        bounds = [(0, 1 / l) for _ in range(l)]

        def constraint_func(beta):
            return beta.dot(np.diag(Y))

        def constraint_grad(beta):
            return np.diag(Y)

        cons = {'type': 'eq', 'fun': constraint_func, 'jac': constraint_grad}

        x0 = np.zeros(l)

        beta_hat = minimize(objective_func, x0, jac=objective_grad,
                            constraints=cons, bounds=bounds, tol=0.0001)['x']

        self.alpha = almost_alpha.dot(beta_hat)

        del almost_alpha, self.Q

        new_K = self.opt['kernel_function'](self.X, X, **self.opt['kernel_parameters'])
        f = np.squeeze(np.array(self.alpha)).dot(new_K)

        self.sv_ind = np.nonzero((beta_hat > 1e-7) * (beta_hat < (1 / l - 1e-7)))[0]
        try:
            ind = self.sv_ind[0]
        except Exception:
            ind = 0
        self.b = np.diag(Y)[ind] - f[ind]
        return f

    def decision_function(self, X):
        new_K = self.opt['kernel_function'](self.X, X, **self.opt['kernel_parameters'])
        f = np.squeeze(np.array(self.alpha)).dot(new_K)
        return f + self.b

    def predict_proba(self, X):
        y_desision = self.decision_function(X)
        y_score = np.full((X.shape[0], 2), 0, np.float64)
        y_score[:, 0] = 1 / (1 + np.exp(y_desision))
        y_score[:, 1] = 1 - y_score[:, 0]
        return torch.tensor(y_score).cpu()

    def predict(self, X):
        y_desision = self.decision_function(X)
        y_pred = np.ones(X.shape[0])
        y_pred[y_desision < 0] = -1
        return torch.tensor(y_pred).cpu()

    def accuracy(self, X, Y):
        predictions = self.predict(X)
        accuracy = sum(predictions == Y) / len(predictions)
        print('Accuracy: {}%'.format(round(accuracy * 100, 2)))
