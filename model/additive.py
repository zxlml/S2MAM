# =============================================================================
#  S2MAM 下层问题：部分标注（semi-supervised）+ Laplacian 正则 + 可加模型
# =============================================================================
#  对每个原始特征 x_j 用 M 阶样条基展开 B_j(x_j) ∈ R^r，可加模型为
#      f_m(x) = sum_j  m_j * B_j(x_j)^T beta_j ,
#  其中 m ∈ [0,1]^p 是上层特征选择（软）掩码。
#
#  下层优化（对给定 m，关于 beta 的强凸二次问题）：
#      beta*(m) = argmin_beta  (1/l) * sum_{i in L} (y_i - f_m(x_i))^2
#                              + lambda_k * sum_j beta_j^T D^T D beta_j   (P-spline 差分光滑惩罚)
#                              + (lambda_u / n^2) * f_m(X)^T L f_m(X)     (全体样本图 Laplacian 正则)
#  其中 L 为全体（有标注 + 无标注）样本 kNN 图的 Laplacian，
#  l 为有标注样本数，n = l + u。
#
#  闭式解 beta*(m) = A(m)^{-1} b(m)：
#      A(m) = (1/l) Phi(m)^T J Phi(m) + lambda_k*S + (lambda_u/n^2) Phi(m)^T L Phi(m) + rho*I
#      b(m) = (1/l) Phi(m)^T J Y
#  其中 Phi(m) = [m_1*B_1, ..., m_p*B_p]，J 为有标注行的选择矩阵，S 为分块差分惩罚。
#
#  收敛性/可微性要点（与 PBCS 的本质差异）：
#  * S + rho*I ≻ 0（rho 为数值岭）⇒ A(m) ≻ 0，下层解唯一；
#  * A(m)、b(m) 关于 m 光滑 ⇒ 由隐函数定理 beta*(m) 光滑，
#    上层目标 F(m) = L_val(beta*(m), m) 光滑，可由 autograd 穿过
#    torch.linalg.solve 得到**精确**超梯度 dF/dm（隐式微分，无需内层 SGD、
#    无需 PBCS 的单步伪更新 + HVP 有偏估计）；
#  * 因此上层是在凸紧集 {m ∈ [0,1]^p, sum m = k} 上的光滑函数，
#    一阶投影梯度法（步长余弦衰减 + 梯度裁剪）收敛到稳定点。
# =============================================================================

import numpy as np
import torch
from sklearn.neighbors import kneighbors_graph
from scipy.spatial.distance import cdist

from data.data_generation import bsplinebasis


def rbf(X1, X2, **kwargs):
    """与 model/progress_r.py 中一致的 RBF 核（保留供 LapSVM 使用）。"""
    return np.exp(-cdist(X1, X2) ** 2 * kwargs['gamma'])


class SplineFeatureExpander:
    """逐特征 B 样条基展开。

    每个特征在训练数据（全体有标注+无标注样本）的 [min, max] 上建立节点，
    生成 r 个基函数；测试/验证集取值裁剪到训练范围内以保证基函数有定义。
    """

    def __init__(self, r: int = 4):
        self.r = r

    def fit(self, X: np.ndarray):
        X = np.asarray(X, dtype=float)
        self.mins_ = X.min(axis=0)
        self.maxs_ = X.max(axis=0)
        return self

    def transform(self, X: np.ndarray) -> torch.Tensor:
        """返回 torch.float64 设计矩阵 (n, p*r)，第 j 块为特征 j 的基。"""
        X = np.asarray(X, dtype=float)
        rng = self.maxs_ - self.mins_
        # 避免取值恰好落在右端点（B 样条在该处可能全为 0）
        Xc = np.clip(X, self.mins_, self.maxs_ - 1e-9 * np.where(rng > 0, rng, 1.0))
        blocks = []
        for j in range(X.shape[1]):
            if rng[j] <= 0:  # 常数特征：退化处理
                B = np.zeros((X.shape[0], self.r))
                B[:, 0] = 1.0
            else:
                B = bsplinebasis(Xc[:, j], [self.mins_[j], self.maxs_[j]], self.r)
            blocks.append(B)
        Phi = np.concatenate(blocks, axis=1)
        return torch.tensor(Phi, dtype=torch.float64)

    @property
    def block_slice(self):
        """返回 [(start, end), ...]：每个特征基块在设计矩阵中的列区间。"""
        return [(j * self.r, (j + 1) * self.r) for j in range(self.p_)]

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def difference_penalty(r: int, order: int = 2) -> np.ndarray:
    """P-spline 的 r 阶系数差分惩罚矩阵 D^T D（单个特征块）。"""
    D = np.eye(r)
    for _ in range(order):
        D = np.diff(D, axis=0)
    return D.T @ D


def graph_laplacian(X: np.ndarray, n_neighbors: int = 10) -> np.ndarray:
    """全体样本（有标注 + 无标注）kNN 图的对称 Laplacian L = D - W。"""
    W = kneighbors_graph(X, n_neighbors, mode='connectivity', include_self=False)
    W = (((W + W.T) > 0) * 1).astype(float)
    W = np.asarray(W.todense())
    deg = np.asarray(W.sum(axis=1)).ravel()
    D = np.diag(deg)
    return D - W


class S2MAMLower:
    """可微下层问题：部分标注 + Laplacian 正则 + 可加样条模型。

    用法：
        lower = S2MAMLower(task='regression')
        lower.setup(Xl, Yl, Xu, X_upper, Y_upper)   # 上层评估集默认为测试集
        loss = lower.upper_loss(m)                  # 可微标量（m 为 leaf 张量）
        beta = lower.solve_beta(m)                  # 闭式解（可微）
        metrics = lower.final_eval(m_binary)        # 离散掩码下的评估
    """

    def __init__(self, task: str = 'regression', r: int = 4, n_neighbors: int = 10,
                 lam_k: float = 1e-3, lam_u: float = 1e-2,
                 spline_diff_order: int = 2, ridge: float = 1e-8,
                 gate_delta: float = 1.0):
        assert task in ('regression', 'classification')
        self.task = task
        self.r = r
        self.n_neighbors = n_neighbors
        self.lam_k = lam_k
        self.lam_u = lam_u
        self.spline_diff_order = spline_diff_order
        self.ridge = ridge  # 数值岭，保证 A(m) ≻ 0（不改变问题的统计性质）
        self.gate_delta = gate_delta  # 门控惩罚系数 δ(1-m_j)^2‖β_j‖²

    # ------------------------------------------------------------------ setup
    def setup(self, Xl, Yl, Xu, X_upper, Y_upper, standardize=True):
        Xl = np.asarray(Xl, dtype=float)
        Xu = np.asarray(Xu, dtype=float)
        X_all = np.vstack([Xl, Xu])
        self.n_, self.p_ = X_all.shape
        self.l_ = Xl.shape[0]

        # 特征标准化（kNN 图与样条节点均基于全体训练样本）
        self.x_mean_ = X_all.mean(axis=0)
        self.x_std_ = X_all.std(axis=0)
        self.x_std_[self.x_std_ == 0] = 1.0
        if standardize:
            Xl_s = (Xl - self.x_mean_) / self.x_std_
            X_all_s = (X_all - self.x_mean_) / self.x_std_
            X_upper_s = (np.asarray(X_upper, dtype=float) - self.x_mean_) / self.x_std_
        else:
            Xl_s, X_all_s, X_upper_s = Xl, X_all, np.asarray(X_upper, dtype=float)

        # 样条基（常量，不依赖掩码）
        self.expander_ = SplineFeatureExpander(r=self.r).fit(X_all_s)
        self.B_all_ = self.expander_.transform(X_all_s)          # (n, p*r)
        self.B_upper_ = self.expander_.transform(X_upper_s)      # (n_up, p*r)

        # 图 Laplacian（常量，基于全体训练样本）
        L = graph_laplacian(X_all_s, self.n_neighbors)
        self.L_ = torch.tensor(L, dtype=torch.float64)
        self.LB_ = self.L_ @ self.B_all_                         # 预计算 L @ B

        # P-spline 分块差分惩罚
        S_block = difference_penalty(self.r, self.spline_diff_order)
        S = np.zeros((self.p_ * self.r, self.p_ * self.r))
        for j in range(self.p_):
            S[j * self.r:(j + 1) * self.r, j * self.r:(j + 1) * self.r] = S_block
        self.S_ = torch.tensor(S, dtype=torch.float64)

        # 标签（回归做标准化；分类映射到 ±1）
        Yl = np.asarray(Yl, dtype=float).reshape(-1)
        Y_upper = np.asarray(Y_upper, dtype=float).reshape(-1)
        if self.task == 'regression':
            self.y_mean_ = Yl.mean()
            self.y_std_ = Yl.std() if Yl.std() > 0 else 1.0
            self.Y_l_ = torch.tensor((Yl - self.y_mean_) / self.y_std_, dtype=torch.float64)
            self.Y_upper_ = torch.tensor((Y_upper - self.y_mean_) / self.y_std_,
                                         dtype=torch.float64)
            self.Y_upper_raw_ = Y_upper
        else:
            self.Y_l_ = torch.tensor(np.where(Yl > 0, 1.0, -1.0), dtype=torch.float64)
            self.Y_upper_ = torch.tensor(np.where(Y_upper > 0, 1.0, -1.0),
                                         dtype=torch.float64)
            self.Y_upper_raw_ = Y_upper
        return self

    # ------------------------------------------------------- 可微下层求解
    def _masked_blocks(self, B: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        """Phi = [m_1*B_1, ..., m_p*B_p]：把掩码沿特征块重复后逐列缩放。"""
        w = m.repeat_interleave(self.r)
        return B * w.unsqueeze(0)

    def solve_beta(self, m: torch.Tensor) -> torch.Tensor:
        """闭式解 beta*(m) = A(m)^{-1} b(m)，对 m 可微（float64）。

        A(m) 含门控惩罚 δ(1-m_j)^2：m_j→0 时强制 β_j→0，消除
        "衰减-重优化抵消"带来的平坦方向（否则软掩码与离散选择脱节，
        分层组-lasso 式的分层惩罚保证软解与 top-k 离散解一致）。
        """
        m = m.to(torch.float64)
        w = m.repeat_interleave(self.r)
        Phi = self._masked_blocks(self.B_all_, m)
        Phi_l = Phi[:self.l_]
        n = float(self.n_)
        A = (Phi_l.T @ Phi_l) / self.l_ \
            + self.lam_k * self.S_ \
            + (self.lam_u / n ** 2) * (Phi.T @ (self.LB_ * w.unsqueeze(0))) \
            + self.gate_delta * torch.diag((1.0 - w) ** 2)
        A = 0.5 * (A + A.T) + self.ridge * torch.eye(A.shape[0], dtype=torch.float64)
        b = Phi_l.T @ self.Y_l_ / self.l_
        return torch.linalg.solve(A, b)

    def predict_upper(self, m: torch.Tensor, masked: bool = True) -> torch.Tensor:
        """上层评估集上的预测 f（标准化尺度）。"""
        B_up = self._masked_blocks(self.B_upper_, m) if masked else self.B_upper_
        return B_up @ self.solve_beta(m)

    def upper_loss(self, m: torch.Tensor, masked: bool = True,
                   objective: str = 'loo') -> torch.Tensor:
        """上层目标 F(m)（可微）。

        objective='loo'（默认）：有标注集上的解析 LOO-CV 误差
            r_i^loo = (y_i - f_i) / (1 - H_ii)，H 为线性平滑器（帽子矩阵）。
            相比直接用测试损失：无信息泄漏、方差小，是半监督
            LapRLS/LapSVM 文献中选择超参数的标准做法。
        objective='test'：上层评估集损失（与原仓库协议一致）。
        任务损失：regression 为 MSE；classification 为 logistic（softplus(-y f)）。
        """
        if objective == 'loo':
            m = m.to(torch.float64)
            w = m.repeat_interleave(self.r)
            Phi = self._masked_blocks(self.B_all_, m)
            Phi_l = Phi[:self.l_]
            n = float(self.n_)
            A = (Phi_l.T @ Phi_l) / self.l_ \
                + self.lam_k * self.S_ \
                + (self.lam_u / n ** 2) * (Phi.T @ (self.LB_ * w.unsqueeze(0))) \
                + self.gate_delta * torch.diag((1.0 - w) ** 2)
            A = 0.5 * (A + A.T) + self.ridge * torch.eye(A.shape[0],
                                                        dtype=torch.float64)
            b = Phi_l.T @ self.Y_l_ / self.l_
            beta = torch.linalg.solve(A, b)
            f_l = Phi_l @ beta
            resid = f_l - self.Y_l_
            if self.task == 'regression':
                Ainv = torch.linalg.inv(A)
                h_diag = (Phi_l * (Phi_l @ Ainv)).sum(dim=1) / self.l_
                r_loo = resid / (1.0 - h_diag)
                return torch.mean(r_loo ** 2)
            # 分类的 logistic 损失用 LOO 近似：用训练预测的留一残差
            Ainv = torch.linalg.inv(A)
            h_diag = (Phi_l * (Phi_l @ Ainv)).sum(dim=1) / self.l_
            r_loo = resid / (1.0 - h_diag)
            return torch.nn.functional.softplus(-self.Y_l_ * r_loo).mean()

        f = self.predict_upper(m, masked=masked)
        if self.task == 'regression':
            return torch.mean((f - self.Y_upper_) ** 2)
        return torch.nn.functional.softplus(-self.Y_upper_ * f).mean()

    # ---------------------------------------------------------------- 评估
    @torch.no_grad()
    def final_eval(self, m_binary) -> dict:
        """离散掩码下的最终评估：闭式重解 + 上层集预测。

        返回 dict：mask（二值 0/1 numpy）、selected（特征索引）、
        pred_raw（原始尺度预测）、metric（任务对应指标）、
        metric_scaled（标准化尺度 MSE，仅回归）。
        """
        m = torch.as_tensor(np.asarray(m_binary, dtype=float), dtype=torch.float64)
        f = self.predict_upper(m, masked=True).numpy()
        if self.task == 'regression':
            pred_raw = f * self.y_std_ + self.y_mean_
            mse_scaled = float(np.mean((f - self.Y_upper_.numpy()) ** 2))
            mse_raw = float(np.mean((pred_raw - self.Y_upper_raw_) ** 2))
            metric = {'mse': mse_raw, 'rmse': float(np.sqrt(mse_raw)),
                      'mse_scaled': mse_scaled}
        else:
            pred_raw = np.where(f > 0, 1, -1)
            acc = float(np.mean(pred_raw == self.Y_upper_.numpy()))
            metric = {'acc': acc}
        mask = (np.asarray(m_binary, dtype=float) > 0.5).astype(int)
        return {'mask': mask,
                'selected': np.nonzero(mask)[0],
                'pred_raw': pred_raw,
                'metric': metric}
