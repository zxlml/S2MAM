# -*- coding: utf-8 -*-
"""下层模型（可加样条 + Laplacian 正则 + 部分标注）的单元测试。"""

import numpy as np
import torch
import pytest

from data.data_generation import s2mam_regression_data
from model.additive import (S2MAMLower, SplineFeatureExpander,
                            difference_penalty, graph_laplacian)


# ------------------------------------------------------------------ 样条基
def test_spline_basis_partition_of_unity():
    rng = np.random.RandomState(0)
    x = rng.uniform(-1, 1, size=200)
    B = SplineFeatureExpander(r=4).fit(x.reshape(-1, 1)).transform(x.reshape(-1, 1))
    B = B.numpy()
    assert B.shape == (200, 4)
    # B 样条基的单位分解性质：每行基函数值之和为 1
    assert np.allclose(B.sum(axis=1), 1.0, atol=1e-8)
    assert B.min() >= -1e-12  # 基函数非负


def test_difference_penalty():
    S = difference_penalty(5, order=2)
    assert S.shape == (5, 5)
    assert np.allclose(S, S.T)
    eigvals = np.linalg.eigvalsh(S)
    assert eigvals.min() >= -1e-12          # 半正定
    assert eigvals.max() > 1.0              # 有实际惩罚强度


def test_graph_laplacian():
    rng = np.random.RandomState(0)
    X = rng.randn(60, 5)
    L = graph_laplacian(X, n_neighbors=5)
    assert L.shape == (60, 60)
    assert np.allclose(L.sum(axis=1), 0, atol=1e-8)   # 行和为零
    assert np.allclose(L, L.T)                        # 对称
    eigvals = np.linalg.eigvalsh(L)
    assert eigvals.min() >= -1e-8                     # 半正定


# ------------------------------------------------- 下层闭式解 & 数值性质
def _make_lower():
    d = s2mam_regression_data(dataset='additive', N=120, Ntest=100,
                              n_labeled=40, n_uninformative=4, n_noisy=2,
                              noise_std=0.5, seed=0)
    lower = S2MAMLower(task='regression', r=4, lam_k=1e-3, lam_u=1e-2)
    lower.setup(d['Xl'], d['Yl'], d['Xu'], d['Xt'], d['Yt'])
    return lower, d


def test_lower_solve_beta_finite_and_pd():
    lower, _ = _make_lower()
    m = torch.full((lower.p_,), 0.5, dtype=torch.float64)
    beta = lower.solve_beta(m)
    assert torch.isfinite(beta).all()


def test_lower_matches_direct_ridge_when_lam_u_zero():
    """lam_u=0 时应退化为纯有标注样本上的（惩罚样条）岭回归闭式解（含门控项）。"""
    lower, _ = _make_lower()
    lower.lam_u = 0.0
    m = torch.full((lower.p_,), 0.8, dtype=torch.float64)
    beta = lower.solve_beta(m).numpy()

    Phi = lower._masked_blocks(lower.B_all_, m).numpy()
    Phi_l = Phi[:lower.l_]
    n = float(lower.n_)
    w_full = np.repeat(m.numpy(), lower.r)   # 门控按样条系数块展开
    A_ref = (Phi_l.T @ Phi_l) / lower.l_ + lower.lam_k * lower.S_.numpy() \
        + lower.gate_delta * np.diag((1.0 - w_full) ** 2) \
        + lower.ridge * np.eye(Phi.shape[1])
    b_ref = Phi_l.T @ lower.Y_l_.numpy() / lower.l_
    beta_ref = np.linalg.solve(A_ref, b_ref)
    assert np.allclose(beta, beta_ref, atol=1e-8)


def test_lower_fits_clean_additive_function():
    """无噪声、仅信息维度时闭式解应几乎精确重构可加函数。"""
    rng = np.random.RandomState(0)
    X = rng.uniform(-1, 1, size=(200, 3))
    # 2 特征可加真值
    Y = -2 * np.sin(2 * X[:, 0]) + 5 * X[:, 1]
    # 用自构造的干净数据
    lower = S2MAMLower(task='regression', r=6, lam_k=1e-6, lam_u=0.0)
    lower.setup(X, Y, X[:10], X, Y)  # 训练=测试（重构性检查）
    m = torch.ones(lower.p_, dtype=torch.float64)
    f = lower.predict_upper(m).numpy()
    Y_true_scaled = (Y - lower.y_mean_) / lower.y_std_
    rmse = np.sqrt(np.mean((f - Y_true_scaled) ** 2))
    assert rmse < 0.05, f'可加模型重构 RMSE 过大: {rmse}'


def test_upper_loss_value_sane():
    lower, d = _make_lower()
    m = torch.full((lower.p_,), 0.5, dtype=torch.float64)
    loss = lower.upper_loss(m)
    assert torch.isfinite(loss)
    assert loss.item() >= 0


# ------------------------------------------------------------------ 超梯度
def test_hypergradient_matches_finite_difference():
    """隐式超梯度（autograd 穿过闭式解）应与中心差分一致 —— 核心正确性测试。"""
    lower, _ = _make_lower()
    rng = np.random.RandomState(1)
    m0 = rng.uniform(0.1, 0.9, size=lower.p_)
    m = torch.tensor(m0, dtype=torch.float64, requires_grad=True)
    loss = lower.upper_loss(m)
    grad = torch.autograd.grad(loss, m)[0].numpy()

    h = 1e-5
    grad_fd = np.zeros_like(m0)
    for j in range(lower.p_):
        mp, mm = m0.copy(), m0.copy()
        mp[j] += h
        mm[j] -= h
        lp = lower.upper_loss(torch.tensor(mp, dtype=torch.float64)).item()
        lm = lower.upper_loss(torch.tensor(mm, dtype=torch.float64)).item()
        grad_fd[j] = (lp - lm) / (2 * h)
    assert np.allclose(grad, grad_fd, rtol=1e-4, atol=1e-6), \
        f'最大偏差 {np.abs(grad - grad_fd).max()}'


def test_final_eval_regression_metrics():
    lower, d = _make_lower()
    mask = np.zeros(lower.p_, dtype=int)
    mask[d['informative_idx']] = 1
    out = lower.final_eval(mask)
    assert out['mask'].sum() == 8
    assert np.isfinite(out['metric']['mse'])
    assert np.isfinite(out['metric']['mse_scaled'])
    assert out['metric']['rmse'] >= 0
