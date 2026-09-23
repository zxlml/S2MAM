# -*- coding: utf-8 -*-
"""上层一阶投影梯度（投影算子 + 双引擎）的单元测试。"""

import numpy as np
import torch

from data.data_generation import s2mam_regression_data
from model.additive import S2MAMLower
from model.bilevel import (BilevelConfig, solve_bilevel, project_cardinality,
                           project_cardinality_np, cardinality_schedule,
                           cosine_lr)


# ------------------------------------------------------------------ 投影算子
def test_projection_feasibility():
    rng = np.random.RandomState(0)
    for _ in range(20):
        x = rng.randn(50)
        z = project_cardinality_np(x, k=10.0)
        assert z.min() >= 0 and z.max() <= 1
        assert abs(z.sum() - 10.0) < 1e-6


def test_projection_is_euclidean_projection():
    """投影应比任意其它可行点更接近原点（概率化最优性检查）。"""
    rng = np.random.RandomState(1)
    for _ in range(10):
        x = rng.randn(30) * 3
        z = project_cardinality_np(x, k=8.0)
        # 随机可行扰动点
        for _ in range(50):
            w = z + rng.randn(30) * 0.3
            w = project_cardinality_np(w, 8.0)
            d_proj = np.sum((x - z) ** 2)
            d_other = np.sum((x - w) ** 2)
            assert d_proj <= d_other + 1e-8


def test_projection_identity_when_feasible():
    rng = np.random.RandomState(2)
    x = rng.uniform(0, 1, size=20)
    x = project_cardinality_np(x, x.sum())
    y = project_cardinality_np(x, x.sum())
    assert np.allclose(x, y)


# ------------------------------------------------------------------ 调度函数
def test_cardinality_schedule():
    assert cardinality_schedule(0, 100, 8, 20, 0.16, 0.6) == 20
    assert cardinality_schedule(10, 100, 8, 20, 0.16, 0.6) == 20
    assert cardinality_schedule(99, 100, 8, 20, 0.16, 0.6) == 8
    assert cardinality_schedule(10, 100, 8, 8, 0.16, 0.6) == 8


def test_cosine_lr_decays():
    lrs = [cosine_lr(t, 100, 1.0) for t in range(100)]
    assert lrs[0] == 1.0
    assert abs(cosine_lr(100, 100, 1.0)) < 1e-9   # t=T 时精确为 0
    assert abs(lrs[-1]) < 5e-3
    assert all(lrs[i] >= lrs[i + 1] - 1e-12 for i in range(99))


# ------------------------------------------------------------------ 双引擎
def _make_small_problem():
    d = s2mam_regression_data(dataset='additive', N=120, Ntest=100,
                              n_labeled=40, n_uninformative=8, n_noisy=2,
                              noise_std=0.5, seed=0)
    lower = S2MAMLower(task='regression', r=4, lam_k=1e-3, lam_u=1e-2)
    lower.setup(d['Xl'], d['Yl'], d['Xu'], d['Xt'], d['Yt'])
    return lower, d


def test_implicit_engine_converges_and_selects():
    lower, d = _make_small_problem()
    p, k = lower.p_, 8
    cfg = BilevelConfig(k=k, max_outer_iter=60, outer_lr=2e-1, engine='implicit',
                        optimizer='sgd', clip_grad=1.0, seed=0, verbose=False)
    res = solve_bilevel(upper_fn=lambda m: lower.upper_loss(m),
                        m0=np.full(p, k / p), cfg=cfg)
    hist = res['history']
    # 目标下降
    assert hist[-1]['loss'] < hist[0]['loss']
    # 可行性
    assert abs(res['m_soft'].sum() - k) < 1e-6
    assert res['m_soft'].min() >= 0 and res['m_soft'].max() <= 1
    # 特征选择：真实 8 个信息维度应大部分被选中
    sel = np.nonzero(res['mask'])[0]
    recall = len(set(sel) & set(d['informative_idx'])) / k
    assert recall >= 0.75, f'recall={recall}, selected={sel}'


def test_implicit_engine_lrschedule_and_clip():
    lower, _ = _make_small_problem()
    cfg = BilevelConfig(k=4, max_outer_iter=30, outer_lr=5e-2, engine='implicit',
                        optimizer='sgd', clip_grad=1e-3, seed=0, verbose=False)
    res = solve_bilevel(upper_fn=lambda m: lower.upper_loss(m),
                        m0=np.full(lower.p_, 4 / lower.p_), cfg=cfg)
    # 极小裁剪下梯度范数都被限制，迭代仍有限且可行
    assert np.isfinite(res['m_soft']).all()


def test_reinforce_engine_runs_pbls_style():
    lower, d = _make_small_problem()
    p, k = lower.p_, 8
    cfg = BilevelConfig(k=k, max_outer_iter=15, outer_lr=5e-2, engine='reinforce',
                        K=4, vr=True, clip_grad=5.0, seed=0, verbose=False,
                        final_select='sample')
    res = solve_bilevel(m0=np.full(p, k / p), cfg=cfg,
                        upper_fn_value=lambda z: float(lower.upper_loss(z)))
    assert np.isfinite(res['m_soft']).all()
    assert res['mask'].sum() > 0
    assert res['best_mask'] is not None
    assert len(res['history']) == 15


def test_iterative_schedule_runs():
    lower, _ = _make_small_problem()
    p = lower.p_
    cfg = BilevelConfig(k=8, max_outer_iter=20, outer_lr=2e-1, engine='implicit',
                        iterative=True, start_k=p, ts=0.16, te=0.6,
                        seed=0, verbose=False)
    res = solve_bilevel(upper_fn=lambda m: lower.upper_loss(m),
                        m0=np.full(p, 0.5), cfg=cfg)
    ks = [h['k'] for h in res['history']]
    assert ks[0] == p and ks[-1] == 8
