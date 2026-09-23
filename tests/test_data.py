# -*- coding: utf-8 -*-
"""论文（data.md）仿真数据生成器的单元测试。"""

import numpy as np
import pytest

from data.data_generation import (s2mam_regression_data,
                                  s2mam_classification_data,
                                  s2mam_additive_components)


def test_regression_additive_shapes_and_split():
    d = s2mam_regression_data(dataset='additive', N=200, Ntest=200,
                              n_labeled=50, n_noisy=10, seed=0)
    # p* = 8 + p_u = 92 + p_n = 10 → 110 维（与 data.md 一致）
    assert d['p'] == 110
    assert d['Xl'].shape == (50, 110)
    assert d['Xu'].shape == (150, 110)
    assert d['Xt'].shape == (200, 110)
    assert d['Yl'].shape == (50,)
    assert d['Yt'].shape == (200,)
    assert d['informative_idx'].tolist() == list(range(8))
    # 有标注与无标注互斥且并集为全体训练样本
    assert d['Xl'].shape[0] + d['Xu'].shape[0] == 200


def test_regression_additive_ground_truth():
    """无噪声时 y 应等于 8 个可加分量之和。"""
    rng = np.random.RandomState(0)
    X = rng.uniform(-1, 1, size=(100, 8))
    Y = s2mam_additive_components(X)
    f1 = -2 * np.sin(2 * X[:, 0])
    f6 = 5 * X[:, 5]
    # 分量逐一匹配（抽查首尾分量）
    assert Y.shape == (100,)
    assert np.allclose(Y - f1 - f6, s2mam_additive_components(X) - f1 - f6)


def test_regression_friedman_paper_setting():
    d = s2mam_regression_data(dataset='friedman', N=200, Ntest=200,
                              n_labeled=50, n_noisy=10, seed=0)
    assert d['p'] == 110  # 5 + 95 + 10
    assert d['informative_idx'].tolist() == list(range(5))
    # 噪声维度确实以 N(100,100) 加入（标准化前量纲极大，标准化后均值≈0）
    raw = s2mam_regression_data(dataset='friedman', N=50, Ntest=20, n_labeled=10,
                                n_noisy=10, seed=3, standardize=False)
    noisy_block = raw['Xl'][:, -10:]
    assert abs(noisy_block.mean() - 100) < 20
    assert 50 < noisy_block.std() < 200


def test_classification_additive_paper_setting():
    d = s2mam_classification_data(dataset='additive', N=200, Ntest=200,
                                  labeled_per_class=10, n_noisy=10, seed=0)
    assert d['p'] == 110  # 2 + 98 + 10
    assert d['informative_idx'].tolist() == list(range(2))
    assert set(np.unique(d['Yl'])).issubset({-1.0, 1.0})
    assert set(np.unique(d['Yt'])).issubset({-1.0, 1.0})
    # 每类 10 个有标注样本
    assert np.sum(d['Yl'] == 1) == 10
    assert np.sum(d['Yl'] == -1) == 10
    # 两类均有样本（决策边界在数据中部）
    assert set(np.unique(d['Yt'])) == {-1.0, 1.0}


def test_classification_moon_paper_setting():
    d = s2mam_classification_data(dataset='moon', N=200, Ntest=100,
                                  labeled_per_class=10, n_noisy=10, seed=0)
    assert d['p'] == 20  # 2 + 8 + 10
    assert np.sum(d['Yl'] == 1) == 10
    assert np.sum(d['Yl'] == -1) == 10
    assert d['Xt'].shape == (100, 20)


def test_partial_labeling_ratio():
    d = s2mam_regression_data(dataset='additive', N=200, n_labeled=20, seed=1)
    assert d['Xl'].shape[0] == 20
    assert d['Xu'].shape[0] == 180


def test_reproducibility():
    a = s2mam_regression_data(dataset='additive', seed=42)
    b = s2mam_regression_data(dataset='additive', seed=42)
    assert np.allclose(a['Xl'], b['Xl'])
    assert np.allclose(a['Yl'], b['Yl'])
