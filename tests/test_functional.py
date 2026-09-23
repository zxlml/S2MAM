# -*- coding: utf-8 -*-
"""端到端功能测试：论文仿真数据 + 部分标注 + 含噪声维度。

验证目标：
  1. 双层优化选出的特征应高度覆盖真实信息维度（噪声维度被剔除）；
  2. 选择后模型的测试效果应不劣于全特征基线、接近 oracle；
  3. 训练/测试效果应接近论文声称的水平（可加数据上接近噪声底限）。
"""

import numpy as np
import pytest

from model.experiment import ExperimentConfig, run_experiment
from model.bilevel import BilevelConfig


def _bcfg(engine='implicit', iters=600, k=-1, seed=0):
    return BilevelConfig(k=k, max_outer_iter=iters, outer_lr=2e-1,
                         engine=engine, optimizer='sgd', K=6, vr=True,
                         clip_grad=10.0, seed=seed, verbose=False, log_freq=50)


def test_regression_additive_partial_labels_noisy_dims():
    """data.md (2)：8 信息维 + 92 无信息维 + 10 噪声维，50/200 部分标注。"""
    cfg = ExperimentConfig(task='regression', dataset='additive',
                           N=200, Ntest=200, n_labeled=50, n_noisy=10,
                           noise_std=1.0, seed=0,
                           bilevel=_bcfg())
    out = run_experiment(cfg)
    sel = out['selection']
    assert sel['recall'] >= 0.75, f"recall={sel['recall']}"
    assert sel['precision'] >= 0.6, f"precision={sel['precision']}"
    # 效果：不劣于全特征；接近 oracle（不超过其 MSE 的 4 倍）
    mse = out['metric']['mse']
    assert mse <= out['baseline_full']['mse'] * 1.05
    assert mse <= out['baseline_oracle']['mse'] * 4.0
    # 噪声维度（最后 10 维）不应被选中
    sel_set = set(sel['selected'])
    assert not (sel_set & set(range(100, 110))), \
        f'噪声维度被选中: {sorted(sel_set & set(range(100, 110)))}'


def test_regression_friedman_partial_labels():
    """data.md (1)：Friedman 5 信息维 + 95 无信息维 + 10 噪声维。

    注：Friedman 含交互项 10·sin(πx1x2)，可加下层只能部分恢复，
    故 recall 阈值放宽至 0.6。
    """
    cfg = ExperimentConfig(task='regression', dataset='friedman',
                           N=200, Ntest=200, n_labeled=50, n_noisy=10,
                           noise_std=1.0, seed=0,
                           bilevel=_bcfg())
    out = run_experiment(cfg)
    sel = out['selection']
    assert sel['recall'] >= 0.6, f"recall={sel['recall']}"
    assert out['metric']['mse'] <= out['baseline_full']['mse'] * 1.1


def test_classification_additive_partial_labels():
    """data.md (3)：2 信息维 + 98 无信息维 + 10 噪声维，每类 10 个标注。

    分类可加模型的 LOO 解析目标不稳（小样本二分类），采用原仓库的
    test 协议（upper_obj='test'）。
    """
    cfg = ExperimentConfig(task='classification', dataset='additive',
                           N=200, Ntest=200, labeled_per_class=10, n_noisy=10,
                           seed=0, r=5, upper_obj='test',
                           bilevel=_bcfg())
    out = run_experiment(cfg)
    sel = out['selection']
    # 两个信息维度都应被找到
    assert sel['recall'] >= 1.0, f"selected={sel['selected']}"
    assert out['metric']['acc'] >= 0.85, f"acc={out['metric']['acc']}"


def test_classification_moon_reinforce_engine():
    """data.md (4)：双月牙 + 无信息/噪声维度，REINFORCE 引擎可运行且有效。

    双月牙含交互结构，可加下层无法表达，改用 LapSVM 下层（原仓库做法，
    experiment.py 自动切换）。
    """
    cfg = ExperimentConfig(task='classification', dataset='moon',
                           N=200, Ntest=100, labeled_per_class=10, n_noisy=10,
                           seed=0, r=5,
                           bilevel=_bcfg(engine='reinforce', iters=30))
    out = run_experiment(cfg)
    sel = out['selection']
    assert sel['recall'] >= 0.5
    assert out['metric']['acc'] >= 0.7, f"acc={out['metric']['acc']}"
