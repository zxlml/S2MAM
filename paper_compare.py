# -*- coding: utf-8 -*-
"""按论文 Table 2 协议复现可加分类结果并对比。

论文协议（arXiv 2604.19072v3 Table 2）：
  f*(x)=(x1-0.5)^2+(x2-0.5)^2-0.08, 200 样本, p*=2, r=5%（每类 5 个标注）,
  无信息维度 N(0,1), 噪声维度 N(100,100)。
  论文报告（100 次重复, Test ACC%）：
    pu=0,  pn=0 : 90.309+/-3.409
    pu=10, pn=0 : 86.015+/-3.575
    pu=0,  pn=10: 81.855+/-4.055
    pu=10, pn=10: 80.112+/-4.370

复现：每组设置 3 个 seed（0/1/2），报告均值+/-标准差。
"""
import numpy as np

from data.data_generation import s2mam_classification_data
from model.experiment import ExperimentConfig, run_experiment

PAPER = {
    (0, 0): (90.309, 3.409),
    (10, 0): (86.015, 3.575),
    (0, 10): (81.855, 4.055),
    (10, 10): (80.112, 4.370),
}

for (pu, pn), (p_mean, p_sd) in PAPER.items():
    accs, sel = [], None
    for seed in (0, 1, 2):
        cfg = ExperimentConfig(
            task='classification', dataset='additive',
            N=200, Ntest=200, labeled_per_class=5,
            n_uninformative=pu, n_noisy=pn,
            seed=seed, r=5, upper_obj='test')
        out = run_experiment(cfg)
        accs.append(out['metric']['acc'] * 100)
        if seed == 0:
            sel = out['selection']['selected']
    m, sd = float(np.mean(accs)), float(np.std(accs))
    print('pu=%2d pn=%2d | paper %.3f+/-%.3f | ours %.3f+/-%.3f | diff %+.3f | sel(seed0)=%s'
          % (pu, pn, p_mean, p_sd, m, sd, m - p_mean, sel))
