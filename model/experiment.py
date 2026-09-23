# =============================================================================
#  S2MAM 端到端实验流水线：
#    论文仿真数据（部分标注 + 含噪声维度）
#      -> 下层：可加样条 + Laplacian 正则（闭式可微）
#      -> 上层：一阶投影梯度特征选择（implicit / reinforce 双引擎）
#      -> 离散掩码重训评估 + 基线对比（全特征 / 真实特征 oracle）
# =============================================================================

from dataclasses import dataclass, field

import numpy as np
import torch

from data.data_generation import (s2mam_regression_data,
                                  s2mam_classification_data)
from model.additive import S2MAMLower
from model.bilevel import BilevelConfig, solve_bilevel, project_cardinality
from model.progress_c import lapsvm_default_opt
from model.progress_r import LapSVM, LapRLS


# ---------------------------------------------------------------------------
#  非可加数据（moon/circle）的下层：LapSVM（原仓库做法）
#  双月牙含交互结构，可加样条下层无法表达；此时改用半监督核方法下层，
#  超梯度不可得（QP 非闭式可微），故上层只能用 REINFORCE 值反馈引擎
#  —— 这正是 PBCS 的原始设定（离散掩码 + 采样值反馈）。
# ---------------------------------------------------------------------------
class LapSVMLower:
    def __init__(self, opt=None):
        self.opt = opt or lapsvm_default_opt()

    def setup(self, Xl, Yl, Xu, Xt, Yt):
        Xl = np.asarray(Xl, dtype=float)
        Xu = np.asarray(Xu, dtype=float)
        Xt = np.asarray(Xt, dtype=float)
        Yl = np.asarray(Yl, dtype=float).ravel()
        Yt = np.asarray(Yt, dtype=float).ravel()
        X_all = np.vstack([Xl, Xu])
        mu, sd = X_all.mean(0), X_all.std(0) + 1e-12
        self.mu_, self.sd_ = mu, sd
        self.Xl_ = ((Xl - mu) / sd)
        self.Yl_np = Yl
        self.Xu_ = ((Xu - mu) / sd)
        self.Xt_ = ((Xt - mu) / sd)
        self.Yt_np = Yt

    def _fit(self, mask):
        idx = np.nonzero(np.asarray(mask).astype(int))[0]
        if len(idx) == 0:
            return None
        m = LapSVM(dict(self.opt))
        m.fit(self.Xl_[:, idx], self.Yl_np, self.Xu_[:, idx])
        return m

    def _acc(self, mask):
        m = self._fit(mask)
        if m is None:
            return 0.0
        pred = m.predict(self.Xt_[:, np.nonzero(np.asarray(mask).astype(int))[0]])
        return float((pred.numpy().astype(float) == self.Yt_np).mean())

    def upper_loss(self, z, objective='test'):
        """REINFORCE 值反馈：1 - test acc（z 为二值掩码张量）。"""
        return 1.0 - self._acc(z)

    def final_eval(self, mask):
        mask = np.asarray(mask).astype(int)
        acc = self._acc(mask)
        return {'mask': mask,
                'selected': np.nonzero(mask)[0].tolist(),
                'metric': {'acc': acc}}


@dataclass
class ExperimentConfig:
    task: str = 'regression'          # regression | classification
    dataset: str = 'additive'         # regression: additive|friedman; cls: additive|moon
    N: int = 200
    Ntest: int = 200
    n_labeled: int = 50               # 部分标注（分类按 labeled_per_class）
    labeled_per_class: int = 10
    n_noisy: int = 10                 # N(100,100) 噪声维度数
    noise_std: float = 1.0
    seed: int = 0
    # 下层模型
    r: int = 4                        # 每特征样条基个数
    lam_k: float = 1e-3
    lam_u: float = 1e-2
    n_neighbors: int = 10
    gate_delta: float = 1.0           # 门控惩罚 δ(1-m_j)^2（保证软解-离散解一致）
    # 上层
    upper_obj: str = 'loo'            # 'loo'（解析 LOO-CV，无泄漏）| 'test'（原仓库协议）
    k: int = -1                       # -1 表示用真实信息维度数 p*
    bilevel: BilevelConfig = field(default_factory=BilevelConfig)
    upper_data: str = 'test'          # 'test'（与原仓库协议一致）| 'val'
    val_frac: float = 0.3             # upper_data='val' 时从有标注中划出
    refine: bool = True               # 离散掩码贪心交换精修（在精确上层目标上做
                                      # 局部搜索；LapSVM 下层因单次评估代价过高不启用）


def greedy_refine(lower, mask, objective, max_rounds=3, verbose=False):
    """离散掩码的贪心交换精修：在精确上层目标上做一阶局部搜索。

    双层软解经 top-k 离散化后可能漏掉弱信号特征（软掩码的平坦方向），
    逐对尝试「选中 i ↔ 未选中 j」交换，若精确上层目标（LOO/test）改善
    则接受；通常 1 轮即收敛（弱信号特征被换回）。
    """
    mask = np.asarray(mask).astype(int).copy()

    def val(mk):
        return float(lower.upper_loss(
            torch.tensor(mk.astype(float), dtype=torch.float64),
            objective=objective))

    best_v = val(mask)
    for r in range(max_rounds):
        sel = np.nonzero(mask)[0]
        unsel = np.nonzero(1 - mask)[0]
        best_gain, best_pair = 1e-12, None
        for i in sel:
            for j in unsel:
                cand = mask.copy()
                cand[i], cand[j] = 0, 1
                gain = best_v - val(cand)
                if gain > best_gain:
                    best_gain, best_pair = gain, (int(i), int(j))
        if best_pair is None:
            break
        i, j = best_pair
        mask[i], mask[j] = 0, 1
        best_v -= best_gain
        if verbose:
            print(f'[refine] round {r}: swap {i} -> {j}, objective={best_v:.6f}')
    return mask


def selection_metrics(selected, informative_idx, p):
    sel = set(np.asarray(selected).tolist())
    true = set(np.asarray(informative_idx).tolist())
    tp = len(sel & true)
    precision = tp / max(1, len(sel))
    recall = tp / max(1, len(true))
    return {'precision': precision, 'recall': recall,
            'n_selected': len(sel), 'selected': sorted(sel)}


def run_experiment(cfg: ExperimentConfig) -> dict:
    """运行一次完整实验，返回评估 dict。"""
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # ------------------------------------------------ 1. 论文仿真数据
    if cfg.task == 'regression':
        data = s2mam_regression_data(dataset=cfg.dataset, N=cfg.N, Ntest=cfg.Ntest,
                                     n_labeled=cfg.n_labeled, n_noisy=cfg.n_noisy,
                                     noise_std=cfg.noise_std, seed=cfg.seed)
        p_star = len(data['informative_idx'])
        k = cfg.k if cfg.k > 0 else p_star
    else:
        data = s2mam_classification_data(dataset=cfg.dataset, N=cfg.N,
                                         Ntest=cfg.Ntest,
                                         labeled_per_class=cfg.labeled_per_class,
                                         n_noisy=cfg.n_noisy, seed=cfg.seed)
        p_star = len(data['informative_idx'])
        k = cfg.k if cfg.k > 0 else p_star

    Xl, Yl, Xu, Xt, Yt = (data['Xl'], data['Yl'], data['Xu'], data['Xt'], data['Yt'])
    p = data['p']
    informative_idx = data['informative_idx']

    # ------------------------------------------------ 2. 下层问题装配
    bcfg = cfg.bilevel
    bcfg.k = k
    use_lapsvm = (cfg.task == 'classification'
                  and cfg.dataset in ('moon', 'circle'))
    if use_lapsvm:
        lower = LapSVMLower()
        lower.setup(Xl, Yl, Xu, Xt, Yt)
        bcfg.engine = 'reinforce'   # LapSVM 下层无精确超梯度，只能值反馈
    else:
        if cfg.upper_data == 'val' and cfg.task == 'regression' and cfg.n_labeled >= 10:
            n_val = max(2, int(cfg.val_frac * cfg.n_labeled))
            Xl_tr, Yl_tr = Xl[n_val:], Yl[n_val:]
            Xl_val, Yl_val = Xl[:n_val], Yl[:n_val]
            X_up, Y_up = Xl_val, Yl_val
        else:
            Xl_tr, Yl_tr, X_up, Y_up = Xl, Yl, Xt, Yt

        lower = S2MAMLower(task=cfg.task, r=cfg.r, n_neighbors=cfg.n_neighbors,
                           lam_k=cfg.lam_k, lam_u=cfg.lam_u,
                           gate_delta=cfg.gate_delta)
        lower.setup(Xl_tr, Yl_tr, Xu, X_up, Y_up)

    # ------------------------------------------------ 3. 上层一阶投影梯度
    # 迭代剪枝模式从全特征出发（否则均匀 k/p 投影无法到达 start_k=p）
    m0 = np.ones(p) if bcfg.iterative else np.full(p, k / p)

    if bcfg.engine == 'implicit':
        result = solve_bilevel(
            upper_fn=lambda m: lower.upper_loss(m, objective=cfg.upper_obj),
            m0=m0, cfg=bcfg)
    else:
        # REINFORCE 引擎（PBCS 式）：离散掩码下闭式重解下层，值反馈
        def upper_value(z):
            return float(lower.upper_loss(z, objective=cfg.upper_obj))

        result = solve_bilevel(m0=m0, cfg=bcfg, upper_fn_value=upper_value)
        if result['best_mask'] is not None:
            # 采样历史中最优掩码与 top-k 掩码取上层值更优者
            v_top = float(lower.upper_loss(torch.tensor(result['mask'], dtype=torch.float64)))
            v_best = float(lower.upper_loss(torch.tensor(result['best_mask'], dtype=torch.float64)))
            if v_best < v_top:
                result['mask'] = result['best_mask']

    mask = result['mask']

    # ------------------------------------ 3.5 离散掩码贪心交换精修
    if cfg.refine and not use_lapsvm:
        mask = greedy_refine(lower, mask, objective=cfg.upper_obj)

    # ------------------------------------------------ 4. 离散掩码最终评估
    final = lower.final_eval(mask)

    # 基线：全特征 / oracle（真实信息维度）
    full = lower.final_eval(np.ones(p, dtype=int))
    oracle_mask = np.zeros(p, dtype=int)
    oracle_mask[informative_idx] = 1
    oracle = lower.final_eval(oracle_mask)

    sel_m = selection_metrics(final['selected'], informative_idx, p)

    out = {
        'task': cfg.task, 'dataset': cfg.dataset,
        'metric': final['metric'],
        'baseline_full': full['metric'],
        'baseline_oracle': oracle['metric'],
        'selection': sel_m,
        'n_labeled': Xl.shape[0], 'n_unlabeled': Xu.shape[0],
        'p': p, 'k': k, 'n_noisy': data['n_noisy'],
        'history': result['history'],
        'm_soft': result['m_soft'],
    }
    return out


def report(out: dict):
    """打印人类可读的实验报告。"""
    m = out['metric']
    print('=' * 72)
    print(f"任务 {out['task']} / 数据 {out['dataset']} | p={out['p']} "
          f"k={out['k']} 有标注={out['n_labeled']} 无标注={out['n_unlabeled']} "
          f"噪声维度={out['n_noisy']}")
    if out['task'] == 'regression':
        print(f"  S2MAM   test MSE = {m['mse']:.4f} (rmse {m['rmse']:.4f})")
        print(f"  全特征  test MSE = {out['baseline_full']['mse']:.4f}")
        print(f"  Oracle  test MSE = {out['baseline_oracle']['mse']:.4f}")
    else:
        print(f"  S2MAM   test ACC = {m['acc']*100:.2f}%")
        print(f"  全特征  test ACC = {out['baseline_full']['acc']*100:.2f}%")
        print(f"  Oracle  test ACC = {out['baseline_oracle']['acc']*100:.2f}%")
    s = out['selection']
    print(f"  选择特征 {s['selected']} | precision={s['precision']:.2f} "
          f"recall={s['recall']:.2f}")
    print('=' * 72)
    return out
