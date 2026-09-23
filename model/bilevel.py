# =============================================================================
#  S2MAM 上层问题：特征选择的一阶投影梯度法（borrowed & adapted from PBCS）
# =============================================================================
#  上层变量：特征选择分数 m ∈ [0,1]^p，基数约束 sum(m) = k。
#  可行域 C = {m ∈ [0,1]^p, sum(m) = k} 为凸紧集。
#
#  两种梯度引擎：
#  * engine='implicit'（默认，S2MAM 特有）：
#      下层闭式可解 ⇒ 对 F(m) = L_val(beta*(m), m) 做 autograd 穿过
#      torch.linalg.solve 的精确隐式超梯度，然后投影梯度迭代
#          m_{t+1} = Proj_C(m_t - eta_t * grad F(m_t)) .
#      步长 eta_t 余弦衰减、梯度裁剪 ⇒ 投影梯度法收敛到稳定点。
#  * engine='reinforce'（PBCS 工程设计，用于离散掩码精调/核方法下层）：
#      伯努利采样离散掩码 + score-function 梯度估计
#          g_i = (z_i - m) / (m(1-m))  （∇ log p(z_i; m)）
#          grad ≈ (1/K) sum_i fn_i * g_i   （方差缩减：fn_i - fn_avg）
#      下层可为任意 fit/predict 模型（LapSVM 等）。
#
#  与 PBCS 的差异：PBCS 下层是 SGD 训练的神经网络，超梯度只能通过
#  单步伪更新 + Hessian-vector product 有偏近似；S2MAM 下层是
#  部分标注 + Laplacian 正则 + 可加模型的强凸二次问题，闭式解使得
#  超梯度精确、且上层投影迭代具备一阶收敛保证。
# =============================================================================

from dataclasses import dataclass, field

import numpy as np
import torch


# --------------------------------------------------------------------- config
@dataclass
class BilevelConfig:
    k: int = 8                    # 目标选中特征数（基数约束）
    max_outer_iter: int = 600     # 上层迭代数
    outer_lr: float = 2e-1        # 上层初始学习率（余弦衰减）
    engine: str = 'implicit'      # 'implicit' | 'reinforce'
    optimizer: str = 'sgd'        # implicit 引擎的上层优化器
    K: int = 8                    # reinforce 引擎的采样数
    vr: bool = True               # 方差缩减（control variate，PBCS 设计）
    clip_grad: float = 10.0       # 梯度范数裁剪（PBCS 设计）
    iterative: bool = True        # 迭代基数调度（PBCS ts/te cubic 调度）：
                                  # 从全特征出发逐步剪枝，梯度判别力更强
    ts: float = 0.16
    te: float = 0.6
    start_k: int = -1             # iterative=True 时的起始基数（-1 表示全特征）
    final_select: str = 'topk'    # 最终离散化：'topk' | 'sample'
    seed: int = 0
    log_freq: int = 20
    verbose: bool = True


# ------------------------------------------------------------------- 投影算子
def solve_v_total(weight: torch.Tensor, k: float) -> float:
    """二分法求 v 使 sum(clamp(m - v, 0, 1)) = k（PBCS 同款）。"""
    a, b = 0.0, float(weight.max().item())
    w = weight.detach()

    def f(v):
        return float((w - v).clamp(0, 1).sum().item()) - k

    if f(0.0) <= 0:
        return 0.0
    v = b
    for _ in range(60):
        v = 0.5 * (a + b)
        obj = f(v)
        if abs(obj) < 1e-10:
            break
        if obj < 0:
            b = v
        else:
            a = v
    return max(0.0, v)


def project_cardinality(scores: torch.Tensor, k: float) -> torch.Tensor:
    """到 C = {m ∈ [0,1]^p, sum(m)=k} 的精确欧氏投影（软阈值形式）。

    即一阶投影梯度法中的 Proj_C；在 torch.no_grad 下原地更新。
    """
    with torch.no_grad():
        v = solve_v_total(scores, k)
        scores.sub_(v).clamp_(0, 1)
    return scores


def project_cardinality_np(x: np.ndarray, k: float) -> np.ndarray:
    t = torch.tensor(np.asarray(x, dtype=float), dtype=torch.float64)
    return project_cardinality(t, k).numpy()


# ------------------------------------------------------- REINFORCE 采样估计器
def obtain_mask(scores: torch.Tensor, generator: torch.Generator = None):
    """伯努利采样离散掩码 + score-function 梯度（PBCS 同款估计器）。

    返回 (subnet ∈ {0,1}^p, grad = (z - m)/(m(1-m)) = ∇_m log p(z; m))。
    """
    z = (torch.rand(scores.shape, generator=generator,
                    dtype=scores.dtype) < scores).float()
    grad = (z - scores) / ((scores + 1e-8) * (1.0 - scores + 1e-8))
    return z, grad


def reinforce_grad(scores: torch.Tensor, fn_list, grad_list, fn_avg, use_vr=True):
    """REINFORCE 梯度聚合（带可选 control-variate 方差缩减）。"""
    g = torch.zeros_like(scores)
    if use_vr:
        for fn, grad in zip(fn_list, grad_list):
            g += (1.0 / (len(fn_list) - 1)) * (fn - fn_avg) * grad
    else:
        for fn, grad in zip(fn_list, grad_list):
            g += (1.0 / len(fn_list)) * fn * grad
    return g


# ------------------------------------------------------------ 迭代基数调度
def cardinality_schedule(t: int, T: int, k_target: int, k_start: int,
                         ts: float, te: float) -> int:
    """PBCS 的 ts/te 三次缓动基数调度（用于 iterative 模式）。"""
    if k_start <= 0 or k_start == k_target:
        return k_target
    s, e = int(ts * T), int(te * T)
    if t < s:
        return k_start
    if t >= e:
        return k_target
    ratio = 1.0 - (t - s) / max(1, (e - s))
    return int(round(k_target + (k_start - k_target) * ratio ** 3))


def cosine_lr(t: int, T: int, base_lr: float) -> float:
    return 0.5 * (1 + np.cos(np.pi * t / T)) * base_lr


# ------------------------------------------------------------------ 主求解器
def solve_bilevel(upper_fn=None, m0=None, cfg: BilevelConfig = None,
                  upper_fn_value=None):
    """上层一阶投影梯度求解。

    参数
    ----
    upper_fn : callable(torch.Tensor m) -> torch.Tensor 标量
        可微上层目标（implicit 引擎必需）。
    upper_fn_value : callable(binary mask tensor) -> float
        离散掩码上层目标值（reinforce 引擎必需；给 implicit 引擎时仅用于记录）。
    m0 : 初始分数（p,），默认均匀 k/p。
    cfg : BilevelConfig

    返回 dict：
        m_soft（投影后的连续分数）、mask（最终二值掩码）、
        history（每步 {iter, loss, k, grad_norm}）、best_mask（reinforce 引擎
        采样中上层值最优的离散掩码）。
    """
    cfg = cfg or BilevelConfig()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    engine = cfg.engine
    assert engine in ('implicit', 'reinforce')

    if m0 is None:
        raise ValueError('需提供初始分数 m0（推荐 np.full(p, k/p) 后投影）')
    m0 = torch.as_tensor(np.asarray(m0, dtype=float), dtype=torch.float64)
    p = m0.numel()
    k_start = cfg.start_k if cfg.start_k > 0 else p

    m = m0.clone()
    project_cardinality(m, k_start if cfg.iterative else cfg.k)

    scores_opt = None
    if engine == 'implicit':
        m.requires_grad_(True)
        if cfg.optimizer == 'adam':
            scores_opt = torch.optim.Adam([m], lr=cfg.outer_lr)
        else:
            scores_opt = torch.optim.SGD([m], lr=cfg.outer_lr)

    history = []
    best_mask, best_value = None, np.inf
    T = cfg.max_outer_iter

    for t in range(T):
        k_t = cardinality_schedule(t, T, cfg.k, k_start if cfg.iterative else cfg.k,
                                   cfg.ts, cfg.te) if cfg.iterative else cfg.k
        lr_t = cosine_lr(t, T, cfg.outer_lr)
        if scores_opt is not None:
            for gparam in scores_opt.param_groups:
                gparam['lr'] = lr_t

        if engine == 'implicit':
            loss = upper_fn(m)
            grad = torch.autograd.grad(loss, m)[0]
            # 梯度裁剪（PBCS 工程设计）
            if cfg.clip_grad and cfg.clip_grad > 0:
                gn = grad.norm()
                if gn > cfg.clip_grad:
                    grad = grad * (cfg.clip_grad / (gn + 1e-12))
            m.grad = grad
            scores_opt.step()
            m.grad = None
            project_cardinality(m, k_t)
            loss_val = float(loss.item())
            grad_norm = float(grad.norm().item())
        else:
            fn_list, grad_list = [], []
            fn_avg = 0.0
            for _ in range(cfg.K):
                z, grad_i = obtain_mask(m)
                fn = float(upper_fn_value(z))
                fn_list.append(fn)
                grad_list.append(grad_i)
                fn_avg += fn / cfg.K
                if fn < best_value:
                    best_value = fn
                    best_mask = z.clone()
            grad = reinforce_grad(m, fn_list, grad_list, fn_avg, cfg.vr)
            if cfg.clip_grad and cfg.clip_grad > 0:
                gn = grad.norm()
                if gn > cfg.clip_grad:
                    grad = grad * (cfg.clip_grad / (gn + 1e-12))
            m = (m - lr_t * grad).detach()
            project_cardinality(m, k_t)
            loss_val = fn_avg
            grad_norm = float(grad.norm().item())

        history.append({'iter': t, 'loss': loss_val, 'k': k_t,
                        'grad_norm': grad_norm})
        if cfg.verbose and (t % cfg.log_freq == 0 or t == T - 1):
            print(f'[bilevel] iter {t:4d}  k={k_t}  lr={lr_t:.4g}  '
                  f'loss={loss_val:.6f}  |g|={grad_norm:.4g}')

    with torch.no_grad():
        m_soft = m.detach().clone()
    if cfg.final_select == 'sample':
        mask = (torch.rand(p) < m_soft).numpy().astype(int)
        if mask.sum() == 0:  # 兜底：保证非空
            mask[np.argsort(-m_soft.numpy())[:cfg.k]] = 1
    else:
        idx = np.argsort(-m_soft.numpy())[:cfg.k]
        mask = np.zeros(p, dtype=int)
        mask[idx] = 1

    return {'m_soft': m_soft.numpy(), 'mask': mask, 'history': history,
            'best_mask': best_mask.numpy() if best_mask is not None else None}
