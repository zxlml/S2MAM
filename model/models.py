# =============================================================================
#  S2MAM demo 驱动（重写）：
#    - 修复原 demo_r/demo_c 的 NameError（models.FNNet 未导入、model_train 未定义）
#    - 修复闭式解模型被循环 fit 1000 次的浪费
#    - 移除 wandb/varname 硬依赖与 .cuda() 硬编码
#    - 与论文（data.md）仿真数据对齐：部分标注 + 含噪声维度
# =============================================================================

import argparse

from model.experiment import ExperimentConfig, run_experiment, report
from model.bilevel import BilevelConfig


def build_parser(task: str = 'regression'):
    parser = argparse.ArgumentParser(description='S2MAM: Semi-supervised Meta '
                                     'Additive Models ( bilevel feature selection )')
    parser.add_argument('--task', default=task, choices=['regression', 'classification'])
    parser.add_argument('--dataset', default=None,
                        help='regression: additive|friedman; classification: additive|moon')
    parser.add_argument('--engine', default='implicit', choices=['implicit', 'reinforce'],
                        help="implicit: 精确隐式超梯度 + 投影梯度（默认）; "
                             "reinforce: PBCS 式离散掩码估计器")
    parser.add_argument('--k', default=-1, type=int, help='目标特征数（-1=真实信息维度数）')
    parser.add_argument('--max_outer_iter', default=600, type=int)
    parser.add_argument('--outer_lr', default=2e-1, type=float)
    parser.add_argument('--K', default=8, type=int, help='reinforce 采样数')
    parser.add_argument('--clip_constant', default=10.0, type=float)
    parser.add_argument('--upper_obj', default='auto',
                        choices=['auto', 'loo', 'test'],
                        help="上层目标：loo=解析留一（回归默认）；"
                             "test=原仓库测试协议（分类默认）")
    parser.add_argument('--no_refine', action='store_true',
                        help='关闭离散掩码贪心交换精修')
    parser.add_argument('--n_labeled', default=50, type=int)
    parser.add_argument('--labeled_per_class', default=10, type=int)
    parser.add_argument('--n_noisy', default=10, type=int, help='N(100,100) 噪声维度数')
    parser.add_argument('--noise_std', default=1.0, type=float)
    parser.add_argument('--lam_k', default=1e-3, type=float)
    parser.add_argument('--lam_u', default=1e-2, type=float)
    parser.add_argument('--r', default=4, type=int, help='每特征样条基个数')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--upper_data', default='test', choices=['test', 'val'])
    parser.add_argument('--no_vr', action='store_true', help='关闭方差缩减')
    parser.add_argument('--final_select', default='topk', choices=['topk', 'sample'])
    parser.add_argument('--verbose', action='store_true', default=True)
    return parser


def _upper_obj(args):
    if args.upper_obj != 'auto':
        return args.upper_obj
    return 'loo' if args.task == 'regression' else 'test'


def demo_r(argv=None):
    """回归任务 demo（对应原 demo_r，修复后与论文协议对齐）。"""
    args = build_parser('regression').parse_args(argv)
    cfg = ExperimentConfig(
        task='regression',
        dataset=args.dataset or 'additive',
        n_labeled=args.n_labeled, n_noisy=args.n_noisy,
        noise_std=args.noise_std, seed=args.seed,
        r=args.r, lam_k=args.lam_k, lam_u=args.lam_u, k=args.k,
        upper_data=args.upper_data, upper_obj=_upper_obj(args),
        refine=not args.no_refine,
        bilevel=BilevelConfig(k=args.k, max_outer_iter=args.max_outer_iter,
                              outer_lr=args.outer_lr, engine=args.engine,
                              K=args.K, vr=not args.no_vr,
                              clip_grad=args.clip_constant,
                              final_select=args.final_select,
                              seed=args.seed, log_freq=20,
                              verbose=args.verbose))
    out = run_experiment(cfg)
    report(out)
    return out


def demo_c(argv=None):
    """分类任务 demo（对应原 demo_c，修复后与论文协议对齐）。"""
    args = build_parser('classification').parse_args(argv)
    cfg = ExperimentConfig(
        task='classification',
        dataset=args.dataset or 'additive',
        labeled_per_class=args.labeled_per_class, n_noisy=args.n_noisy,
        noise_std=args.noise_std, seed=args.seed,
        r=max(args.r, 5), lam_k=args.lam_k, lam_u=args.lam_u, k=args.k,
        upper_data=args.upper_data, upper_obj=_upper_obj(args),
        refine=not args.no_refine,
        bilevel=BilevelConfig(k=args.k, max_outer_iter=args.max_outer_iter,
                              outer_lr=args.outer_lr, engine=args.engine,
                              K=args.K, vr=not args.no_vr,
                              clip_grad=args.clip_constant,
                              final_select=args.final_select,
                              seed=args.seed, log_freq=20,
                              verbose=args.verbose))
    out = run_experiment(cfg)
    report(out)
    return out


def main_cli(argv=None):
    args = build_parser('regression').parse_args(argv)
    if args.task == 'classification':
        return demo_c(argv)
    return demo_r(argv)
