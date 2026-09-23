# =============================================================================
#  兼容层：分类侧的 LapSVM 等实现统一收敛到 model/progress_r.py
#  （原文件在模块导入时执行 argparse + wandb.init，已移除该副作用）
# =============================================================================

from model.progress_r import LapSVM, LapRLS, rbf  # noqa: F401


def lapsvm_default_opt():
    """与原 demo_c/demo_r 中一致的默认超参数。"""
    return {'neighbor_mode': 'connectivity',
            'n_neighbor': 10,
            't': 10,
            'kernel_function': rbf,
            'kernel_parameters': {'gamma': 1},
            'gamma_A': 0.001,
            'gamma_I': 0.01}
