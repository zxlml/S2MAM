<div align="center">

# S2MAM: Semi-supervised Meta Additive Model for Robust Estimation and Variable Selection

**基于精确隐式超梯度的双层优化特征选择（部分标注场景）**

[![Paper](https://img.shields.io/badge/Paper-arXiv%3A2604.19072-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2604.19072) [![English](https://img.shields.io/badge/Language-English-blue?style=for-the-badge)](README.md) [![简体中文](https://img.shields.io/badge/Language-简体中文-red?style=for-the-badge)](README_zh-CN.md)

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/Tests-29%20passed-brightgreen?logo=pytest&logoColor=white)](tests/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

</div>

---

S2MAM 是**半监督元可加模型（Semi-supervised Meta Additive Models）**的官方 PyTorch 实现，面向**部分标注**样本、**海量冗余维度**与**极端噪声特征**场景的双层优化特征选择框架：

- **下层问题**：半监督可加模型——逐特征 B 样条基 + 光滑性惩罚 + 跨标注/无标注样本的**图 Laplacian 正则**，具有**闭式解**。
- **上层问题**：基数约束下的特征选择分数 `m ∈ [0,1]^p`，通过**一阶投影梯度法**求解，超梯度为**精确隐式微分**（autograd 穿过下层闭式解）。

> **与 PBCS 的差异。** 在 [Probabilistic Bilevel Coreset Selection（PBCS）](https://github.com/x-zho14/Probabilistic-Bilevel-Coreset-Selection)中，下层模型是 SGD 训练的神经网络，超梯度只能通过单步伪更新 + Hessian-vector 乘积有偏近似。而 S2MAM 的下层（部分标注 + Laplacian 正则可加模型）是强凸二次问题，闭式可解，因此超梯度**精确**，上层投影梯度迭代在凸紧集 `{m ∈ [0,1]^p, Σm = k}` 上具备收敛保证。PBCS 式的 REINFORCE 机制被保留为备选引擎，用于下层无可微闭式解的情形（如双月牙等非可加数据上的 LapSVM）。

## 📑 目录

- [✨ 核心亮点](#-核心亮点)
- [📢 新闻](#-新闻)
- [🔧 安装](#-安装)
- [⚡ 快速开始](#-快速开始)
- [🏗️ 项目结构](#️-项目结构)
- [⚙️ 配置说明](#️-配置说明)
- [📁 实验结果](#-实验结果)
- [✅ 测试](#-测试)
- [☑️ 计划](#️-计划)
- [🤝 致谢](#-致谢)
- [📖 引用](#-引用)
- [📄 许可证](#-许可证)

## ✨ 核心亮点

- **精确隐式超梯度**——下层闭式求解（`torch.linalg.solve`），autograd 直接对解微分，得到真实超梯度 `∂F/∂m`。
- **解析 LOO-CV 上层目标**——通过帽子矩阵对角元 `h_ii = (Φ_l * (Φ_l A⁻¹)).sum(1)/l` 计算线性平滑器的留一残差；无需划分验证集，无测试集泄漏。
- **门控惩罚** `δ(1−m_j)²‖β_j‖²`——打破掩码可加模型的"衰减–重优化抵消"不变性（包络平坦性），保证软解与离散 top-k 选择一致。
- **借鉴并改进 PBCS 工程设计**——带控制变量方差缩减的 REINFORCE 伯努利估计器、梯度裁剪、余弦学习率衰减、二分法基数投影、ts/te 三次缓动的**迭代剪枝**调度（从全特征出发逐步剪枝）。
- **贪心交换精修**——top-k 离散化后，在精确上层目标上逐对评估「选中 ↔ 未选中」交换，换回弱信号特征。
- **双引擎**——`implicit`（默认；精确超梯度）与 `reinforce`（PBCS 式值反馈；下层为 LapSVM 时自动选用，如非可加的双月牙数据）。
- **鲁棒性设计**——在部分标注（25% 标注率）、92–98% 冗余维度、极端噪声特征 `N(100,100)` 下验证。

## 📢 新闻

- **[2026-09]** 代码重构：可微下层（`model/additive.py`）、一阶投影梯度上层（`model/bilevel.py`）、端到端流水线（`model/experiment.py`）、完整单元 + 功能测试（`tests/`，29 个测试全部通过）。
- **[2026-09]** 修复原仓库缺陷：模块级 `argparse` 副作用、demo 驱动中未定义变量、`LapRLS` 的 Laplacian 构造错误、闭式解模型被循环 fit 1000 次的浪费。

## 🔧 安装

```bash
# 1. 克隆仓库
git clone https://github.com/zxlml/S2MAM.git
cd S2MAM

# 2. 创建环境
conda create -n s2mam python=3.10 -y
conda activate s2mam

# 3. 安装依赖
pip install torch numpy scipy scikit-learn pandas openpyxl pytest
```

> **说明**——全部功能均可在纯 CPU 机器上运行（默认闭式下层求解器使用 float64 CPU）；GPU 为可选项。
>
> Windows 上若遇到 `OMP: Error #15 (libiomp5md.dll already initialized)`，请设置 `KMP_DUPLICATE_LIB_OK=TRUE`。

## ⚡ 快速开始

在仓库根目录运行论文的四个仿真实验（部分标注 + 噪声维度）：

```bash
# (2) 可加回归：8 信息维 + 92 冗余维 + 10 噪声维，50/200 标注
python main.py --task regression --dataset additive

# (1) Friedman 回归：5 信息维 + 95 冗余维 + 10 噪声维
python main.py --task regression --dataset friedman

# (3) 可加分类：2 信息维 + 98 冗余维 + 10 噪声维，每类 10 个标注
python main.py --task classification --dataset additive

# (4) 双月牙分类（非可加 -> LapSVM 下层 + REINFORCE 引擎）
python main.py --task classification --dataset moon --engine reinforce --max_outer_iter 40
```

每次运行会打印所选特征、测试指标（MSE / 准确率）以及**全特征**与 **oracle** 基线的对比报告。也可以在代码中调用：

```python
from model.experiment import ExperimentConfig, run_experiment, report

cfg = ExperimentConfig(task='regression', dataset='additive',
                       N=200, Ntest=200, n_labeled=50, n_noisy=10,
                       noise_std=1.0, seed=0)
out = run_experiment(cfg)
report(out)
```

### 运行测试

```bash
python -m pytest tests -q        # 29 个单元 + 功能测试
```

## 🏗️ 项目结构

```
S2MAM/
├── main.py                    # CLI 入口（回归 / 分类 demo）
├── model/
│   ├── additive.py            # 下层：可微闭式半监督可加模型（样条展开、光滑性
│   │                          #   惩罚、图 Laplacian、门控惩罚、解析 LOO）
│   ├── bilevel.py             # 上层：一阶投影梯度法（精确隐式超梯度 / REINFORCE
│   │                          #   双引擎、二分投影、余弦学习率、迭代剪枝调度）
│   ├── experiment.py          # 端到端流水线 + 基线对比 + 贪心交换精修
│   ├── models.py              # demo_r / demo_c / CLI 解析（无模块级副作用）
│   ├── progress_r.py          # LapRLS / LapSVM / RBF 核（原仓库实现，已修复 bug）
│   └── progress_c.py          # 分类侧兼容层
├── data/
│   ├── data_generation.py     # 论文仿真数据：部分标注 + N(100,100) 噪声维度
│   └── data.md                # 论文数据集描述
├── tests/                     # 29 个单元 + 功能测试（pytest）
│   ├── test_additive.py       # 闭式解正确性、有限差分超梯度验证
│   ├── test_bilevel.py        # 投影算子、基数调度、双引擎
│   ├── test_data.py           # 与论文一致的仿真数据
│   └── test_functional.py     # 端到端：相对 oracle / 全特征基线的选择质量断言
├── hypergrad/                 # 超梯度工具（沿用原仓库）
├── coreset_utils/             # 工具函数（沿用原仓库）
├── reinforce_utils/           # REINFORCE 辅助（沿用原仓库）
└── logging_utils/             # 日志 / 目录管理（沿用原仓库）
```

## ⚙️ 配置说明

所有行为由 `ExperimentConfig`（或等价的 CLI 参数）控制。主要选项：

| 选项 | 默认值 | 说明 |
| --- | --- | --- |
| `task` | `'regression'` | `regression`（回归）或 `classification`（分类） |
| `dataset` | `'additive'` | 回归：`additive`、`friedman`；分类：`additive`、`moon` |
| `n_labeled` / `labeled_per_class` | `50` / `10` | 部分标注预算（回归 / 分类） |
| `n_noisy` | `10` | 追加的极端噪声特征 `N(100,100)` 个数 |
| `upper_obj` | `'loo'` | 上层目标：`loo` = 解析留一（无泄漏），`test` = 原仓库测试协议 |
| `gate_delta` | `1.0` | 门控惩罚系数 `δ(1−m_j)²` |
| `r` | `4` / `5` | 每特征 B 样条基个数（回归 / 分类） |
| `lam_k` / `lam_u` | `1e-3` / `1e-2` | 光滑性惩罚 / Laplacian 正则强度 |
| `bilevel.engine` | `'implicit'` | `implicit`（精确超梯度）或 `reinforce`（PBCS 式） |
| `bilevel.iterative` | `True` | 迭代剪枝调度（ts/te 三次缓动，从全特征出发） |
| `bilevel.max_outer_iter` | `600` | 上层迭代次数 |
| `bilevel.outer_lr` | `0.2` | 上层初始学习率（余弦衰减） |
| `bilevel.clip_grad` | `10.0` | 梯度范数裁剪 |
| `refine` | `True` | 离散掩码贪心交换精修 |

等价 CLI 参数：`--task --dataset --engine --k --max_outer_iter --outer_lr --K --clip_constant --upper_obj --no_refine --n_labeled --labeled_per_class --n_noisy --noise_std --lam_k --lam_u --r --seed --upper_data --no_vr --final_select --verbose`。

## 📁 实验结果

本 README 不列写实验结果。机器可读的复现汇总见 [`results/summary.csv`](results/summary.csv)，列含义与范围说明见 [`results/summary_readme.txt`](results/summary_readme.txt)。可按[快速开始](#-快速开始)中的命令或 `python paper_compare.py` 复现。

## ✅ 测试

```bash
python -m pytest tests -q
# .............................  [100%]
# 29 passed
```

测试覆盖：B 样条单位分解、惩罚矩阵半正定性、Laplacian 性质、**闭式解与直接岭回归参考实现一致性**、**超梯度与中心有限差分一致性**、投影算子正确性、基数调度、双引擎、论文一致的仿真数据，以及部分标注 + 噪声维度下端到端选择质量断言。

## ☑️ 计划

- [ ] 论文中的真实数据基准（UCI / ADNI）
- [ ] 超高维 `p` 下的 GPU 批量 LOO 评估
- [ ] 上层求解器热启动多重启
- [ ] 学习到的可加分量的交互式可视化

## 🤝 致谢

REINFORCE 风格的上层机制借鉴自 [Probabilistic Bilevel Coreset Selection](https://github.com/x-zho14/Probabilistic-Bilevel-Coreset-Selection)（PBCS）的工程设计；半监督核模型（LapRLS/LapSVM）源自本仓库的初始版本。

## 📖 引用

如果 S2MAM 对您的研究有帮助，请引用：

```bibtex
@misc{zhang2026s2mamsemisupervisedmetaadditive,
      title={S2MAM: Semi-supervised Meta Additive Model for Robust Estimation and Variable Selection},
      author={Xuelin Zhang and Hong Chen and Yingjie Wang and Tieliang Gong and Bin Gu},
      year={2026},
      eprint={2604.19072},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2604.19072},
}
```

## 📄 许可证

本项目基于 [MIT License](LICENSE) 发布。
