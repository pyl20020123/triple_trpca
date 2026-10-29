# TriTD–Half TRPCA 深度展开 V2

这是规格 V2.0 的完整、可复现实验工程。网络把 `K` 次可解释更新展开为
`Half → ProxMALS(A → B → C)`，输入只有受污染张量 `Y`，输出低 triple 秩估计 `L`
和稀疏异常估计 `S`。每层学习非负的 `lambda`、`eta` 和 `eta_s`；Half 映射在活动分支
反向传播，在不连续阈值处选择零解。

本项目研究“已知低 triple 秩上界 + 人为随机稀疏污染”的受控问题，不是通用真实高光谱
混合噪声去除器，也不声称展开网络一定优于固定数值算法。

## 安装与测试

需要 Python 3.9+。建议使用独立虚拟环境：

```bash
python -m pip install -e '.[test]'
pytest -q
```

测试覆盖 NumPy/Torch 收缩与 MALS 对齐、Half 阈值和数值梯度、Stage 顺序、末层 λ 梯度、
批处理和可变尺寸、数据划分防泄漏、训练冒烟与 checkpoint 重载。CUDA 测试在无 GPU 时跳过。

## 快速运行

先运行 12/4/4 样本的小规模流程：

```bash
python train.py --config configs/smoke.yaml --out outputs/smoke_v2
python evaluate.py --config configs/smoke.yaml \
  --checkpoint outputs/smoke_v2/checkpoints/best.pt \
  --out outputs/smoke_test
python compare_baselines.py --config configs/smoke.yaml \
  --checkpoint outputs/smoke_v2/checkpoints/best.pt \
  --out outputs/smoke_comparison
```

正式配置：

```bash
python train.py --config configs/train_v2.yaml --out outputs/train_v2
python evaluate.py --config configs/eval_v2.yaml \
  --checkpoint outputs/train_v2/checkpoints/best.pt --out outputs/test_v2
```

从 `last.pt` 恢复训练：

```bash
python train.py --config configs/train_v2.yaml --out outputs/train_v2 \
  --resume outputs/train_v2/checkpoints/last.pt
```

## 输出

训练目录保存解析后的配置、环境版本、互斥 seed manifest、逐 epoch 的 train/val CSV、
逐层参数和 `best.pt`/`last.pt`。checkpoint 同时保存模型、优化器、调度器、配置和 Python/
NumPy/Torch 随机状态。评估目录保存逐样本指标、按条件汇总、逐 Stage 指标、运行时间/显存、
恢复面板及所有失败案例。

`compare_baselines.py` 输出普通 MALS（B0）、传统 V1 MALS→Half（B1）、固定参数的 K 层
Half→MALS（B2）和训练后的展开网络（B4）。sweep 数会落盘；NumPy CPU 与 Torch GPU 的耗时
不应直接解释为算法加速比。

## 数值约定

- `L = einsum('bipq,bsjq,bspk->bijk', A, B, C)`；轴顺序不可更改。
- Half 阈值为 `1.5 * beta**(2/3)`，阈值平局取零。
- MALS 使用 `torch.linalg.solve`，不构造矩阵逆。
- 初始化只依赖观测 shape/scale 与独立初始化 seed；绝不接收合成真值因子。
- Train/Val/Test seeds 分别从 1,000,000 / 2,000,000 / 3,000,000 开始，互不重叠。
- 训练监督损失不同于原始 P2 目标，展开网络不继承 V1 的目标单调性结论。

## 代码导航

- `src/numpy_core.py`：传统数值参考与基线
- `src/tritd_torch.py`：批量 TriTD、设计矩阵和可微 MALS
- `src/half_torch.py`：精确 Half 前向和活动分支梯度
- `src/stage.py` / `src/unfold_net.py`：单 Stage 与 K 层网络
- `src/synthetic_dataset.py`：确定性合成数据和 split manifest
- `train.py` / `evaluate.py` / `compare_baselines.py`：训练、冻结测试和公平对照
