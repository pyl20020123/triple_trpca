# TriTD–Half 彩色图像去噪（DIV2K）

这是从 TriTD–Half TRPCA V2 扩展出的 RGB 图像去噪工程。网络把 `K` 次可解释更新展开为
`Half → ProxMALS(A → B → C)`，输入只有受污染张量 `Y`，输出低 triple 秩估计 `L`
和稀疏异常估计 `S`。每层学习非负的 `lambda`、`eta` 和 `eta_s`；Half 映射在活动分支
反向传播，在不连续阈值处选择零解。

RGB 图像按 `(height,width,3)` 作为三阶张量。默认任务是彩色盐椒/脉冲噪声，因为 Half
正则对应稀疏异常；配置也能生成高斯或混合噪声，但高斯噪声是稠密的，不再严格符合原始
稀疏模型假设。自然图像也只近似满足低 triple 秩，因此必须用实际 DIV2K 测试 PSNR/SSIM，
不能从合成张量实验直接推断效果。

RGB 已归一化到 `[0,1]`，所以默认 `lambda_init=0.002`，明显小于合成张量的 0.1；训练时
应关注 `active_ratio`。如果所有 Stage 长期接近零，Half 活动分支及 λ 会几乎收不到梯度。

## 安装与测试

需要 Python 3.9+。建议使用独立虚拟环境：

```bash
python -m pip install -e '.[test]'
pytest -q
```

测试覆盖 NumPy/Torch 收缩与 MALS 对齐、Half 阈值和数值梯度、Stage 顺序、末层 λ 梯度、
批处理和可变尺寸、数据划分防泄漏、训练冒烟与 checkpoint 重载。CUDA 测试在无 GPU 时跳过。

## 准备 DIV2K

脚本可下载官方 HR 压缩包，并把官方 validation 图像按源图像划分成互斥的 val/test，之后
生成无损 RGB patch 和 manifest。默认每张源图最多保留 32 个 `64×64` patch，避免一次生成
过大的数据目录。

```bash
python prepare_div2k.py \
  --source data/div2k_raw \
  --out data/div2k_processed \
  --download \
  --patch-size 64 \
  --stride 64 \
  --max-patches-per-image 32 \
  --val-images 50
```

如果已经下载数据，去掉 `--download`，并确保源目录包含 `DIV2K_train_HR/` 和
`DIV2K_valid_HR/`。预处理输出包含 `patches/{train,val,test}` 和对应的三个 manifest；同一
源图绝不会同时进入 val 与 test。

## DIV2K 训练与评估

先用小样本配置检查环境和数据：

```bash
python train.py --config configs/div2k_smoke.yaml --out outputs/div2k_smoke
python evaluate.py --config configs/eval_div2k.yaml \
  --checkpoint outputs/div2k_smoke/checkpoints/best.pt \
  --out outputs/div2k_smoke_test
```

正式训练：

```bash
python train.py --config configs/train_div2k.yaml --out outputs/train_div2k
python evaluate.py --config configs/eval_div2k.yaml \
  --checkpoint outputs/train_div2k/checkpoints/best.pt \
  --out outputs/test_div2k
```

评价结果包含 RRE、污染支撑 Precision/Recall/F1、PSNR、SSIM、逐 Stage 指标、彩色恢复面板、
输入 PSNR/SSIM、PSNR 增益、推理耗时和失败样本。测试集不参与学习率、Stage 数或阈值选择。

如果目标是常见 AWGN，可使用 `configs/train_div2k_gaussian.yaml` 与
`configs/eval_div2k_gaussian.yaml`，内置 σ=15/25/50（配置中除以 255）。由于高斯噪声是稠密
噪声，Precision/Recall/F1 不具有稀疏支撑解释，主要看 PSNR、SSIM 与 `psnr_gain`；该实验
不能宣称仍是原 Half 稀疏近端模型的严格对应。

## 对单张彩色图片去噪

整图推理使用重叠 patch 并平均融合，避免把完整 2K 图像的计算图一次放入显存：

```bash
python denoise_image.py \
  --input noisy.png \
  --checkpoint outputs/train_div2k/checkpoints/best.pt \
  --output outputs/denoised.png \
  --tile-size 64 \
  --overlap 16
```

同时生成 `denoised_noise.npy`（原始残差数值）和 `denoised_noise.png`（加 0.5 后的可视化）。

## 合成张量兼容流程

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

原 V2 合成张量正式配置：

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
- `prepare_div2k.py`：DIV2K 下载、安全解压、RGB 切块和互斥 manifest
- `src/image_dataset.py`：RGB patch、增强和脉冲/高斯/混合噪声
- `denoise_image.py`：任意尺寸 RGB 图片的重叠分块去噪
- `train.py` / `evaluate.py` / `compare_baselines.py`：训练、冻结测试和公平对照
