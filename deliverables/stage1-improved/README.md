# S21：针对噪点残留与细节模糊的新增恢复流程

本版基于上传的 `tritd_trpca_stage1 (3).zip`，没有修改环境中的另一个 triple_trpca 项目。原有 Python 实验脚本保留；新增 `s21_detail_preserving.py`、`evaluate_s21.py`、`test_s21.py`，补齐依赖。无需训练或模型权重。

## 改动和原因

原 S19 使用低秩 L 重建整块时会改变原本正常的纹理；support_fill 虽保护了一部分观测，却仍依赖 GroupHalf 残差重新决定支持集，可能把纹理误作异常，或漏掉弱脉冲。附件 S20 表格也显示，多个 TriTD 变体弱于 local_median。单纯加 rank 或 guide_beta 并不能保证解决。

新增 nonlocal 方法先依据 RGB 中值偏差、局部边缘差异及最近邻颜色差异检测脉冲；再只用双方都可靠的 5×5 邻域匹配。匹配排除中心点，排除被检测为损坏的候选中心，在搜索半径 8 内聚合最多 12 个候选颜色。未标记像素逐值保留，避免整图低秩投影的模糊。无可靠候选时退回局部中值。整图按带 halo 的块处理，限制内存。

这是新增的图像修复方法，不是 TriTD–Half 的等价求解器，也不能作为原优化模型改进的理论证据。原算法仍可运行。

另外提供 median 方法：仅替换 RGB 中值偏差超过 0.2 的像素。它是明确的对照及可用回退，并非声称新算法。

## 实测（相同保存数组，不重新生成噪声）

| 数据 | 样例数 | 原输出平均 PSNR | 中值修复平均 PSNR | 新 nonlocal 平均 PSNR | 原 / 新平均 SSIM |
|---|---:|---:|---:|---:|---:|
| S13 开发数组 | 16 | 40.19 | 39.79 | 44.19 | 0.9822 / 0.9887 |
| S10 测试数组 | 12 | 22.65 | 41.63 | 30.39 | 0.6487 / 0.9247 |

开发集原输出为 estimated_X，测试集原输出为 recovered，二者不是同一版算法；请不要跨行比较原模型强弱。上表 PSNR 为逐图 dB 算术平均；verification 内同时保存 pooled PSNR，避免混淆。

开发集新方法 13/16 胜过原结果，最大退步 -6.89 dB；测试数组 12/12 胜过原结果，但明显弱于中值基线。开发集和测试数组均为 48×48 图块，并非完整图像测试。方法参数在查看测试指标前固定，之后未根据测试指标调整。根据已见测试结果建议中值方法，不构成新的独立测试结论。

适用建议：对附件 S13 的 random_rgb/additive_rgb 类稀疏污染可先尝试 nonlocal；对附件 S10 pixel 类污染优先比较 median。弱脉冲可能漏检，细小孤立纹理也可能被误判，不能保证所有图片提升。此流程不针对高斯稠密噪声。

## 运行

```bash
python -m pip install -r requirements.txt
# 新方法，输入为 [0,255] RGB 图片，内部归一化
python s21_detail_preserving.py --input noisy.png --output result.png --method nonlocal
# 中值对照/回退
python s21_detail_preserving.py --input noisy.png --output result_median.png --method median
python -m unittest test_s21 -v
```

同时输出 result_mask.png，白色为被修复位置。`--threshold` 可以手工设置，但应在独立开发集选择，不能逐测试图用干净真值选参数。

复现本次指标（包内附 28 个原始数组）：

```bash
python evaluate_s21.py --source verification_inputs/dev --out rerun/dev
python evaluate_s21.py --source verification_inputs/test --out rerun/test
```

verification/dev 和 verification/test 包含逐图 CSV、汇总 JSON、恢复数组及最大改善/最大退步图。评估中的 clean 仅用于计算指标、绘制结果，不进入 repair 函数。S19/S20 只有表格和面板、没有完整原图数组，因此未复跑该组；本次不声称在 S20 上取得提升。

4 项单元测试通过：平坦区域/阶跃边缘保持不变、孤立脉冲修复且未标记像素严格不变、分块/整块结果一致、非法输入校验。新增脚本通过语法检查。未运行所有历史实验；其 DIV2K 数据路径仍需用户本地数据。
