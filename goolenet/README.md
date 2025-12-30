# 数据集说明

默认情况下，`main.py` 期望在仓库根目录下找到 `MESSDIOR` 目录（等同于 `--root` 的默认值），结构如下：

```
MESSDIOR/
├── images/                  # 原始眼底图像，按 Base11、Base12 等子目录组织
└── annotation of images/    # Excel 文件（Annotation*.xls/Annotation*.xlsx）记录标签
```

程序会遍历 Excel 表格，将 `Image name` 与图像文件名匹配，并读取 `Retinopathy grade` 与 `Risk of macular edema` 作为两个目标字段。通过 `--target-field` 参数可以在 `retinopathy` 和 `macular_edema` 之间切换训练目标。

如果数据位于其他位置，可用 `--root` 指向新的根路径；`RetinaDataset` 构造函数会自动读取该目录下的 `images` 与 `annotation of images`。

调试相关命令：

- `--debug-samples`：打印前十个样本及其标签，不做训练。
- `--dump-samples`（可配合 `--dump-dir`）：将归一化后的图像样本保存到磁盘，默认保存在 `preprocessed_samples/`。输出采用验证/推理时使用的确定性 resize+normalize 流程，方便肉眼检查处理效果。

如需可视化训练曲线或统计训练/验证精度，可查看 `training_plots/training_progress.png` 以及程序打印的 `history` 信息。