
这段代码定义了一个名为 `Normalize` 的 PyTorch 模块，它实现了一个灵活的归一化和去归一化（反归一化）层，可以在深度学习模型中用于数据标准化和恢复原始数据。这个层特别适用于需要对数据进行预处理和后处理的场景，如图像处理或序列数据建模。以下是对这个类及其方法的详细说明：

### 类 Normalize
- **功能**：
  - 提供了数据的归一化和去归一化操作。
  - 支持有条件的加权变换，可以选择性地包含可学习的仿射（affine）参数，即比例（scale）和偏移（shift）。
  - 允许通过 `subtract_last` 选项从每个样本中减去最后一个特征，以处理可能的异常值或偏差。
  - 可以通过 `non_norm` 选项禁用归一化和去归一化的行为。

### 参数
- **num_features**：特征或通道数。
- **eps**：为数值稳定性添加的小值。
- **affine**：是否包含可学习的仿射变换参数。
- **subtract_last**：是否从每个样本中减去最后一个特征。
- **non_norm**：是否禁用归一化和去归一化操作。

### 方法
- **_init_params**：
  - 初始化仿射变换参数（如果 `affine=True`），包括权重和偏置。
  
- **_get_statistics**：
  - 计算输入 `x` 的均值和标准差。如果启用 `subtract_last`，则记录最后一个特征值；否则，计算所有特征的均值和标准差。
  
- **_normalize**：
  - 执行归一化操作：将输入 `x` 减去其均值并除以其标准差，以此来标准化数据。如果有仿射变换，还会应用这些变换。
  
- **_denormalize**：
  - 执行去归一化操作：将归一化的数据恢复到原始的尺度和位置。如果有仿射变换，会先撤销这些变换。

### 使用方法
- 在模型的前向传播中，可以根据需要调用 `forward` 方法，并指定 `mode` 为 `'norm'` 进行归一化或 `'denorm'` 进行去归一化。

这个类提供了一个灵活而强大的工具，用于控制深度学习模型中数据的尺度和分布，这在很多应用场景中都非常重要，尤其是在数据预处理和网络训练的不同阶段需要不同数据处理策略的情况下。
