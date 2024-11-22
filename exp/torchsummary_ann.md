以下是逐行中文注释的代码：

```python
import torch
import torch.nn as nn
from torch.autograd import Variable

from collections import OrderedDict
import numpy as np


def summary(model, input_size, batch_size=-1, device="cuda"):
    # 定义一个函数，用于输出模型的结构信息、每层的输出形状以及参数数量

    def register_hook(module):
        # 定义一个函数，用于注册钩子函数，用于在前向传播时收集模块的输入和输出信息

        def hook(module, input, output):
            # 钩子函数，用于在前向传播时记录输入和输出的形状，以及参数数量
            class_name = str(module.__class__).split(".")[-1].split("'")[0]  # 获取模块的类名
            module_idx = len(summary)  # 当前模块的索引

            m_key = "%s-%i" % (class_name, module_idx + 1)  # 为当前模块生成唯一标识符
            summary[m_key] = OrderedDict()  # 使用有序字典记录信息

            # 记录输入形状
            if isinstance(input[0], (list, tuple)):
                summary[m_key]["input_shape"] = [
                    [-1] + list(i.size())[1:] for i in input[0]
                ]
                summary[m_key]["input_shape"][0] = batch_size  # 设置批量大小
            else:
                summary[m_key]["input_shape"] = list(input[0].size())
                summary[m_key]["input_shape"][0] = batch_size

            # 记录输出形状
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            # 记录参数数量
            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        # 为非容器模块注册钩子
        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
            and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    # 检查设备是否有效
    device = device.lower()
    assert device in [
        "cuda",
        "cpu",
    ], "输入设备无效，请指定 'cuda' 或 'cpu'"

    # 如果使用CUDA且可用，则选择CUDA浮点类型，否则选择CPU浮点类型
    if device == "cuda" and torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    # 如果输入大小是一个元组，则将其转换为列表
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # 为网络输入创建一个随机张量（批量大小为2，用于兼容BatchNorm层）
    x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size]

    # 初始化模型的摘要字典和钩子列表
    summary = OrderedDict()
    hooks = []

    # 为模型中的所有模块注册钩子
    model.apply(register_hook)

    # 执行一次前向传播以触发钩子并收集信息
    model(*x)

    # 移除所有钩子
    for h in hooks:
        h.remove()

    # 打印模型结构摘要头部
    print("----------------------------------------------------------------")
    line_new = "{:>20}  {:>25} {:>15}".format("Layer (type)", "Output Shape", "Param #")
    print(line_new)
    print("================================================================")
    
    # 初始化总参数数、总输出大小和可训练参数数
    total_params = 0
    total_output = 0
    trainable_params = 0

    # 遍历所有模块，打印每个模块的信息
    for layer in summary:
        line_new = "{:>20}  {:>25} {:>15}".format(
            layer,
            str(summary[layer]["output_shape"]),
            "{0:,}".format(summary[layer]["nb_params"]),
        )
        total_params += summary[layer]["nb_params"]  # 累计参数总数
        total_output += np.prod(summary[layer]["output_shape"])  # 累计输出总大小
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"]:
                trainable_params += summary[layer]["nb_params"]  # 累计可训练参数数
        print(line_new)

    # 计算模型总的内存占用（以MB为单位）
    total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # 前向和反向传播的大小（乘2）
    total_params_size = abs(total_params.numpy() * 4. / (1024 ** 2.))  # 参数大小

    # 打印模型参数摘要
    print("================================================================")
    print("Total params: {0:,}".format(total_params))  # 总参数数
    print("Trainable params: {0:,}".format(trainable_params))  # 可训练参数数
    print("Non-trainable params: {0:,}".format(total_params - trainable_params))  # 非可训练参数数
    print("----------------------------------------------------------------")
    print("Forward/backward pass size (MB): %0.2f" % total_output_size)  # 前向和反向传播内存占用
    print("Params size (MB): %0.2f" % total_params_size)  # 参数内存占用
    print("----------------------------------------------------------------")
    # 返回模型摘要
    # return summary
```

### 注释摘要：
1. **钩子函数** (`hook`)：
   - 收集每个模块的输入形状、输出形状和参数数量。
2. **注册钩子** (`register_hook`)：
   - 为所有非容器类型的模块（如卷积层、线性层）注册钩子。
3. **前向传播**：
   - 使用随机输入触发前向传播，记录模块的信息。
4. **内存估算**：
   - 估算参数大小和前向/反向传播的内存占用。
5. **打印摘要**：
   - 打印每一层的类型、输出形状和参数数量。

这个函数主要用于分析PyTorch模型的结构信息，可帮助理解模型的参数规模和内存占用情况。

