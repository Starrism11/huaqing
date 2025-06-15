# DAY4实习课程笔记

## 一、自定义数据集的准备与加载

### 1. 数据集划分

首先，将原始图像数据集划分为训练集和验证集。通常，我们会创建一个脚本（例如 `deal_with_datasets.py`）来自动化这个过程，常见比例为 70% 训练，30% 验证。

- **核心思路**：遍历每个类别文件夹，使用 `sklearn.model_selection.train_test_split` 对其中的图片列表进行划分，然后将图片移动到新建的 `train` 和 `val` 目录下的对应类别子目录中。

### 2. 生成图像路径文件

为了方便后续 `Dataset` 类读取，我们会为训练集和验证集分别生成一个 `.txt` 文件（如 `train.txt`, `val.txt`）。文件中每一行包含一张图片的绝对路径和一个整数标签。

- **核心思路**：遍历 `train` 和 `val` 目录，为每个类别分配一个数字索引（label），然后将“图片路径 标签”的格式写入 `txt` 文件。

### 3. 自定义数据集类 (ImageTxtDataset)

PyTorch 允许我们通过继承 `torch.utils.data.Dataset` 来创建自定义的数据集加载器。

- **功能**：
    1. `__init__`：初始化函数，接收 `txt` 文件路径和数据预处理（transform）流程。它会读取 `txt` 文件，将图像路径和标签分别存入两个列表中。
    2. `__getitem__`：根据索引（index）获取单个数据样本。它会打开对应路径的图像，进行 RGB 转换，应用预处理（transform），最后返回处理后的图像张量和标签。
    3. `__len__`：返回数据集中样本的总数。

### 4.数据预处理与加载

在加载数据时，需要对图像进行一系列预处理和增强操作，以满足模型输入要求并提升模型泛化能力。

- **常用预处理流程 (transforms.Compose)**：
    1. `transforms.Resize(224)`：将图像尺寸统一调整为 224x224，以适应 AlexNet 或 ViT 的输入。
    2. `transforms.RandomHorizontalFlip()`：随机水平翻转，一种常见的数据增强手段。
    3. `transforms.ToTensor()`：将 PIL.Image 对象或 numpy 数组转换为张量。
    4. `transforms.Normalize(...)`：使用 ImageNet 的均值和标准差进行归一化，有助于模型更快收敛。
- **数据加载器 (DataLoader)**：将 `Dataset` 封装成一个迭代器，实现批量（batch）加载、打乱数据（shuffle）等功能。

## 二、核心模型架构解析

### 1. 经典卷积网络：AlexNet

AlexNet 是一个经典的深度卷积神经网络（CNN），它证明了深层 CNN 在复杂图像分类任务上的有效性。

- **核心结构**：
    - **卷积层**：包含 5 个卷积层，用于提取图像的局部特征。
    - **池化层**：使用 `MaxPool2d` 进行下采样，减小特征图尺寸，提取关键特征。
    - **全连接层**：包含 3 个全连接层，用于整合特征并进行最终分类。
    - **输出层**：最后一层的输出节点数等于任务的类别数（例如，10 类分类任务则输出为 10）。

### 2. 视觉新范式：Vision Transformer (ViT)

ViT 将自然语言处理领域的 Transformer 架构成功应用于计算机视觉任务，其核心思想是**将图像视为一个序列**。

- **核心架构**：
    1. **Patch Embedding（图像切块与线性嵌入）**：
        - 将输入图像（如 224x224）分割成一系列固定大小的小块（patch），例如 16x16。这样一张图就被分成了 (224/16) * (224/16) = 196 个 patches。
        - 将每个 patch 展平（flatten）成一个向量，并通过一个线性层将其映射（嵌入）到指定的维度（`dim`）。这一步将图像数据转换成了 Transformer 能处理的 token 序列。
        - `einops` 库的 `Rearrange` 函数可以非常方便地实现这个操作。
    2. **Position Embedding & CLS Token（位置编码与分类令牌）**：
        - **Position Embedding**：由于 Transformer 本身不感知序列的顺序，需要为每个 patch token 添加一个可学习的位置编码，以保留图像的空间信息。
        - **CLS Token**：在序列的开头加入一个特殊的可学习的 `[CLS]` 令牌。在模型处理完整个序列后，这个令牌对应的输出将作为整个图像的聚合特征，用于最终的分类。
    3. **Transformer Encoder（编码器）**：
        - 由多个相同的层堆叠而成，是 ViT 的核心。
        - **每个编码器层包含**：
            - **多头自注意力 (Multi-Head Self-Attention)**：捕捉序列中所有 token 之间的全局依赖关系。
            - **前馈网络 (Feed-Forward Network)**：一个简单的多层感知机（MLP）。
            - **残差连接 & 层归一化 (LayerNorm)**：在每个模块前后都使用残差连接和 LayerNorm，以稳定训练过程。
    4. **MLP Head（分类头）**：
        - 将 Transformer Encoder 输出的 `[CLS]` 令牌对应的向量输入到一个标准的全连接层（MLP），最终输出每个类别的得分。

## 三、模型训练与常见问题

### 1. 核心训练流程

一个标准的 PyTorch 训练循环包含以下步骤：

1. **设置设备**：检查 GPU (CUDA) 是否可用，并定义 `device`。
`device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`
2. **迁移模型和数据**：将模型 (`model.to(device)`) 和每个批次的数据 (`inputs.to(device)`, `labels.to(device)`) 移动到指定设备。
3. **迭代训练**：
    - 将模型设置为训练模式 (`model.train()`)。
    - 遍历 `DataLoader` 获取数据批次。
    - 清空过往梯度 (`optimizer.zero_grad()`)。
    - 前向传播，得到模型输出 (`outputs = model(inputs)`)。
    - 计算损失 (`loss = criterion(outputs, labels)`)。
    - 反向传播，计算梯度 (`loss.backward()`)。
    - 更新模型参数 (`optimizer.step()`)。
4. **评估与保存**：
    - 每个 epoch 结束后，在验证集上评估模型性能（准确率、损失）。
    - 使用 `torch.no_grad()` 禁用梯度计算以节省资源。
    - 使用 TensorBoard 记录损失和准确率，实现训练过程可视化。
    - 定期保存模型权重（`.pth` 文件）。

### 2.常见问题排查

问题：模型输出维度与标签不匹配
可能原因：num_classes 参数设置错误。
解决方法：检查并修改模型定义中的 num_classes，使其与数据集的类别总数一致。

问题：图片路径错误 / 找不到文件
可能原因：.txt 文件中的路径是相对路径，或根目录不正确。
解决方法：确保 .txt 文件中存储的是绝对路径，或在 Dataset 类中正确拼接路径。

问题：GPU 不可用
可能原因：CUDA 环境未正确安装或 PyTorch 版本不匹配。
解决方法：确认 CUDA 和 cuDNN 已安装，并安装与之对应的 PyTorch 版本。

问题：ViT 图像与 patch 尺寸不匹配
可能原因：图像尺寸不能被 patch 尺寸整除。
解决方法：调整 transforms.Resize 中的 image_size 或模型定义中的 patch_size，确保二者可以整除。

## 四、总结与学习重点

- **数据处理是基础**：掌握如何使用 `sklearn`、`os` 等库来处理原始文件，以及如何编写自定义的 `Dataset` 类，是进行任何深度学习任务的第一步。
- **理解模型架构**：深入理解 CNN (以 AlexNet 为例) 和 Transformer (以 ViT 为例) 的核心组件和工作原理，是解决问题和模型创新的关键。
- **掌握训练范式**：熟悉 PyTorch 的标准训练、验证循环，包括如何使用 GPU、如何进行可视化、如何保存模型。
- **代码工具库**：善用 `einops` 等库可以极大简化张量操作，使代码更简洁易读。