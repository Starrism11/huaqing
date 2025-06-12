# DAY2 实习课程笔记

## **1. 深度学习基础**

### (一) 概述

定义：深度学习是机器学习的一个分支，它通过构建包含多个层级的神经网络结构来学习和识别数据中的复杂模式与特征。
应用领域：深度学习被广泛应用于图像识别、语音识别、自然语言处理（NLP）、推荐系统等多个前沿领域。

### (二) 神经网络核心概念

- 神经元模型：模仿生物神经元，接收输入信号，通过加权求和与激活函数处理，最终产生输出。
- 激活函数：为网络引入非线性能力，使得网络可以学习更复杂的函数。
    - ReLU (Rectified Linear Unit): f(x) = max(0, x)，是CNN中最常用的激活函数，因其计算简单、收敛速度快。
    - Sigmoid 和 Tanh: 在深层网络中可能引发梯度消失问题，现在使用较少。
    - Leaky ReLU: 对ReLU的改进，允许微小的负值通过，以缓解“神经元死亡”问题。
- 损失函数 (Loss Function)：用于衡量模型预测结果与真实标签之间的差距。常见的有均方误差（MSE）和交叉熵损失（Cross-Entropy Loss）。
- 优化算法 (Optimizer)：用于根据损失函数调整网络中的权重参数，以最小化损失。常用算法包括梯度下降法及其变体，如随机梯度下降（SGD）、Adam等。

### (三) 训练关键问题

- 欠拟合 (Underfitting)：模型在训练集和测试集上的表现都很差，说明模型复杂度不够，未能学习到数据的基本模式。
- 过拟合 (Overfitting)：模型在训练集上表现优异，但在未见过的测试集上表现很差。这通常是由于模型过度学习了训练数据中的噪声和细节。

## **2. 卷积神经网络 (CNN)**

CNN是一类专门用于处理网格状数据（如图像）的深度学习模型。

### (一) CNN核心层级结构

一个典型的CNN网络结构通常遵循以下顺序：
输入 -> 卷积层 -> 激活函数(ReLU) -> 池化层 -> ... -> 全连接层 -> 输出

### (二) 层级详解

- 卷积层 (Convolutional Layer)
    - 作用：通过卷积核（filter）在输入数据上滑动，提取局部特征，如边缘、纹理等。
    - 核心参数：
        - in_channels: 输入特征图的通道数（如RGB图像为3）。
        - out_channels: 输出特征图的通道数，也等于卷积核的数量。
        - kernel_size: 卷积核的尺寸，如(3, 3)或3。
        - stride: 卷积核滑动的步长。
        - padding: 在输入数据的边缘进行填充，以控制输出特征图的尺寸。
    - 输出尺寸计算公式: Output_Size = floor((W - K + 2P) / S) + 1
    (W: 输入尺寸, K: 卷积核尺寸, P: 填充大小, S: 步长)
- 池化层 (Pooling Layer)
    - 作用：对特征图进行下采样，降低其尺寸，从而减少计算量和参数数量，同时有助于防止过拟合。
    - 常见类型：
        - 最大池化 (Max Pooling): 选取窗口内的最大值作为输出，保留最显著的特征。
        - 平均池化 (Average Pooling): 计算窗口内的平均值作为输出。
- 全连接层 (Fully Connected Layer)
    - 作用：在网络的末端，将前面卷积和池化层提取到的高级特征图进行展平（Flatten），然后连接到一个或多个全连接层，最终映射到输出空间以进行分类或回归。
- 其他重要层
    - 批归一化 (Batch Normalization): 在网络层之间对数据进行归一化处理，可以加速训练过程，提高模型的稳定性。
    - Dropout 层: 在训练期间，随机地“丢弃”（即置零）一部分神经元的输出，是一种非常有效的正则化手段，用于防止过拟合。

## **3. 模型训练与测试实战 (PyTorch)**

### (一) 数据集准备

使用`torchvision.datasets`加载CIFAR10等标准数据集，并用`torch.utils.data.DataLoader`创建数据加载器，以便在训练时进行批量处理、打乱等操作。

### (二) 模型构建

使用`torch.nn.Module`构建自定义的CNN模型。通过`nn.Sequential`可以方便地将卷积层、池化层、全连接层等有序地组合起来

**示例CNN结构:**

```python
class Chen(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(1024, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x
```

### (三) 模型训练

- 定义损失函数：例如，`nn.CrossEntropyLoss()` 用于多分类任务。
- 定义优化器：例如，`optim.SGD(model.parameters(), lr=0.01)`。
- 训练循环：
    - 遍历设定的训练轮次（epoch）。
    - 在每个epoch中，遍历数据加载器（DataLoader）。
    - 将数据输入模型，执行前向传播，得到预测输出。
    - 计算预测输出与真实标签之间的损失。
    - 执行 `optimizer.zero_grad()` 清空上一轮的梯度。
    - 执行 `loss.backward()` 进行反向传播，计算梯度。
    - 执行 `optimizer.step()` 更新模型权重。

### (四) 模型测试

- 将模型设置为评估模式：`model.eval()` 或使用 `with torch.no_grad():` 上下文管理器，以关闭梯度计算。
- 遍历测试数据集，计算总损失和准确率等评估指标。
- 准确率计算: `accuracy = (outputs.argmax(1) == targets).sum()`。

### (五) 训练过程可视化

使用`torch.utils.tensorboard.SummaryWriter`可以记录训练过程中的损失变化、准确率、图像等信息。

1. 启动TensorBoard: `tensorboard --logdir=your_log_directory`
2. 在训练循环中，使用`writer.add_scalar()`等方法记录数据。

## **4. Git 与 GitHub 版本控制教程**

### (一) 远程仓库创建

- 登录 GitHub，点击右上角“+” -> "New repository"。
- 填写仓库名，保持公开或设为私有。**关键：不要勾选初始化README文件的选项**。
- 点击 "Create repository"。

### (二) 本地项目初始化

- 在你的本地项目文件夹内，右键选择 "Git Bash Here"。
- 运行 `git init` 命令，初始化本地Git仓库。

### (三) 连接远程仓库并推送

- 添加远程仓库地址:
`git remote add origin https://github.com/你的用户名/你的仓库名.git`
- 将本地主分支推送到远程仓库:
`git push -u origin master`
(注意：现在新的GitHub仓库默认分支可能是 `main`，命令需相应改为 `git push -u origin main`)

### (四) 日常更新与提交

- 修改代码后，将更改添加到暂存区:
`git add .` (添加所有修改过的文件)
- 提交更改到本地仓库，并附上描述信息:
`git commit -m "你的更新说明"`
- 将本地提交推送到远程仓库:
`git push`