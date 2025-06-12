# DAY3实习课程笔记

## 一、核心概念：激活函数

### 1. 激活函数的作用

- **引入非线性**：为神经网络引入非线性特性，使其能够学习和表示复杂的非线性映射关系，否则多层网络也只相当于一个线性模型。
- **决定神经元状态**：决定神经元是否被激活，以及它向下传递的信息强度。

### 2. 常见激活函数详解

### Sigmoid

- **公式**：f(x) = 1 / (1 + e^(-x))
- **特点**：输出在(0,1)之间，平滑连续，适合用作概率输出。
- **缺点**：计算复杂，存在严重的梯度消失问题，输出非零中心化。
- **应用**：主要用于二分类任务的输出层。

### Tanh

- **公式**：f(x) = (e^x - e^(-x)) / (e^x + e^(-x))
- **特点**：输出在(-1,1)之间，是零中心化的，收敛速度通常比Sigmoid快。
- **缺点**：计算复杂，仍存在梯度消失问题。
- **应用**：常用于RNN等需要中心对称输出的任务。

### ReLU (Rectified Linear Unit)

- **公式**：f(x) = max(0, x)
- **特点**：计算非常高效，收敛速度快，有效缓解梯度消失问题。
- **缺点**：存在"死亡ReLU"问题，即当输入小于0时，梯度为0，神经元可能永久不被激活。
- **应用**：目前是CNN和大多数深度网络中最常用的默认激活函数。

```python
self.relu = torch.nn.ReLU()
output = self.relu(input)
```

### Leaky ReLU

- **公式**：如果 x > 0, f(x) = x; 否则, f(x) = alpha * x (alpha通常是0.01)
- **特点**：为负数区域引入一个小的固定斜率，解决了“死亡ReLU”问题。
- **应用**：需要避免神经元失活的场景，如深度CNN。

### PReLU (Parametric ReLU)

- **公式**：与Leaky ReLU相同，但alpha是一个可学习的参数。
- **特点**：比Leaky ReLU更灵活，网络可以自主学习最佳的负斜率。
- **应用**：大型数据集和计算机视觉任务。

### ELU (Exponential Linear Unit)

- **公式**：如果 x > 0, f(x) = x; 否则, f(x) = alpha * (e^x - 1)
- **特点**：具有ReLU的所有优点，同时输出均值接近0，收敛更稳定。
- **缺点**：计算复杂度略高于ReLU。
- **应用**：适用于需要稳定收敛的深层网络。

### Softmax

- **公式**：sigma(x_i) = e^(x_i) / sum_j(e^(x_j))
- **特点**：将一组任意实数转换为一个概率分布，所有输出项之和为1。
- **应用**：专门用于多分类任务的输出层。

## 二、数据集准备与处理

### 1. 步骤一：划分数据集文件（物理移动）

此脚本将原始数据集按比例划分为训练集和验证集文件夹。

```python
# deal_with_datasets.py
import os
import shutil
from sklearn.model_selection import train_test_split
import random

# 设置参数
random.seed(42)
dataset_dir = r'/path/to/your/dataset/image_root' # 替换为你的数据集根目录
train_dir = os.path.join(dataset_dir, 'train')
val_dir = os.path.join(dataset_dir, 'val')
train_ratio = 0.7

# 创建目标文件夹
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# 遍历每个类别
for class_name in os.listdir(dataset_dir):
    if class_name in ["train", "val"]:
        continue
    
    class_path = os.path.join(dataset_dir, class_name)
    if not os.path.isdir(class_path):
        continue

    images = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    images_with_class = [os.path.join(class_name, img) for img in images]

    # 划分
    train_images, val_images = train_test_split(images_with_class, train_size=train_ratio, random_state=42)

    # 创建类别子目录
    os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)

    # 移动文件
    for img in train_images:
        shutil.move(os.path.join(dataset_dir, img), os.path.join(train_dir, img))
    for img in val_images:
        shutil.move(os.path.join(dataset_dir, img), os.path.join(val_dir, img))
    
    # 移动后删除原类别文件夹
    shutil.rmtree(class_path)
```

### 2. 步骤二：生成数据集索引文件 (.txt)

为训练集和验证集生成路径-标签索引文件，方便后续加载。

```python
# prepare.py
import os

def create_txt_file(root_dir, txt_filename):
    with open(txt_filename, 'w') as f:
        # 获取类别并排序，确保标签一致性
        categories = sorted(os.listdir(root_dir))
        for label, category in enumerate(categories):
            category_path = os.path.join(root_dir, category)
            if os.path.isdir(category_path):
                for img_name in os.listdir(category_path):
                    # 写入相对路径和标签
                    img_path = os.path.join(category, img_name)
                    f.write(f"{img_path} {label}\n")

# 使用方法
train_dir = r'/path/to/your/dataset/image_root/train'
val_dir = r'/path/to/your/dataset/image_root/val'
create_txt_file(train_dir, 'train.txt')
create_txt_file(val_dir, "val.txt")
```

### 3. 步骤三：加载与预处理数据

使用自定义的Dataset类和transforms来加载和处理数据。

### 数据预处理 (Transforms)

定义一系列操作，如调整尺寸、转换为张量、归一化，以满足模型输入要求。

```python
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

### 自定义数据集加载器

通过读取上一步生成的.txt文件来加载图像和标签。

```python
import os
from torch.utils import data
from PIL import Image

class ImageTxtDataset(data.Dataset):
    def __init__(self, txt_path, transform=None):
        self.transform = transform
        self.imgs_path = []
        self.labels = []
        # data_dir是train或val文件夹的路径
        self.data_dir = os.path.dirname(txt_path)

        with open(txt_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            img_rel_path, label = line.strip().split()
            # 拼接成完整路径
            self.imgs_path.append(os.path.join(self.data_dir, img_rel_path))
            self.labels.append(int(label))

    def __getitem__(self, index):
        img_path = self.imgs_path[index]
        label = self.labels[index]
        img = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        
        return img, label

    def __len__(self):
        return len(self.imgs_path)
```

## 三、常用神经网络模型简介

### 1. GoogLeNet

- **核心思想**：Inception模块。该模块在一个层内使用不同尺寸的卷积核（1x1, 3x3, 5x5）和池化操作，并将结果拼接起来，从而在不显著增加计算量的情况下提升网络的宽度和深度，增强特征提取能力。
- **特点**：性能优异，计算效率较高。

### 2. MobileNet_v2

- **核心思想**：Inverted Residual（倒置残差）和 Depth-wise Separable Convolution（深度可分离卷积）。先用1x1卷积升维，再进行深度卷积，最后用1x1卷积降维。
- **特点**：极为轻量，专为移动和嵌入式设备设计，在保持较高准确率的同时极大减少了计算量和参数量。

### 3. ResNet18

- **核心思想**：残差连接 (Residual Connection)。通过“快捷连接”将输入直接加到输出上，解决了深度网络中的梯度消失和网络退化问题，使得训练非常深的网络成为可能。
- **特点**：结构简单有效，易于训练，是许多计算机视觉任务的基准模型。可通过预训练模型进行迁移学习。

```python
# 使用预训练的ResNet18模型进行迁移学习
from torchvision.models import resnet18
import torch

# 加载预训练模型
model = resnet18(pretrained=True)

# 替换最后的全连接层以适应新的分类任务（例如10分类）
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 10)

# 将模型移至设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
```

### 4. MogaNet

- **核心思想**：这是一个泛指，可能指代结构相对简单、使用普通卷积层堆叠而成的自定义网络。
- **特点**：结构简洁，易于初学者理解和实现，可作为学习和改进的基础模型。

## 四、模型训练与测试

### 1. 训练流程

训练过程是在训练集上迭代，通过反向传播不断优化模型参数。

### 损失函数

衡量模型预测值与真实标签之间的差距。对于分类任务，交叉熵损失是标准选择。

```python
import torch.nn as nn
criterion = nn.CrossEntropyLoss()
```

### 优化器

根据损失函数计算出的梯度来更新模型的权重。Adam是一种高效且常用的自适应学习率优化器。

```python
import torch.optim as optim
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

### 2. 测试流程

在验证集或测试集上评估模型性能，此过程不进行梯度更新。

### 准确率计算

计算模型预测正确的样本占总样本的比例。

```python
correct = 0
total = 0
model.eval() # 将模型设置为评估模式
with torch.no_grad(): # 禁用梯度计算
    for images, labels in val_dataloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Accuracy on the test set: {accuracy} %')
```

### 3. 日志记录与可视化

使用TensorBoard等工具可以记录和可视化训练过程中的关键指标。

```python
from torch.utils.tensorboard import SummaryWriter

# 创建一个writer实例，日志将保存在 "logs/resnet18" 目录下
writer = SummaryWriter("logs/resnet18")

# 在训练循环中记录损失
# writer.add_scalar("Train Loss", train_loss, epoch)

# 在测试后记录准确率
# writer.add_scalar("Test Acc", test_acc, epoch)

# 还可以可视化图像
# writer.add_images("input_images", images, global_step)

writer.close()
```