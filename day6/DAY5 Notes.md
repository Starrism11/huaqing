# DAY5实习课程笔记

## 一.yolov8和yolo12的区别

### 1.yolov8

Overriding model.yaml nc=83 with nc=80

0                  -1  1       464  ultralytics.nn.modules.conv.Conv             [3, 16, 3, 2]

1                  -1  1      4672  ultralytics.nn.modules.conv.Conv             [16, 32, 3, 2]

2                  -1  1      7360  ultralytics.nn.modules.block.C2f             [32, 32, 1, True]

3                  -1  1     18560  ultralytics.nn.modules.conv.Conv             [32, 64, 3, 2]

4                  -1  2     49664  ultralytics.nn.modules.block.C2f             [64, 64, 2, True]

5                  -1  1     73984  ultralytics.nn.modules.conv.Conv             [64, 128, 3, 2]

6                  -1  2    197632  ultralytics.nn.modules.block.C2f             [128, 128, 2, True]

7                  -1  1    295424  ultralytics.nn.modules.conv.Conv             [128, 256, 3, 2]

8                  -1  1    460288  ultralytics.nn.modules.block.C2f             [256, 256, 1, True]

9                  -1  1    164608  ultralytics.nn.modules.block.SPPF            [256, 256, 5]

10                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']

11             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]

12                  -1  1    148224  ultralytics.nn.modules.block.C2f             [384, 128, 1]

13                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']

14             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]

15                  -1  1     37248  ultralytics.nn.modules.block.C2f             [192, 64, 1]

16                  -1  1     36992  ultralytics.nn.modules.conv.Conv             [64, 64, 3, 2]

17            [-1, 12]  1         0  ultralytics.nn.modules.conv.Concat           [1]

18                  -1  1    123648  ultralytics.nn.modules.block.C2f             [192, 128, 1]

19                  -1  1    147712  ultralytics.nn.modules.conv.Conv             [128, 128, 3, 2]

20             [-1, 9]  1         0  ultralytics.nn.modules.conv.Concat           [1]

21                  -1  1    493056  ultralytics.nn.modules.block.C2f             [384, 256, 1]

22        [15, 18, 21]  1    897664  ultralytics.nn.modules.head.Detect           [80, [64, 128, 256]]

YOLOv8n summary: 129 layers, 3,157,200 parameters, 3,157,184 gradients

### 2.yolo12

0                  -1  1       464  ultralytics.nn.modules.conv.Conv             [3, 16, 3, 2]

1                  -1  1      4672  ultralytics.nn.modules.conv.Conv             [16, 32, 3, 2]

2                  -1  1      6640  ultralytics.nn.modules.block.C3k2            [32, 64, 1, False, 0.25]

3                  -1  1     36992  ultralytics.nn.modules.conv.Conv             [64, 64, 3, 2]

4                  -1  1     26080  ultralytics.nn.modules.block.C3k2            [64, 128, 1, False, 0.25]

5                  -1  1    147712  ultralytics.nn.modules.conv.Conv             [128, 128, 3, 2]

6                  -1  2    180864  ultralytics.nn.modules.block.A2C2f           [128, 128, 2, True, 4]

7                  -1  1    295424  ultralytics.nn.modules.conv.Conv             [128, 256, 3, 2]

8                  -1  2    689408  ultralytics.nn.modules.block.A2C2f           [256, 256, 2, True, 1]

9                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']

10             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]

11                  -1  1     86912  ultralytics.nn.modules.block.A2C2f           [384, 128, 1, False, -1]

12                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']

13             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]

14                  -1  1     24000  ultralytics.nn.modules.block.A2C2f           [256, 64, 1, False, -1]

15                  -1  1     36992  ultralytics.nn.modules.conv.Conv             [64, 64, 3, 2]

16            [-1, 11]  1         0  ultralytics.nn.modules.conv.Concat           [1]

17                  -1  1     74624  ultralytics.nn.modules.block.A2C2f           [192, 128, 1, False, -1]

18                  -1  1    147712  ultralytics.nn.modules.conv.Conv             [128, 128, 3, 2]

19             [-1, 8]  1         0  ultralytics.nn.modules.conv.Concat           [1]

20                  -1  1    378880  ultralytics.nn.modules.block.C3k2            [384, 256, 1, True]

21        [14, 17, 20]  1    464912  ultralytics.nn.modules.head.Detect           [80, [64, 128, 256]]

YOLO12n summary: 272 layers, 2,602,288 parameters, 2,602,272 gradients

### **3.核心模块差异**

- **YOLOv8n**：
    - 使用 **`C2f`** 模块（跨阶段部分融合）作为基础块
    - 包含 **`SPPF`** 模块（空间金字塔池化快速版）
- **YOLO12n**：
    - 引入 **`C3k2`**（轻量级C3模块）和 **`A2C2f`**（带注意力机制的C2f变体）
    - 无 SPPF 模块

### **4.结构复杂度**

- **层数**：
    - YOLOv8n：129 层
    - YOLO12n：272 层（增加约 110%）
- **参数量**：
    - YOLOv8n：315.7 万参数
    - YOLO12n：260.2 万参数（减少约 17.6%）
- **梯度数**：
    - YOLOv8n：315.7 万梯度
    - YOLO12n：260.2 万梯度

### 5.**特征融合机制**

- **上采样路径**：
    - YOLOv8n 使用标准 **`C2f`** 处理特征融合
    - YOLO12n 使用混合模块：
        - 深层特征：**`A2C2f`**（带注意力机制）
        - 浅层特征：**`C3k2`**（轻量级设计）
- **连接方式**：
    - 两者均使用 **`Concat`** 进行特征拼接，但 YOLO12n 的拼接点更密集

### 6.**模块配置差异**

| **模块位置** | **YOLOv8n** | **YOLO12n** | **差异说明** |
| --- | --- | --- | --- |
| 第2层 | C2f [32,32,1] | C3k2 [32,64,1] | YOLO12n 输出通道加倍 |
| 第6层 | C2f [128,128] | A2C2f [128,128] | YOLO12n 引入注意力机制 |
| 第8层 | C2f [256,256] | A2C2f [256,256] | 同上 |
| 第9层 | SPPF [256,256] | 无 | YOLO12n 移除了金字塔池化模块 |
| 第20层 | C2f [384,256] | C3k2 [384,256] | YOLO12n 使用轻量模块替代 |

### 7.**设计特点**

- **YOLO12n 优化方向**：
    - **轻量化**：通过 **`C3k2`** 减少浅层计算量
    - **注意力机制**：在关键路径使用 **`A2C2f`** 增强特征选择能力
    - **结构精简**：移除 SPPF 模块降低计算复杂度
- **性能权衡**：
    - 层数增加但总参数量下降，适合资源受限场景
    - 注意力机制可能提升小目标检测精度

**总结:**

YOLO12n 是在 YOLOv8n 基础上的改进版本，核心创新点包括：

1. 引入注意力机制模块（A2C2f）强化特征提取
2. 采用轻量级 C3k2 模块降低计算开销
3. 移除 SPPF 简化结构
4. 通过模块重构实现参数量减少 17.6%，更适合移动端部署

两者均输出 80 类检测结果，但 YOLO12n 通过模块重组在保持检测能力的同时优化了计算效率。