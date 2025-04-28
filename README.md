
## 主要文件说明

### 1. 模型相关文件
<mcfile name="model.py" path="d:\VsCode\Pytorch\Source_Code\src\model.py"></mcfile>
- 定义了一个基于Sequential的CNN模型
- 包含3个卷积层和2个全连接层
- 输入尺寸: 3x32x32 (RGB图像)
- 输出尺寸: 10 (分类任务)

### 2. 训练脚本
<mcfile name="train.py" path="d:\VsCode\Pytorch\Source_Code\src\train.py"></mcfile>
- 完整训练流程
- 使用CIFAR10数据集
- 包含训练/测试循环
- 支持TensorBoard日志记录

### 3. GPU训练版本
<mcfile name="train_gpu_2.py" path="d:\VsCode\Pytorch\Source_Code\src\train_gpu_2.py"></mcfile>
- 改进的GPU训练脚本
- 使用`device`统一管理计算设备
- 添加了训练时间统计

### 4. 实用工具
<mcfile name="nn.Sequential.py" path="d:\VsCode\Pytorch\Source_Code\src\nn.Sequential.py"></mcfile>
- 演示了Sequential构建模型的方法
- 包含TensorBoard模型可视化

## 使用建议

1. 基础训练: 使用`train.py`进行CPU训练
2. GPU加速: 使用`train_gpu_2.py`进行GPU训练
3. 模型测试: 使用`test.py`加载保存的模型进行预测
4. 预训练模型: 参考`model_pretrained.py`处理迁移学习

## 数据流
```mermaid
graph LR
A[原始数据] --> B[数据加载]
B --> C[模型训练]
C --> D[模型保存]
D --> E[模型测试]
