## 📁 项目结构

text

```
根目录/
│
├── build_dataset.py          # 多模态数据集定义与加载
├── build_model_v5_0.py       # 模型结构定义
├── run.py                    # 主训练脚本
├── trainer.py                # 训练器封装
├── test.py                   # 测试脚本
├── utils.py                  # 工具函数（collate_fn、早停策略等）
├── save/                     # 模型保存目录
│   └── best_model_v5.pth     # 预训练模型权重
├── dataset/                  # 数据集目录
│   └── test_dataset/         # 测试集数据与图像
└── figs/                     # 训练曲线与结果可视化保存目录
```

------

## 🚀 快速开始

### 1. 训练模型

使用以下命令启动训练：

bash

```
python run.py
```

可通过参数调整批次大小、学习率等，具体请参考 `run.py` 中的参数设置。

### 2. 测试模型

使用以下命令对测试集进行预测：

bash

```
python test.py --ckp save/best_model_v5.pth --data dataset/test_dataset --batch 4 --numwork 4
```

预测结果将保存在测试集的 CSV 文件中，新增一列 `target` 表示预测标签。

------

## 🧠 模型架构

本项目采用多模态融合架构，主要包括以下模块：

- **文本编码器**：基于 BERT 提取文本特征

- **图像编码器**：基于 ResNet18 + SE 注意力机制提取图像特征

- **特征融合模块**：自适应加权融合文本与图像特征

- **分类器**：全连接层输出最终分类结果

  ![image-20250831222555631](C:\Users\张芷若\AppData\Roaming\Typora\typora-user-images\image-20250831222555631.png)

![image-20250831222531701](C:\Users\张芷若\AppData\Roaming\Typora\typora-user-images\image-20250831222531701.png)

------

## 📊 数据预处理

### 文本处理

- 使用 BERT Tokenizer 进行分词与编码
- 最大长度设置为 256，不足则填充，过长则截断

### 图像处理

- 统一缩放至 224×224
- 转换为 Tensor 并归一化
- 支持多图输入，自动填充或补零

### 标签分布

真假新闻占比

![image-20250831221918813](C:\Users\张芷若\AppData\Roaming\Typora\typora-user-images\image-20250831221918813.png)

------

## 📈 训练与验证曲线

训练过程中记录 Loss、Accuracy 和 F1 Score，并可视化如下：

![image-20250831222044261](C:\Users\张芷若\AppData\Roaming\Typora\typora-user-images\image-20250831222044261.png)


![image-20250831222112509](C:\Users\张芷若\AppData\Roaming\Typora\typora-user-images\image-20250831222112509.png)


------

## ✅ 测试结果

测试脚本支持输出以下评估结果：

- 混淆矩阵
- ROC 曲线
- PR 曲线

可通过取消注释 `test.py` 中对应函数调用来生成图像。

------

## 🧩 可扩展性

- 可替换其他预训练文本模型（如 RoBERTa、ERNIE）
- 可替换其他图像编码器（如 ViT、EfficientNet）
- 支持自定义融合策略（如 Cross-Attention、Co-Attention）

------

## 📄 许可证

本项目仅用于学术研究，如需商用请联系作者。