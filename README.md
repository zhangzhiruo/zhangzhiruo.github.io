# zhangzhiruo.github.io
Multi-Modal Fake News Detection (多模态真假新闻识别)
📌 项目简介
本项目旨在构建一个多模态（文本 + 图像）的真假新闻识别模型，通过融合文本和图像特征，提升对新闻真实性的判断准确率。项目使用 PyTorch 框架，结合预训练的 BERT 和 ResNet-18 模型进行特征提取，并设计了自定义的特征融合与分类模块。

🗂️ 文件结构
text
.
├── build_dataset.py          # 多模态数据集加载与预处理
├── build_model_v5_0.py      # 模型结构定义（文本/图像编码器、融合模块、分类器）
├── run.py                   # 主训练脚本
├── trainer.py               # 训练器类封装
├── test.py                  # 测试脚本
├── utils.py                 # 工具函数（collate_fn、早停、可视化等）
├── requirements.txt         # 依赖包列表（需自行补充）
└── README.md
🛠️ 环境依赖
建议使用 Python 3.8+，并安装以下依赖：

bash
pip install torch torchvision transformers pandas pillow scikit-learn matplotlib
🚀 快速开始
1. 数据准备
将数据集按如下格式组织：

text
dataset/
├── train.csv
├── val.csv
├── test.csv
└── images/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
CSV 文件中应包含以下列：

text: 新闻文本

images_list: 图像文件名，多个图像用 \t 分隔

label: 标签（0=假新闻，1=真新闻）

2. 训练模型
bash
python run.py \
  --train_csv path/to/train.csv \
  --img_dir path/to/images \
  --val_csv path/to/val.csv \
  --batch_size 16 \
  --epochs 50
3. 测试模型
bash
python test.py \
  --ckp path/to/best_model.pth \
  --data path/to/test_folder \
  --batch 8
🧠 模型架构
文本编码器（TextEncoderV5）
基于 BERT 提取文本特征

使用多头注意力增强表示

图像编码器（ImageEncoderV5）
使用 ResNet-18 提取图像特征

引入 SE（Squeeze-and-Excitation）模块增强通道注意力

特征融合模块（FeatureFusionV5）
动态加权融合文本与图像特征

使用全连接层进行降维与非线性变换

分类器（ImprovedClassifier）
两层全连接网络 + Dropout

输出二分类结果

📊 实验结果
训练过程中记录以下指标：

准确率（Accuracy）

F1-score

损失（Loss）

混淆矩阵

可通过 trainer.py 中的可视化工具绘制训练曲线。

📈 可视化示例
项目中支持以下可视化：

标签分布饼图

训练/验证指标曲线（Loss、Accuracy、F1）

混淆矩阵、ROC 曲线、PR 曲线（测试阶段）

🙌 致谢
本项目使用了以下开源模型：

BERT

ResNet-18

📜 许可证
本项目仅用于学术研究，请遵守数据使用许可和模型版权声明。
