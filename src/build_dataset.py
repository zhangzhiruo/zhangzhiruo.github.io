import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import numpy as np
from transformers import BertTokenizer
from transformers import AutoTokenizer


class MultiModalDataset(Dataset):
    def __init__(self, csv_file, img_dir, tokenizer, max_len=256, transform=None, version=None):
        """
        初始化数据集类。

        参数:
            csv_file (str): 数据集文件路径 (CSV 文件)
            img_dir (str): 图像文件夹路径
            tokenizer: 用于文本处理的预训练模型的 tokenizer
            max_len (int): 文本最大长度
            transform (callable, optional): 图像变换操作
        """
        self.data:pd.DataFrame = pd.read_csv(csv_file, encoding='utf-8')
        self.img_dir = img_dir
        self.tokenizer = tokenizer if float(version)>=7.0 else self.get_tokenizer()
        self.max_len = max_len
        self.transform = transform
        if 'label' not in self.data.columns:
            self.data['label'] = 0

        # 设置最大的图片张数
        self.max_pic_nums = max([len(row['images_list'].split('\t')) for idx, row in self.data.iterrows() if pd.notna(row['images_list'])])

    def get_tokenizer(self):
        """从版本7开始，序列化使用BERT序列化"""
        # tokenizer = BertTokenizer.from_pretrained('/home/hyyjs/yangq/open_data_competation/model')
        tokenizer = AutoTokenizer.from_pretrained('/home/hyyjs/yangq/open_data_competation/model2', use_fast=False)
        return tokenizer


    def __len__(self):
        """返回数据集中的样本数量。"""
        return len(self.data)

    def __getitem__(self, idx):
        """
        获取数据集中指定索引的数据项。

        参数:
            idx (int): 数据项的索引

        返回:
            dict: 包含输入 ID、注意力掩码、图像和标签的字典
        """
        # 获取数据项
        item = self.data.iloc[idx]
        text = item['text']
        label = torch.tensor(item['label'], dtype=torch.long)

        # 文本处理：tokenization + padding
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)

        # 图像处理
        images_list = item['images_list'].split('\t') if pd.notna(item['images_list']) else []
        images = []

        for img_name in images_list:
            img_path = os.path.join(self.img_dir, img_name)
            try:
                img = Image.open(img_path).convert('RGB')
            except OSError:
                # 如果打开失败，创建一个空白图像
                img = Image.fromarray(np.ones((224, 224, 3), dtype=np.uint8) * 255)

            # 应用图像变换（如果有）
            if self.transform:
                img = self.transform(img)
            images.append(img)

        # 如果有多个图像，拼接它们
        if len(images) > 1:
            images = torch.cat([img.unsqueeze(0) for img in images], dim=0)  # 按第一维度拼接多个图像
        elif len(images) == 1:
            images = images[0].unsqueeze(0)  # 确保输出是 [1, 3, 224, 224] 形状
        else:
            # 如果没有图像，则返回零图像
            images = torch.zeros(1, 3, 224, 224)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'images': images,
            'label': label
        }