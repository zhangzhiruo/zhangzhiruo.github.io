import pandas as pd
import numpy as np
import torch
from torch.utils.data import random_split
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from transformers import AutoTokenizer


VERSION = 7.0
from build_model_v7_0 import MultimodalClassifier as Model


TRAIN_NUMWORKS = 4
TRAIN_BATCHSIZE = 4
VAL_NUMWORKS = TRAIN_NUMWORKS
VAL_BATCHSIZE = TRAIN_BATCHSIZE



data_path = '/home/hyyjs/yangq/open_data_competation/dataset/train_dataset/train_final.csv'
model_path = '/home/hyyjs/yangq/open_data_competation/model'
image_path = '/home/hyyjs/yangq/open_data_competation/dataset/train_dataset/images'
test_path = '/home/hyyjs/yangq/open_data_competation/dataset/test_dataset/test_final.csv'
result_path = f'/home/hyyjs/yangq/open_data_competation/dataset/test_dataset/test_final_V{VERSION}.csv'


tokenizer = AutoTokenizer.from_pretrained(model_path)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])



from build_dataset import MultiModalDataset
dataset = MultiModalDataset(csv_file=data_path,img_dir=image_path,tokenizer=tokenizer,transform=transform, version=VERSION)

# 定义训练集和评估集的长度（80% 训练，20% 评估）
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

from utils import custom_collate_fn, EarlyStopping, plot_training_curves
# 使用 random_split 进行划分
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCHSIZE, shuffle=True, num_workers=TRAIN_NUMWORKS, collate_fn=custom_collate_fn)
val_loader = DataLoader(val_dataset, batch_size=VAL_BATCHSIZE, shuffle=False, num_workers=VAL_NUMWORKS, collate_fn=custom_collate_fn)


model = Model(num_classes=2)
# 定义优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
early_stopping = EarlyStopping(patience=10, delta=0.0001)

# 定义分类损失函数
criterion = nn.CrossEntropyLoss()

from trainer import Trainer
trainer = Trainer(
    model=model,
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    optimizer=optimizer,
    criterion=criterion,
    num_epochs=100,
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    enable_visualization=False,
    is_jupyter=False,
    scheduler=scheduler,
    early_stopping=early_stopping,
    version=VERSION
    )
trainer.train()
# 打印验证的信息
trainer.print_indicator(False, True)
# 分别绘制训练和验证的结果
trainer.plot_indicator(True, True)

# 验证模型
test_dataset = MultiModalDataset(csv_file=test_path,img_dir=image_path,tokenizer=tokenizer,transform=transform)
test_loader = DataLoader(test_dataset, batch_size=24, shuffle=False, num_workers=16, collate_fn=custom_collate_fn)
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.eval()
all_preds = []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        images = batch['images'].to(device)
        outputs = model(input_ids, images)
        _, preds = torch.max(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())

predDF = pd.read_csv(test_path)
predDF['target'] = all_preds
predDF.to_csv(result_path)