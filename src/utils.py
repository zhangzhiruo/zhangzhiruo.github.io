import torch

def custom_collate_fn(batch):
    """
    自定义 collate 函数，用于处理不同数量的图像。

    参数:
        batch (list): 一批数据

    返回:
        dict: 包含输入 ID、注意力掩码、图像和标签的字典
    """
    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    images = [item['images'] for item in batch]
    labels = [item['label'] for item in batch]

    # 计算当前批次中图像的最大数量
    max_num_images = max([img.shape[0] for img in images])

    # 对图像进行零填充
    padded_images = []
    for img in images:
        num_padding = max_num_images - img.shape[0]
        if num_padding > 0:
            padding = torch.zeros(num_padding, 3, 224, 224)
            padded_img = torch.cat([img, padding], dim=0)
        else:
            padded_img = img
        padded_images.append(padded_img)

    # 堆叠成张量
    input_ids = torch.stack(input_ids, dim=0)
    attention_mask = torch.stack(attention_mask, dim=0)
    images = torch.stack(padded_images, dim=0)
    labels = torch.stack(labels, dim=0)

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'images': images,
        'label': labels
    }



class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        """
        :param patience: 连续多少个epoch没有改进后停止训练
        :param delta: 需要的最小改进（用来避免过早停止）
        """
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, score):
        """
        判断是否早停
        :param score: 当前epoch的评价分数（比如验证集准确率或损失值）
        :return: 是否需要停止训练 (True/False)
        """
        if self.best_score is None:
            self.best_score = score  # 初始化最佳分数
            self.counter = 0  # 重置计数器
            return False  # 继续训练
        elif score < self.best_score + self.delta:
            # 连续未改进的计数加一
            self.counter += 1
            if self.counter >= self.patience and self.best_score>0.0:
                self.early_stop = True
                return True  # 停止训练
        else:
            # 当前分数改进，重置计数器和最佳分数
            self.best_score = score
            self.counter = 0
        return False  # 继续训练

import matplotlib.pyplot as plt

def plot_training_curves(f1_list, loss_list):
    epochs = list(range(1, len(f1_list) + 1))

    # 创建两个子图
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # 绘制 F1 分数曲线
    color = 'tab:blue'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('F1 Score', color=color)
    ax1.plot(epochs, f1_list, color=color, label='F1 Score')
    ax1.tick_params(axis='y', labelcolor=color)

    # 创建共享 x 轴的第二个 y 轴，绘制 Loss 曲线
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Loss', color=color)
    ax2.plot(epochs, loss_list, color=color, linestyle='--', label='Loss')
    ax2.tick_params(axis='y', labelcolor=color)

    # 标题和图例
    fig.tight_layout()
    plt.title('Training Progress')
    fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.85))
    plt.grid(True)
    plt.show()
