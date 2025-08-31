import plotly.graph_objects as go
import torch
from IPython.display import display, clear_output
from plotly.subplots import make_subplots
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from tqdm import tqdm

import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.pyplot as plt

class IndicatorSingleEpoch():
    def __init__(self) -> None:
        self.f1 = []
        self.cm = []
        self.acc = []
        self.loss = []

    def update(self, f1=None, cm=None, acc=None, loss=None):
        if f1: self.f1.append(f1)
        if cm is not None: self.cm.append(cm)
        if acc: self.acc.append(acc)
        if loss: self.loss.append(loss)

    def show_last_info(self):
        return (f'Loss={self.loss[-1]:.4f}, F1={self.f1[-1]:.4f}, Acc={self.acc[-1]:.4f}')

    def show(self):
        for i in range(len(self.f1)):
            print("=======================")
            print(f"Epoch {i+1}/len(self.f1)")
            print(f'Loss={self.loss[i]}, F1={self.f1[i]}, Acc={self.acc[i]}')
            print(self.cm[i])

    def plot(self, name):
        epochs = range(1, len(self.f1) + 1)

        fig, ax1 = plt.subplots(figsize=(10, 6))

        # F1 Score
        ax1.plot(epochs, self.f1, 'bo-', label='F1 Score')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('F1 Score', color='b')
        ax1.tick_params(axis='y', labelcolor='b')

        # Accuracy
        ax2 = ax1.twinx()
        ax2.plot(epochs, self.acc, 'go-', label='Accuracy')
        ax2.set_ylabel('Accuracy', color='g')
        ax2.tick_params(axis='y', labelcolor='g')

        # Loss
        ax3 = ax1.twinx()
        ax3.spines['right'].set_position(('outward', 60))
        ax3.plot(epochs, self.loss, 'ro-', label='Loss')
        ax3.set_ylabel('Loss', color='r')
        ax3.tick_params(axis='y', labelcolor='r')

        # 图例
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        lines_3, labels_3 = ax3.get_legend_handles_labels()

        ax1.legend(lines_1 + lines_2 + lines_3, labels_1 + labels_2 + labels_3, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3)

        plt.title('Indicator per Epoch')
        plt.tight_layout()
        plt.savefig(f'/home/hyyjs/yangq/open_data_competation/figs/{name}.jpg')


   


class Trainer:
    def __init__(self, model, train_dataloader, val_dataloader, optimizer,
                 criterion, num_epochs, device,
                 enable_visualization=True, is_jupyter=True, scheduler=None, early_stopping=None,
                 version:str=None):
        """
        初始化训练器类
        :param model: 训练的模型
        :param train_dataloader: 训练数据加载器
        :param val_dataloader: 验证数据加载器
        :param optimizer: 优化器
        :param criterion: 损失函数
        :param num_epochs: 训练的轮数
        :param device: 设备（cpu/gpu）
        :param enable_visualization: 是否启用可视化 (默认启用)
        :param is_jupyter: 是否是 Jupyter 环境
        :param scheduler: 学习率调度器 (可选)
        """
        self.device = device
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.criterion = criterion
        self.num_epochs = num_epochs
        self.scheduler = scheduler  # 新增的scheduler参数
        self.early_stopping = early_stopping  # 新增早停
        self.best_f1 = 0
        self.train_indicator = IndicatorSingleEpoch()
        self.val_indicator = IndicatorSingleEpoch()
        self.enable_visualization = enable_visualization
        self.is_jupyter = is_jupyter
        self.version = version
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)

    def eval_step(self):
        self.model.eval()
        eval_all_labels, eval_all_preds, eval_total_loss = [], [], 0.0
        with torch.no_grad():
            for batch in self.val_dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                images = batch['images'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(input_ids, images, attention_mask)
                loss = self.criterion(outputs, labels)
                eval_total_loss += loss.item()

                _, preds = torch.max(outputs, dim=1)
                eval_all_labels.extend(labels.cpu().numpy())
                eval_all_preds.extend(preds.cpu().numpy())

        avg_loss = eval_total_loss / len(self.val_dataloader)
        f1 = f1_score(eval_all_labels, eval_all_preds, average='weighted')
        acc = accuracy_score(eval_all_labels, eval_all_preds)
        cm = confusion_matrix(eval_all_labels, eval_all_preds)
        self.val_indicator.update(f1, cm, acc, avg_loss)
    
    def train_step(self):
        pbar = tqdm(total=len(self.train_dataloader), dynamic_ncols=True)


        self.model.train()
        train_all_labels, train_all_preds, train_total_loss = [], [], 0.0
        for i, batch in enumerate(self.train_dataloader):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            images = batch['images'].to(self.device)
            labels = batch['label'].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(input_ids, images, attention_mask)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            _, preds = torch.max(outputs, dim=1)
            train_total_loss += loss.item()
            train_all_labels.extend(labels.cpu().numpy())
            train_all_preds.extend(preds.cpu().numpy())

            pbar.set_postfix({'Current Batch {i}': f'{loss.item():.4f}'})
            pbar.update(1)

        avg_loss = train_total_loss / len(self.train_dataloader)
        f1 = f1_score(train_all_labels, train_all_preds, average='weighted')
        acc = accuracy_score(train_all_labels, train_all_preds)
        cm = confusion_matrix(train_all_labels, train_all_preds)
        
        self.train_indicator.update(f1, cm, acc, avg_loss)

        pbar.close()
    
            
    def train(self, val_per_epoch:int=5):
        for epoch in range(1, self.num_epochs+1):
            print(f"Epoch Train {epoch}/{self.num_epochs}")
            self.train_step()
            print(f'Train Indicator : {self.train_indicator.show_last_info()}')
            if (epoch) % val_per_epoch == 0:
                self.eval_step()
                print(f'Eval Indicator: {self.val_indicator.show_last_info()}')

                if self.val_indicator.f1[-1] >= max(self.val_indicator.f1):
                    torch.save(self.model.state_dict(), f'/home/hyyjs/yangq/open_data_competation/save/best_model_{self.version}.pth')


                # 检查EarlyStopping
                if self.early_stopping:
                    self.early_stopping(self.val_indicator.acc[-1])
                    if self.early_stopping.early_stop:
                        print(f"Early stopping at epoch {epoch}")
                        break
                


            # 更新scheduler
            if self.scheduler:
                if isinstance(self.scheduler, lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(self.val_indicator.f1[-1])  # 对于ReduceLROnPlateau，传入指标
                else:
                    self.scheduler.step()  # 对于StepLR等直接更新

            torch.save(self.model.state_dict(), f'/home/hyyjs/yangq/open_data_competation/save/last_model_{self.version}.pth')


    def update_visualization(self):
        fig = make_subplots(rows=1, cols=2, subplot_titles=('F1 Score', 'Loss'))
        fig.add_trace(go.Scatter(x=list(range(len(self.f1_list))), y=self.f1_list, mode='lines', name='F1 Score'), row=1, col=1)
        fig.add_trace(go.Scatter(x=list(range(len(self.loss_list))), y=self.loss_list,mode='lines', name='Loss'),row=1, col=2)
        fig.update_layout(
            title_text="Training Progress",
            showlegend=True,
            xaxis_title="Epochs",
            yaxis_title="F1 Score",
            xaxis2_title="Batch",
            yaxis2_title="Loss",
            height=600,
            width=1800,
            template="plotly_white"
        )
        if self.is_jupyter:
            clear_output(wait=True)
            display(fig)
    

    def print_indicator(self, show_train_indicator=False, show_val_indicator=False):
        if show_train_indicator:
            self.train_indicator.show()

        if show_val_indicator:
            self.val_indicator.show()

    def plot_indicator(self, show_train_indicator=False, show_val_indicator=False):
        if show_train_indicator:
            self.train_indicator.plot(f'train_{self.version}_{self.train_indicator.fc[-1]}')

        if show_val_indicator:
            self.val_indicator.plot(f'val{self.version}_{self.train_indicator.acc[-1]}')
