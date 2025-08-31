import argparse  # 导入命令行解析库
import os
import torch
import matplotlib.pyplot as plt
import pandas as pd
# import seaborn as sns

import torch.multiprocessing as mp

mp.set_start_method('forkserver', force=True)
torch.multiprocessing.set_sharing_strategy('file_system')



ROOT = os.path.abspath(os.path.curdir)


def get_default_pth_path() -> str:
    return os.path.join(ROOT, 'save', 'best_model_v5.pth')


def get_test_data_path() -> str:
    return os.path.join(ROOT, 'dataset', 'test_dataset')

def get_sub_file_folder(parent_folder) -> str:
    sub_files, sub_folders = [], []

    for subfile in os.listdir(parent_folder):
        subfile_path = os.path.join(parent_folder, subfile)
        if os.path.isfile(subfile_path) and subfile.endswith('.csv'):
            sub_files.append(subfile_path)
        elif os.path.isdir(subfile_path):
            sub_folders.append(subfile_path)

    return sub_files, sub_folders


def build_dataloader(test_folder:str, batch, num_work):
    from build_dataset import MultiModalDataset
    from transformers import AutoTokenizer
    from torchvision import transforms
    from utils import custom_collate_fn
    from torch.utils.data import DataLoader


    # 从test_folder中构建Dataloader
    
    # 遍历指定目录下的子目录 含文件和imgae路径
    xlsx_file_paths, img_folders = get_sub_file_folder(test_folder)

    if len(xlsx_file_paths) != 1 or len(img_folders) != 1:
        raise "指定目录下有多Excel文件或多个目录，不支持"
    
    tokenizer = AutoTokenizer.from_pretrained('/home/hyyjs/yangq/open_data_competation/model')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    dataset = MultiModalDataset(csv_file=xlsx_file_paths[0],img_dir=img_folders[0], tokenizer=tokenizer,transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch, shuffle=True, num_workers=num_work, collate_fn=custom_collate_fn)

    return dataloader

def build_model(check_point_path:str):
    from build_model_v5_0 import MultimodalClassifier
    
    model = MultimodalClassifier(num_classes=2)
    if check_point_path and os.path.exists(check_point_path):
        model.load_state_dict(torch.load(check_point_path))

    return model

def run(dataloader, model):
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    model.to(device)  # Ensure the model is moved to the correct device
    model.eval()
    reals, preds = [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            images = batch['images'].to(device)
            masks = batch['attention_mask'].to(device)
            outputs = model(input_ids, images, masks)
            _, pred_lables = torch.max(outputs, dim=1)
            
            preds.extend(pred_lables.cpu().numpy())
            reals.extend(batch['label'])
    
    return reals, preds

def plot_confusion_matrix(y_true, y_pred):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Predicted 0', 'Predicted 1'], yticklabels=['Actual 0', 'Actual 1'])
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()

def plot_roc_curve(y_true, y_pred):
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()

def plot_pr_curve(y_true, y_pred):
    from sklearn.metrics import precision_recall_curve, auc
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    pr_auc = auc(recall, precision)

    plt.figure()
    plt.plot(recall, precision, color='blue', lw=2, label='PR curve (area = %0.2f)' % pr_auc)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a detector')  # 创建命令行解析器
    parser.add_argument('--ckp', '-P', default=get_default_pth_path(), help='train config file path')  # 添加配置文件路径参数
    parser.add_argument('--data', '-D', default=get_test_data_path(), help='train config file path')  # 添加配置文件路径参数
    parser.add_argument('--batch', '-B', default=4, help='train config file path')  # 添加配置文件路径参数
    parser.add_argument('--numwork', '-N', default=4, help='train config file path')  # 添加配置文件路径参数
    args = parser.parse_args()  # 解析命令行参数

    # 初始化数据集
    dataloader = build_dataloader(args.data, args.batch, args.numwork)
    
    # 初始化模型
    model = build_model(args.ckp)

    # 开始测试
    y_true, y_pred = run(dataloader, model)
    
    # 测试结果可视化
    # plot_confusion_matrix(y_true, y_pred)
    # plot_roc_curve(y_true, y_pred)
    # plot_pr_curve(y_true, y_pred)

    # 测试结束 结果写入csv文件中
    test_csv_file = get_sub_file_folder(args.data)[0][0]
    predDF = pd.read_csv(test_csv_file)
    predDF['target'] = y_pred
    predDF.to_csv(test_csv_file, encoding='utf-8')

