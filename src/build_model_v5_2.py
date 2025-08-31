import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from transformers import BertModel


class FeatureFusionV51(nn.Module):
    def __init__(self, image_feature_dim=128*1*1, text_feature_dim=256, hidden_dim=512):
        super(FeatureFusionV51, self).__init__()
        self.fc_text = nn.Linear(text_feature_dim, hidden_dim)
        self.fc_image = nn.Linear(image_feature_dim, hidden_dim)
        self.weight_fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, text_features, image_features):
        batch_size, N, _, _, _ = image_features.shape

        fused_features = []
        for i in range(N):
            image_feature_i = image_features[:, i, :, :, :]  # batch, channel, height, width = batch, 128, 1, 1
            image_feature_i = image_feature_i.view(image_features.size(0), -1)  # batch, 128*1*1
            image_feature_i = self.fc_image(image_feature_i)  # batch, hidden_dim=512

            fused_features.append(image_feature_i)

        fused_features = torch.stack(fused_features, dim=1)  # batch, img_num, hidden_dim=512
        fused_features = torch.mean(fused_features, dim=1)  # batch, hidden_dim=512

        # 特征映射
        text_proj = self.fc_text(text_features) # batch, 512

        # 动态生成权重
        concat_features = torch.cat((text_proj, fused_features), dim=1)
        weight = self.weight_fc(concat_features)

        # 自适应加权融合
        fused_features = weight * text_proj + (1 - weight) * fused_features

        return fused_features # 应该也是batch, 512

class TextEncoderV5(nn.Module):
    def __init__(self, num_heads=8, hidden_dim=512, output_dim=256):
        super(TextEncoderV5, self).__init__()
        self.bert = BertModel.from_pretrained('/home/hyyjs/yangq/open_data_competation/model')

        # 冻结 BERT 参数
        for param in self.bert.encoder.layer[:6].parameters():  # 冻结前6层
            param.requires_grad = False

        # 映射到 hidden_dim
        self.bert_to_hidden = nn.Linear(768, hidden_dim)
        
        # 多头注意力
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads)
        
        # 残差连接和归一化
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)

        # 最终映射到 output_dim
        self.fc = nn.Linear(hidden_dim, output_dim)

        # 增加前馈层
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )


    def forward(self, x, mask=None):
        with torch.no_grad():
            bert_outputs = self.bert(input_ids=x, attention_mask=mask)

        # 获取 BERT 输出并映射到 hidden_dim
        last_hidden_state = bert_outputs.last_hidden_state  # (batch_size, seq_len, 768)

        last_hidden_state = self.bert_to_hidden(last_hidden_state)  # (batch_size, seq_len, hidden_dim=512)


        # 使用多头注意力
        attn_output, _ = self.attention(last_hidden_state.permute(1, 0, 2), last_hidden_state.permute(1, 0, 2), last_hidden_state.permute(1, 0, 2))
        attn_output = attn_output.mean(dim=0)  # 计算序列维度上的均值 (batch_size, hidden_dim=512)

        # 残差连接
        # 确保维度匹配
        residual = self.layer_norm1(attn_output + last_hidden_state.mean(dim=1))  # 对所有序列进行均值池化后与注意力输出进行相加
        feed_forward_output = self.feed_forward(residual)
        features = self.fc(self.layer_norm2(feed_forward_output + residual))
        
        return features


def freeze_model_params(model):
    for param in model.parameters():
        param.requires_grad = False

class SEBlock(nn.Module):
    def __init__(self, input_dim, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim // reduction)  # input_dim // reduction
        self.fc2 = nn.Linear(input_dim // reduction, input_dim)  # back to input_dim
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Squeeze
        x = x.mean(dim=(2, 3))  # Global Average Pooling (batch_size, channels)
        x = self.fc1(x)         # (batch_size, channels // reduction)
        x = self.relu(x)
        x = self.fc2(x)         # (batch_size, channels)
        x = self.sigmoid(x).unsqueeze(2).unsqueeze(3)  # (batch_size, channels, 1, 1)
        return x

class ImageEncoderV5(nn.Module):
    def __init__(self, freeze=True, output_dim=128):
        super(ImageEncoderV5, self).__init__()
        self.resnet_features = models.resnet50(pretrained=False)
        self.resnet_features.load_state_dict(torch.load('/home/hyyjs/yangq/open_data_competation/model/resnet50-19c8e357.pth'))  # 加载预训练的ResNet-18模型权重

        # 仅冻结前几层
        for param in self.resnet_features.layer1.parameters():
            param.requires_grad = False
        self.resnet_features.fc = nn.Identity()  # 去掉全连接层

        # Squeeze-and-Excitation (SE)模块
        self.se_block = SEBlock(input_dim=2048) # ResNet50 output is 2048

        # 加入额外卷积层调整输出通道
        self.fc = nn.Linear(2048, 512)  # ResNet50 output is 2048
        self.fc2 = nn.Linear(512, output_dim)


    def forward(self, image_features):
        batch_size, N, c, h, w = image_features.shape

        fusion_image_features = []
        for i in range(N):
            image_feature_i = image_features[:, i, :, :, :]  # batch, channel, height, width = batch, 3, 224, 224
            image_feature_i = self.resnet_features(image_feature_i) # batch, 2048

            image_feature_i = image_feature_i.unsqueeze(2).unsqueeze(3)  # [batch, 512, 1, 1]

            se_weighted_features = self.se_block(image_feature_i)  # [batch, 512, 1, 1]

            se_weighted_features = se_weighted_features.view(se_weighted_features.size(0), -1)  # [batch, 512]

            output_fc = self.fc(se_weighted_features)  # [batch, 256]

            output = self.fc2(output_fc).unsqueeze(2).unsqueeze(3)  # [batch, 128, 1, 1]

            fusion_image_features.append(output)

        
        return torch.stack(fusion_image_features, dim=1)  # batch, img_num, 128, 1, 1


class ImprovedClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ImprovedClassifier, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x):
        return self.fc(x)
    
class MultimodalClassifier(nn.Module):
    def __init__(self, num_classes):
        super(MultimodalClassifier, self).__init__()
        
        # 文本编码器
        self.text_encoder = TextEncoderV5()
        
        # 图像编码器
        self.image_encoder = ImageEncoderV5()
        
        # 特征融合层
        self.fusion_module = FeatureFusionV51()
        
        # 新增的先进分类器模块
        self.classifier = ImprovedClassifier(input_dim=512, hidden_dim=128, output_dim=num_classes)

    def forward(self, input_ids, images, text_mask=None, image_mask=None):
        # 文本特征提取
        text_features = self.text_encoder(input_ids, mask=text_mask)  # [batch_size, 256]

        # 图像特征提取
        image_features = self.image_encoder(images)  # batch_size, img_num, 128, 1, 1

        # 多模态特征融合
        # print(text_features.shape) # batch_size, 256
        # print(image_features.shape) # batch_size, img_num, 128, 1, 1
        fused_features = self.fusion_module(text_features, image_features)  #  batch_size, 512

        # 分类层
        logits = self.classifier(fused_features) # batch, 2
        return logits
