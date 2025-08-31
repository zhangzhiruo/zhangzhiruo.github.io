import torch
import torch.nn as nn
from transformers import BertModel, DebertaModel, AutoModelForMaskedLM
from torchvision import models
import torch.nn.functional as F

class CrossModalAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=8):
        super(CrossModalAttention, self).__init__()

        # 将文本和图像特征投影到相同的空间
        self.text_proj = nn.Linear(hidden_dim, hidden_dim)
        self.image_proj = nn.Linear(hidden_dim, hidden_dim)

        # 多头自注意力机制
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads)

        # 输出层
        self.fc_out = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, text_features, image_features):
        """
        :param text_features: (batch_size, hidden_dim)
        :param image_features: (batch_size, hidden_dim)
        :return: (batch_size, hidden_dim)
        """
        # 投影到相同的空间
        text_proj = self.text_proj(text_features).unsqueeze(0)  # (1, batch_size, hidden_dim)
        image_proj = self.image_proj(image_features).unsqueeze(0)  # (1, batch_size, hidden_dim)

        # 多头自注意力
        # 注意力的查询（query）、键（key）、值（value）都来自图像和文本特征
        attn_output, _ = self.attention(text_proj, image_proj, image_proj)  # (1, batch_size, hidden_dim)

        # 输出通过全连接层
        output = self.fc_out(attn_output.squeeze(0))  # (batch_size, hidden_dim)
        return output
    
class AutoFusionV7(nn.Module):
    def __init__(self, text_feature_dim=256, image_feature_dim=128, output_dim=512):
        super(AutoFusionV7, self).__init__()
        
        # 全连接层处理文本特征
        self.text_fc = nn.Linear(text_feature_dim, output_dim // 2)
        
        # 全连接层处理图像特征
        self.image_fc = nn.Linear(image_feature_dim, output_dim // 2)
        
        # MLP用于计算自适应权重
        self.mlp = nn.Sequential(
            nn.Linear(output_dim, output_dim // 4),
            nn.ReLU(),
            nn.Linear(output_dim // 4, 2),
            nn.Softmax(dim=1)
        )
        
        # 最终的全连接层
        self.final_fc = nn.Linear(output_dim, output_dim)
        
    def forward(self, text_features, image_features):
        """
        Args:
            text_features: shape (batch_size, text_feature_dim)
            image_features: shape (batch_size, c, h, w)
        Returns:
            fused_features: shape (batch_size, output_dim)
        """      
        last_features = []

        text_out = F.relu(self.text_fc(text_features)) # batch, 256 >>> batch, 256


        # 将图像特征从 (batch_size, c, h, w) 转换为 (batch_size, image_feature_dim)
        batch_size, N, _, _, _ = image_features.shape

        fused_features = []
        for i in range(N):
            image_feature_i = image_features[:, i, :, :, :]  # batch, channel, height, width = batch, 128, 1, 1
            image_feature_i = image_feature_i.view(image_features.size(0), -1)  # batch, 128*1*1
            image_feature_i = F.relu(self.image_fc(image_feature_i))  # batch,512 ==> batch, 256

            concatenated = torch.cat((text_out, image_feature_i), dim=1) # batch,256+batch,256 = batch,512

            # 计算自适应权重
            weights = self.mlp(concatenated)    # batch,512 >>> batch,2

            # 应用自适应权重
            weighted_text = text_out * weights[:, 0].unsqueeze(1)   # batch,256 * batch, 1, 1 >>> batch,256
            weighted_image = image_feature_i * weights[:, 1].unsqueeze(1)    # batch,256 * batch, 1, 1 >>> batch,256

            # 拼接加权后的特征
            fused_features = torch.cat((weighted_text, weighted_image), dim=1)  # batch,256 + batch,256 = batch,512
            
            # 最终的全连接层
            output = self.final_fc(fused_features)      # batch,512 >>> batch,512


            last_features.append(output)
        
        # 整型一下
        last_features = torch.stack(last_features, dim=1)  # batch, img_num, hidden_dim=512
        last_features = torch.mean(last_features, dim=1)  # batch, hidden_dim=512

        
        return last_features

from transformers import AutoModel
class TextEncoderV7(nn.Module):
    def __init__(self, num_heads=8, hidden_dim=512, output_dim=256):
        super(TextEncoderV7, self).__init__()
        # 使用 DeBERTa Backbone
        self.deberta=AutoModel.from_pretrained('/home/hyyjs/yangq/open_data_competation/model2')
        # model_path = "/home/hyyjs/yangq/open_data_competation/model2/pytorch_model.bin"
        # state_dict = torch.load(model_path, map_location="cpu")
        # self.deberta = AutoModelForMaskedLM.from_pretrained(pretrained_model_name_or_path="None", config="/home/hyyjs/yangq/open_data_competation/model2/config.json", state_dict=state_dict)


        # 冻结 DeBERTa 的前几层参数（根据任务调整层数）
        for param in self.deberta.encoder.layer[:6].parameters():  # 冻结前 6 层
            param.requires_grad = False

        # 将 DeBERTa 的输出映射到隐藏维度
        self.deberta_to_hidden = nn.Linear(self.deberta.config.hidden_size, hidden_dim)

        # 多头注意力
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads)

        # 残差连接和归一化
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)

        # 前馈网络
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 最终将隐藏状态映射到输出维度
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, mask=None):
        # 获取 DeBERTa 的输出
        with torch.no_grad():
            deberta_outputs = self.deberta(input_ids=x, attention_mask=mask)

        # 提取最后一层的隐藏状态
        last_hidden_state = deberta_outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)

        # 映射到隐藏维度
        last_hidden_state = self.deberta_to_hidden(last_hidden_state)  # (batch_size, seq_len, hidden_dim)

        # 使用多头注意力机制
        attn_output, _ = self.attention(
            last_hidden_state.permute(1, 0, 2),  # 变换维度以适配 MultiheadAttention (seq_len, batch_size, hidden_dim)
            last_hidden_state.permute(1, 0, 2),
            last_hidden_state.permute(1, 0, 2)
        )
        attn_output = attn_output.mean(dim=0)  # 在序列维度求均值 (batch_size, hidden_dim)

        # 残差连接
        residual = self.layer_norm1(attn_output + last_hidden_state.mean(dim=1))  # 残差加权 (batch_size, hidden_dim)

        # 前馈网络和第二次残差连接
        feed_forward_output = self.feed_forward(residual)
        features = self.fc(self.layer_norm2(feed_forward_output + residual))  # 最终输出特征 (batch_size, output_dim)

        return features


class SEBlock(nn.Module):
    def __init__(self, input_dim, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim // reduction)
        self.fc2 = nn.Linear(input_dim // reduction, input_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.mean(dim=(2, 3))  
        x = self.fc1(x)         
        x = self.relu(x)
        x = self.fc2(x)         
        x = self.sigmoid(x).unsqueeze(2).unsqueeze(3)
        return x

from transformers import ViTImageProcessor, ViTForImageClassification

class ImageEncoderV7(nn.Module):
    def __init__(self, freeze=True, output_dim=128):
        super(ImageEncoderV7, self).__init__()
        # 使用 timm 加载 ViT 模型，不加载预训练权重
        self.backbone = ViTForImageClassification.from_pretrained('/home/hyyjs/yangq/open_data_competation/modelVit')


        # 加载指定的预训练权重文件
        # self.backbone.load_state_dict(torch.load(r'/home/hyyjs/yangq/open_data_competation/model/vit_b_16-c867db91.pth'), strict=True)

        # 冻结或解冻模型参数
        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # 修改最后的分类头为恒等映射
        self.backbone.classifier = nn.Identity()

        # 注意：ViT 输出特征维度为 768 (对于 vit_base_patch16_224)
        self.se_block = SEBlock(input_dim=768)
        self.fc = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, output_dim)

    def forward(self, image_features):
        """
        前向传播函数，用于处理输入的图像特征并生成融合后的图像特征。
        
        参数:
            image_features (torch.Tensor): 输入的图像特征张量，形状为 (batch_size, N, c, h, w)。
                                           其中 batch_size 是批次大小，N 是每个样本中的图像数量，
                                           c 是通道数，h 和 w 分别是高度和宽度。
                                           
        返回:
            torch.Tensor: 融合后的图像特征张量，形状为 (batch_size, N, ..., 1, 1)。
                          最终维度取决于 `fc2` 层的输出。
        """
        batch_size, N, c, h, w = image_features.shape

        fusion_image_features = []
        for i in range(N):
            # 提取当前图像的特征，形状为 (batch_size, c, h, w)
            image_feature_i = image_features[:, i, :, :, :]

            # 使用 ViT 模型进行特征提取
            outputs = self.backbone.vit(image_feature_i)  # 输出形状为 (batch_size, 768)

            features = outputs.last_hidden_state[:, 0, :]  # 形状为 (batch_size, 768)

            # 将特征展平到 (batch_size, 768, 1, 1)，以准备送入 SE 模块
            features = features.unsqueeze(-1).unsqueeze(-1)

            # 通过 SE（Squeeze-and-Excitation）模块进行自注意力加权
            se_weighted_features = self.se_block(features)

            # 将 SE 加权后的特征展平成 (batch_size, -1)，以便送入全连接层
            se_weighted_features = se_weighted_features.view(se_weighted_features.size(0), -1)

            # 通过第一个全连接层，将特征映射到中间维度
            output_fc = self.fc(se_weighted_features)

            # 通过第二个全连接层，并恢复原始的空间维度 (1, 1)
            output = self.fc2(output_fc).unsqueeze(2).unsqueeze(3)

            # 将当前图像的输出添加到结果列表中
            fusion_image_features.append(output)

        # 将所有图像的结果堆叠在一起，形成最终的输出张量
        return torch.stack(fusion_image_features, dim=1)

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

        self.text_encoder = TextEncoderV7()
        self.image_encoder = ImageEncoderV7()
        self.fusion_module = AutoFusionV7()
        self.classifier = ImprovedClassifier(input_dim=512, hidden_dim=128, output_dim=num_classes)

    def forward(self, input_ids, images, text_mask=None, image_mask=None):
        text_features = self.text_encoder(input_ids, mask=text_mask)
        image_features = self.image_encoder(images)
        fused_features = self.fusion_module(text_features, image_features)
        logits = self.classifier(fused_features)
        return logits
