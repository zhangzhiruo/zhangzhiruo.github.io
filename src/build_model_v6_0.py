import torch
import torch.nn as nn
from transformers import BertModel
from torchvision import models


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
    

class FeatureFusionV51(nn.Module):
    def __init__(self, image_feature_dim=128*1*1, text_feature_dim=256, hidden_dim=512):
        super(FeatureFusionV51, self).__init__()
        self.fc_text = nn.Linear(text_feature_dim, hidden_dim)
        self.fc_image = nn.Linear(image_feature_dim, hidden_dim)

        # 跨模态注意力机制
        self.cross_attention = CrossModalAttention(hidden_dim)

        self.weight_fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, text_features, image_features):
        batch_size, N, _, _, _ = image_features.shape

        # 提取图像特征
        fused_features = []
        for i in range(N):
            image_feature_i = image_features[:, i, :, :, :]
            image_feature_i = image_feature_i.view(image_features.size(0), -1)
            image_feature_i = self.fc_image(image_feature_i)
            fused_features.append(image_feature_i)

        fused_features = torch.stack(fused_features, dim=1)  # batch, img_num, hidden_dim
        fused_features = torch.mean(fused_features, dim=1)  # batch, hidden_dim

        # 提取文本特征
        text_proj = self.fc_text(text_features)  # batch, hidden_dim

        # 跨模态注意力融合
        attention_features = self.cross_attention(text_proj, fused_features)

        # 动态生成权重
        concat_features = torch.cat((text_proj, fused_features), dim=1)
        weight = self.weight_fc(concat_features)

        # 自适应加权融合
        fused_features = weight * text_proj + (1 - weight) * attention_features

        return fused_features


class TextEncoderV5(nn.Module):
    def __init__(self, num_heads=8, hidden_dim=512, output_dim=256):
        super(TextEncoderV5, self).__init__()
        self.bert = BertModel.from_pretrained('/home/hyyjs/yangq/open_data_competation/model')

        for param in self.bert.encoder.layer[:6].parameters():
            param.requires_grad = False

        self.bert_to_hidden = nn.Linear(768, hidden_dim)

        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads)

        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)

        self.fc = nn.Linear(hidden_dim, output_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x, mask=None):
        with torch.no_grad():
            bert_outputs = self.bert(input_ids=x, attention_mask=mask)

        last_hidden_state = bert_outputs.last_hidden_state

        last_hidden_state = self.bert_to_hidden(last_hidden_state)

        attn_output, _ = self.attention(last_hidden_state.permute(1, 0, 2), last_hidden_state.permute(1, 0, 2), last_hidden_state.permute(1, 0, 2))
        attn_output = attn_output.mean(dim=0)

        residual = self.layer_norm1(attn_output + last_hidden_state.mean(dim=1))
        feed_forward_output = self.feed_forward(residual)
        features = self.fc(self.layer_norm2(feed_forward_output + residual))

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


class ImageEncoderV5(nn.Module):
    def __init__(self, freeze=True, output_dim=128):
        super(ImageEncoderV5, self).__init__()
        self.resnet_features = models.resnet50(pretrained=False)
        self.resnet_features.load_state_dict(torch.load('/home/hyyjs/yangq/open_data_competation/model/resnet50-19c8e357.pth'))

        for param in self.resnet_features.layer1.parameters():
            param.requires_grad = False
        self.resnet_features.fc = nn.Identity()

        self.se_block = SEBlock(input_dim=2048)
        self.fc = nn.Linear(2048, 512)
        self.fc2 = nn.Linear(512, output_dim)

    def forward(self, image_features):
        batch_size, N, c, h, w = image_features.shape

        fusion_image_features = []
        for i in range(N):
            image_feature_i = image_features[:, i, :, :, :]
            image_feature_i = self.resnet_features(image_feature_i)
            image_feature_i = image_feature_i.unsqueeze(2).unsqueeze(3)

            se_weighted_features = self.se_block(image_feature_i)

            se_weighted_features = se_weighted_features.view(se_weighted_features.size(0), -1)

            output_fc = self.fc(se_weighted_features)

            output = self.fc2(output_fc).unsqueeze(2).unsqueeze(3)

            fusion_image_features.append(output)

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

        self.text_encoder = TextEncoderV5()
        self.image_encoder = ImageEncoderV5()
        self.fusion_module = FeatureFusionV51()
        self.classifier = ImprovedClassifier(input_dim=512, hidden_dim=128, output_dim=num_classes)

    def forward(self, input_ids, images, text_mask=None, image_mask=None):
        text_features = self.text_encoder(input_ids, mask=text_mask)
        image_features = self.image_encoder(images)
        fused_features = self.fusion_module(text_features, image_features)
        logits = self.classifier(fused_features)
        return logits
