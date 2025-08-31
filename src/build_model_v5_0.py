import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from transformers import BertModel


class FeatureFusionV5(nn.Module):
    def __init__(self, image_feature_dim=128*1*1, text_feature_dim=256, hidden_dim=512):
        super(FeatureFusionV5, self).__init__()
        self.weight = nn.Parameter(torch.tensor(0.5))


        self.text_conv = nn.Conv1d(text_feature_dim, hidden_dim, kernel_size=1)  
        self.image_fc = nn.Linear(image_feature_dim, hidden_dim)
        self.fc = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim *2 ),  
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )


    def forward(self, text_features, image_features):
        batch_size, N, _, _, _ = image_features.shape

        """
        遍历每一张图片的特征，为每一张图片特征展平并映射至和text相同维度特征
        """
        fused_features = []
        for i in range(N):
            image_feature_i = image_features[:, i, :, :, :]  
            image_feature_i = image_feature_i.view(image_features.size(0), -1)  
            image_feature_i = self.image_fc(image_feature_i)  

            fused_features.append(image_feature_i)

        """
        将N张图片的特征堆叠并在图像张数上计算均值
        """
        fused_features = torch.stack(fused_features, dim=1)  
        fused_features = torch.mean(fused_features, dim=1)  

        """
        扁平化文本特征并进行卷积
        """
        text_features = text_features.view(text_features.size(0), -1)
        text_features = self.text_conv(text_features.unsqueeze(2)).squeeze(2)

        """
        自适应加权融合，根据权重参数self.weight融合文本和图像特征
        """
        fused_features = self.weight * text_features + (1 - self.weight) * fused_features  

        """
        拼接后进行全连接处理
        """
        return self.fc(fused_features)


class TextEncoderV5(nn.Module):
    def __init__(self, num_heads=8, hidden_dim=512, output_dim=256):
        super(TextEncoderV5, self).__init__()
        self.bert = BertModel.from_pretrained('/home/hyyjs/yangq/open_data_competation/model')

        for param in self.bert.parameters():
            param.requires_grad = False

        self.bert_to_hidden = nn.Linear(768, hidden_dim)
        
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads)
        
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, mask=None):
        """
        通过冻结参数的预训练BERT模型对token文本编码
        """
        with torch.no_grad():
            bert_outputs = self.bert(input_ids=x, attention_mask=mask)

        """
        将BERT的输出的特征降维至全连接层
        """
        last_hidden_state = self.bert_to_hidden(bert_outputs.last_hidden_state)  

        """
        通过注意力机制在序列维度上计算均值
        """
        attn_output, _ = self.attention(last_hidden_state.permute(1, 0, 2), last_hidden_state.permute(1, 0, 2), last_hidden_state.permute(1, 0, 2))
        attn_output = attn_output.mean(dim=0)

        """
        将注意力和原始特征相加得到残差模块，并通过全连接层归一处理
        """
        residual = self.layer_norm1(attn_output + last_hidden_state.mean(dim=1))  
        features = self.fc(self.layer_norm2(residual))

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
        self.resnet_features = models.resnet18(pretrained=False)
        self.resnet_features.load_state_dict(torch.load('/home/hyyjs/yangq/open_data_competation/model/resnet18-f37072fd.pth'))

        
        for param in self.resnet_features.parameters():
            param.requires_grad = False
        self.resnet_features.fc = nn.Identity()  

        
        self.se_block = SEBlock(input_dim=512)

        
        self.fc = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, output_dim)

    def forward(self, image_features):
        batch_size, N, c, h, w = image_features.shape

        """
        为每一张图片通过冻结参数且取消最后一层的ResNet50进行特征提取
        过 SE（Squeeze-and-Excitation）模块进行自注意力加权
        并通过全连接层恢复原始空间维度
        
        最终堆叠所有结果返回张量
        """
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
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.fc(x)
    

class MultimodalClassifier(nn.Module):
    def __init__(self, num_classes):
        super(MultimodalClassifier, self).__init__()
        
        
        self.text_encoder = TextEncoderV5()
        
        
        self.image_encoder = ImageEncoderV5()
        
        
        self.fusion_module = FeatureFusionV5()
        
        
        self.classifier = ImprovedClassifier(input_dim=512, hidden_dim=128, output_dim=num_classes)

    def forward(self, input_ids, images, text_mask=None, image_mask=None):
        
        text_features = self.text_encoder(input_ids, mask=text_mask)  

        
        image_features = self.image_encoder(images)  

        
        fused_features = self.fusion_module(text_features, image_features)  
        
        logits = self.classifier(fused_features) 
        return logits
