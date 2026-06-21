import torch.nn as nn
import torch

class Couple_CNN(nn.Module):
    def __init__(self, input_channel):
        super(Couple_CNN, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(input_channel, 32, kernel_size=3, stride=1, padding=1,
                      bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2)
        )

        self.block2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1,
                                              bias=False),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(),
                                    )
        self.block3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1,
                                              bias=False),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU(),
                                    # nn.MaxPool2d(kernel_size=2)
                                    )

        for m in self.modules():
            torch.manual_seed(couple_seed)
            torch.cuda.manual_seed(couple_seed)
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x


class fusion_module(nn.Module):
    def __init__(self, input_channel):
        super().__init__()
        self.block_5_1 = nn.Sequential(nn.Conv2d(input_channel, 128, kernel_size=3, stride=1, padding=1,
                                                 bias=False),
                                       nn.BatchNorm2d(128),
                                       nn.ReLU(),
                                       )

        self.block_5_2 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1,
                                                 bias=False),
                                       nn.BatchNorm2d(64),
                                       nn.ReLU(),
                                       )

    def forward(self, x):
        x = self.block_5_1(x)
        x = self.block_5_2(x)
        return x


def modality_drop(data, p):
    B = data[0].size(0)
    num_modalities = len(data)

    # 创建形状为 [B, num_modalities, 1, 1, 1] 的掩码张量
    p_tensor = torch.tensor(p, device='cuda', dtype=torch.float32)
    p_tensor = p_tensor.view(1, -1).expand(B, num_modalities)
    p_tensor = p_tensor.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

    # 对每个模态应用掩码
    for i in range(num_modalities):
        data[i] = data[i] * p_tensor[:, i]

    return data, p_tensor


class ShaSpec_Classfication(nn.Module):
    def __init__(self, args, modality_1_channel, modality_2_channel):
        super().__init__()

        self.modality_encoder_1 = Couple_CNN(modality_1_channel)
        self.modality_encoder_2 = Couple_CNN(modality_2_channel)
        channel_list = [modality_1_channel, modality_2_channel]
        max_channel_index = channel_list.index(max(channel_list))
        self.min_channel_index = channel_list.index(min(channel_list))

        self.share_encoder = Couple_CNN(channel_list[max_channel_index])
        self.modality_projector = torch.nn.Conv2d(channel_list[self.min_channel_index], channel_list[max_channel_index],
                                                  1, 1, 0)
        self.fusion_projector = nn.Conv2d(256, 128, 1, 1, 0)
        self.fusion = fusion_module(256)
        self.modality_2_transfer = fusion_module(128)
        self.target_classifier = nn.Linear(64, args.class_num)
        self.unimodal_target_classifier = nn.Linear(128, args.class_num)
        self.modality_classifier = nn.Linear(128, 2)
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.p = args.p
        self.args = args

    def forward(self, modality1, modality2):
        m1_feature_specific = self.modality_encoder_1(modality1)
        m2_feature_specific = self.modality_encoder_2(modality2)

        m1_feature_specific_cache = m1_feature_specific
        m2_feature_specific_cache = m2_feature_specific

        if self.min_channel_index == 0:
            modality1_transfer = self.modality_projector(modality1)
            m1_feature_share = self.share_encoder(modality1_transfer)
            m2_feature_share = self.share_encoder(modality2)
        elif self.min_channel_index == 1:
            modality_transfer = self.modality_projector(modality2)
            m2_feature_share = self.share_encoder(modality_transfer)
            m1_feature_share = self.share_encoder(modality1)
        else:
            raise ValueError

        m1_feature_share_cache = m1_feature_share
        m2_feature_share_cache = m2_feature_share


        m1_missing_index = [not bool(i) for i in (torch.sum(m1_feature_share, dim=[1, 2, 3]))]
        m1_feature_share[m1_missing_index] = m2_feature_share[m1_missing_index]

        m2_missing_index = [not bool(i) for i in (torch.sum(m2_feature_share, dim=[1, 2, 3]))]
        m2_feature_share[m2_missing_index] = m1_feature_share[m2_missing_index]

        # fusion

        m1_fusion_out = self.fusion_projector(torch.cat((m1_feature_specific, m1_feature_share), dim=1))
        m1_fusion_out = m1_fusion_out + m1_feature_specific

        m2_fusion_out = self.fusion_projector(torch.cat((m2_feature_specific, m2_feature_share), dim=1))
        m2_fusion_out = m2_fusion_out + m2_feature_specific

        fusion_feature = self.fusion(torch.cat([m1_fusion_out, m2_fusion_out], dim=1))

        fusion_feature = self.pooling(fusion_feature)
        fusion_feature = fusion_feature.view(fusion_feature.shape[0], -1)

        # calculate loss

        target_predict = self.target_classifier(fusion_feature)

        m1_feature_specific_cache_pooling = self.pooling(m1_feature_specific_cache)
        m2_feature_specific_cache_pooling = self.pooling(m2_feature_specific_cache)

        m1_feature_specific_cache_pooling = m1_feature_specific_cache_pooling.view(
            m1_feature_specific_cache_pooling.shape[0],
            -1)
        m2_feature_specific_cache_pooling = m2_feature_specific_cache_pooling.view(
            m2_feature_specific_cache_pooling.shape[0],
            -1)

        specific_feature = torch.cat((m1_feature_specific_cache_pooling, m2_feature_specific_cache_pooling), dim=0)
        specific_feature_label = torch.cat(
            [torch.zeros(m1_feature_specific_cache_pooling.shape[0]),
             torch.ones(m2_feature_specific_cache_pooling.shape[0])], dim=0).long().cuda()

        dco_predict = self.modality_classifier(specific_feature)

        m1_predict = self.unimodal_target_classifier(m1_feature_specific_cache_pooling)
        m2_predict = self.unimodal_target_classifier(m2_feature_specific_cache_pooling)

        return target_predict, dco_predict, specific_feature_label, m1_feature_share_cache, m2_feature_share_cache, m1_predict, m2_predict