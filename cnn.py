import timm
import torch
import torch.nn as nn

from timm.models.adaptive_avgmax_pool import SelectAdaptivePool2d


class EfficientNet(nn.Module):
    def __init__(self):
        super(EfficientNet, self).__init__()
        model_name = "tf_efficientnet_b0"
        pretrained = True
        input_channels = 1
        pool_type = "avg"
        drop_connect_rate = 0.2
        num_classes = 2

        backbone = timm.create_model(
            model_name=model_name,
            pretrained=pretrained,
            in_chans=input_channels,
            drop_connect_rate=drop_connect_rate,
        )
        self.conv_stem = backbone.conv_stem
        self.bn1 = backbone.bn1
        self.act1 = backbone.act1

        for i in range(len((backbone.blocks))):
            setattr(self, "block{}".format(str(i)), backbone.blocks[i])

        self.conv_head = backbone.conv_head
        self.bn2 = backbone.bn2
        self.act2 = backbone.act2
        self.global_pool = SelectAdaptivePool2d(pool_type=pool_type)
        self.num_features = backbone.num_features * self.global_pool.feat_mult()
        self.fc = nn.Linear(self.num_features, num_classes)
        del backbone

    def _features(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.block0(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.conv_head(x)
        x = self.bn2(x)
        x = self.act2(x)
        return x

    def forward(self, x):
        x = self._features(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        logits = self.fc(x)
        return logits