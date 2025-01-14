import torch
import torch.nn as nn
from torchvision.models import resnet18


class FlexibleResNet(nn.Module):
    def __init__(self, num_classes):
        super(FlexibleResNet, self).__init__()
        # ResNet18 をベースにするが、事前学習済みウェイトは明示的に指定
        base_model = resnet18(weights=None)  # 警告を避けるために weights=None を指定

        # 最初の層を3x3に変更 ( ストライド 2 を 1 に )
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)

        # ResNet の中間層を再利用
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4

        # Global Average Pooling と新しい分類層
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

        # 重みの初期化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
