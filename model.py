import torch
import torch.nn as nn
import torch.nn.functional as F


class BCNE_Original(nn.Module):
    def __init__(self, latent_dim=3):
        super(BCNE_Original, self).__init__()

        # 编码器 (Encoder)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # 映射头 (Projection Head / Bottleneck)
        # 这里的 Head 是为了把 128维 压缩到 3维
        # 训练完保留，用于生成可视化坐标
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = self.pool(x)

        # 输出特征和空列表(保持接口兼容)
        return self.head(x), []