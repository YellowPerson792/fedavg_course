"""
模型动物园：所有客户端 + 服务器都用这里的 build_model 拿到结构相同的实例。

为什么模型放在共享模块里：
  FedAvg 聚合要求所有客户端使用"完全一致的网络结构"——参数张量名、形状、dtype
  必须逐一对齐，否则 fedavg() 里 zip(states, ...) 跨客户端按 key 累加就会出错。
  把构造逻辑收敛到一个 build_model() 是最稳妥的做法。

针对 Pi 4 的算力做了"小而精"的取舍：
  - tinycnn_mnist : 体积小、收敛快，作为基线做完整对比矩阵。
  - dscnn_cifar   : 用 depthwise-separable 卷积大幅压参，CIFAR 上仍跑得动。
  - simplecnn_cifar: 朴素 VGG 风格，参数较多，作为对比基线。
  - squeezenet_cifar / mobilenetv3_cifar / resnet18_cifar:
    torchvision 模型改造成 CIFAR-10 输入，用于树莓派效率对比。
"""

from __future__ import annotations

import torch
from torch import nn


class TinyCNNMnist(nn.Module):
    """MNIST 用迷你 CNN：2× (Conv→BN→ReLU→MaxPool) + 2× FC，约 ~10 万参数。"""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        # 特征提取：28×28×1 → 14×14×16 → 7×7×32
        # padding=1 + kernel=3：保持空间尺寸，仅靠 MaxPool 降采样，便于推算输出 shape。
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),                # 加速收敛；FedAvg 下 BN 统计也跟权重一起被平均
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                   # 28→14
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                   # 14→7
        )
        # 分类头：扁平化 → 64 → num_classes (10)
        # 32*7*7 = 1568 是上面卷积输出的展平维度。
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 标准两段式：先抽特征，再分类。返回 raw logits，由外面的 CrossEntropyLoss 处理 softmax。
        return self.classifier(self.features(x))


class DepthwiseSeparableBlock(nn.Module):
    """深度可分离卷积块：把"标准 3×3 卷积"拆成 depthwise + pointwise，参数量降一个量级。

    标准 conv 参数量 ≈ Cin * Cout * 9
    DW-Sep    参数量 ≈ Cin * 9 + Cin * Cout
    Cout 越大省得越多，是 MobileNet 系列的核心 trick，特别适合 Pi 这种算力受限设备。
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            # depthwise：每个输入通道独立卷一个 3×3 (groups=in_channels 实现"分组到通道")
            nn.Conv2d(in_channels, in_channels, 3, stride=stride, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            # pointwise：1×1 卷积做跨通道线性组合，把 in_channels 映射到 out_channels。
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DSCNNCifar(nn.Module):
    """Depthwise-Separable CNN for CIFAR-10：MobileNet 风格的轻量结构，适合 Pi。"""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            # 入口仍用普通 conv，让通道翻起来 (3→24)，后续再用 DW-Sep 块。
            nn.Conv2d(3, 24, 3, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            DepthwiseSeparableBlock(24, 48, stride=1),  # 32×32×48
            nn.MaxPool2d(2),                            # 32→16
            DepthwiseSeparableBlock(48, 96, stride=1),  # 16×16×96
            nn.MaxPool2d(2),                            # 16→8
            DepthwiseSeparableBlock(96, 128, stride=1), # 8×8×128
            # 全局平均池化把任意空间尺寸压到 1×1，再接 FC，比 Flatten+大 FC 省参数。
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        # x.shape = (B, 128, 1, 1)，flatten 成 (B, 128) 喂给最后的线性层。
        return self.classifier(torch.flatten(x, 1))


class SimpleCNNCifar(nn.Module):
    """对照基线：朴素 VGG 风格 4 层卷积 CNN，参数量大但实现直观。"""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.net = nn.Sequential(
            # 第 1 段：32×32×3 → 32×32×32 → 32×32×32 → 16×16×32
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # 第 2 段：16×16×32 → 16×16×64 → 16×16×64 → 8×8×64
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # 分类头：64*8*8 = 4096 → 128 → num_classes
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SqueezeNetCifar(nn.Module):
    """SqueezeNet 1.0 adapted for CIFAR-10 (32x32), about 1.2M parameters."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        import torchvision

        model = torchvision.models.squeezenet1_0(weights=None)
        model.features[0] = nn.Conv2d(3, 96, kernel_size=3, stride=1, padding=1)
        model.features[2] = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=1)
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.flatten(self.model(x), 1)


class MobileNetV3SmallCifar(nn.Module):
    """MobileNetV3-Small adapted for CIFAR-10 (32x32), about 2.5M parameters."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        import torchvision

        model = torchvision.models.mobilenet_v3_small(weights=None)
        old_conv = model.features[0][0]
        model.features[0][0] = nn.Conv2d(
            old_conv.in_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=1,
            padding=old_conv.padding,
            bias=old_conv.bias,
        )
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class ResNet18Cifar(nn.Module):
    """ResNet-18 adapted for CIFAR-10 (32x32), about 11.7M parameters."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        import torchvision

        model = torchvision.models.resnet18(weights=None)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def build_model(name: str) -> nn.Module:
    """按配置 (config['model']) 字符串实例化模型。

    服务器和客户端都通过这个工厂函数拿模型，确保结构完全一致——
    这是 FedAvg 加权平均能逐 key 对齐的前提。
    """
    name = name.lower()
    if name == "tinycnn_mnist":
        return TinyCNNMnist()
    if name == "dscnn_cifar":
        return DSCNNCifar()
    if name == "simplecnn_cifar":
        return SimpleCNNCifar()
    if name == "squeezenet_cifar":
        return SqueezeNetCifar()
    if name == "mobilenetv3_cifar":
        return MobileNetV3SmallCifar()
    if name == "resnet18_cifar":
        return ResNet18Cifar()
    raise ValueError(f"unknown model: {name}")
